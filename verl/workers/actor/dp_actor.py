# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_self_distillation_loss,
    compute_va_high_group_mask,
    compute_va_rollout_weights,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.trainer.ppo.sa_opd import compute_sa_filter_mask
from verl.trainer.ppo.rank_tie import rank_tie_token_loss
from verl.trainer.ppo.ssp_opd import compute_ssp_listwise_loss, compute_ssp_v2_listwise_loss
from verl.trainer.ppo.vzero import compute_vzero_evidence_weights, compute_vzero_trajectory_scores
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_, load_fsdp_optimizer, offload_fsdp_optimizer
from verl.utils.logit_reduction import chunked_logsumexp, chunked_topk, select_response_logits
from verl.utils.gradient_sync import gradient_sync_context
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.padded_forward import trim_padded_micro_batch, restore_response_width
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.optimizer_step_only_offload = bool(
            self.config.fsdp_config.get("optimizer_offload", False)
            and self.config.fsdp_config.get("optimizer_step_only_offload", False)
        )
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> Optional[float]:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                freeze_after = getattr(self_distillation_cfg, "teacher_ema_freeze_after_step", None)
                if freeze_after is not None:
                    # Called after the optimizer: include the step-N EMA update,
                    # then hold these Teacher weights for steps N+1 onward.
                    if self._current_global_steps > freeze_after:
                        update_rate = 0.0
                if update_rate == 0.0:
                    return 0.0
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    if self.config.get("train_vision_only", False) and not student_param.requires_grad:
                        continue
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return float(update_rate)

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    @staticmethod
    def _build_topk_union_indices(*topk_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not topk_indices:
            raise ValueError("At least one top-k tensor is required to build a union support.")
        reference_shape = topk_indices[0].shape[:-1]
        for tensor in topk_indices:
            if tensor.shape[:-1] != reference_shape:
                raise ValueError(
                    "All top-k tensors must have the same leading shape, got "
                    f"{[tuple(item.shape) for item in topk_indices]}"
                )

        stacked = torch.cat(topk_indices, dim=-1)
        sorted_indices, _ = torch.sort(stacked, dim=-1)
        unique_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
        unique_mask[..., 1:] = sorted_indices[..., 1:] != sorted_indices[..., :-1]
        support_size = unique_mask.sum(dim=-1)
        max_support = int(support_size.max().detach().item())
        ranks = torch.clamp(unique_mask.cumsum(dim=-1) - 1, min=0)
        union_indices = sorted_indices.new_zeros(*sorted_indices.shape[:-1], max_support)
        union_indices.scatter_(dim=-1, index=torch.clamp(ranks, max=max_support - 1), src=sorted_indices)
        range_index = torch.arange(max_support, device=stacked.device).view(
            *([1] * support_size.dim()),
            max_support,
        )
        support_mask = range_index < support_size.unsqueeze(-1)
        return union_indices, support_mask, support_size

    @staticmethod
    def _gather_logps_from_response_logits(
        response_logits: torch.Tensor,
        support_indices: torch.Tensor,
        support_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gathered_logits = torch.gather(response_logits, dim=-1, index=support_indices)
        logsumexp = torch.logsumexp(response_logits, dim=-1, keepdim=True)
        gathered_logps = gathered_logits - logsumexp
        if support_mask is not None:
            gathered_logps = gathered_logps.masked_fill(~support_mask, -1e9)
        return gathered_logps

    @staticmethod
    def _compute_opd_alignment_metrics(
        student_topk_indices: torch.Tensor,
        student_topk_logps: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logps: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> dict[str, float]:
        """Rethinking-OPD diagnostics on the student and teacher native top-k sets."""

        matches = student_topk_indices.unsqueeze(-1) == teacher_topk_indices.unsqueeze(-2)
        overlap_mask = matches.any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        teacher_aligned_logps = torch.where(
            matches,
            teacher_topk_logps.unsqueeze(-2),
            torch.full_like(teacher_topk_logps.unsqueeze(-2), -torch.inf),
        ).amax(dim=-1)

        student_norm_logps = student_topk_logps - torch.logsumexp(student_topk_logps, dim=-1, keepdim=True)
        teacher_norm_logps = teacher_topk_logps - torch.logsumexp(teacher_topk_logps, dim=-1, keepdim=True)
        student_entropy = -(student_norm_logps.exp() * student_norm_logps).sum(dim=-1)
        teacher_entropy = -(teacher_norm_logps.exp() * teacher_norm_logps).sum(dim=-1)

        student_overlap_logps = student_topk_logps.masked_fill(~overlap_mask, -torch.inf)
        teacher_overlap_logps = teacher_aligned_logps.masked_fill(~overlap_mask, -torch.inf)
        has_overlap = overlap_count > 0
        student_overlap_logz = torch.where(
            has_overlap,
            torch.logsumexp(student_overlap_logps, dim=-1),
            torch.zeros_like(overlap_count, dtype=student_topk_logps.dtype),
        )
        teacher_overlap_logz = torch.where(
            has_overlap,
            torch.logsumexp(teacher_overlap_logps, dim=-1),
            torch.zeros_like(overlap_count, dtype=teacher_topk_logps.dtype),
        )
        student_overlap_norm = student_overlap_logps - student_overlap_logz.unsqueeze(-1)
        teacher_overlap_norm = teacher_overlap_logps - teacher_overlap_logz.unsqueeze(-1)
        overlap_advantage = torch.where(
            overlap_mask,
            student_overlap_norm.exp() * (teacher_overlap_norm - student_overlap_norm),
            torch.zeros_like(student_overlap_norm),
        ).sum(dim=-1) / overlap_count.clamp(min=1)

        valid = loss_mask.to(student_topk_logps.dtype)
        valid_count = valid.sum().clamp(min=1.0)

        def masked_mean(value: torch.Tensor) -> float:
            return ((value * valid).sum() / valid_count).detach().item()

        return {
            "self_distillation/topk_overlap_ratio": masked_mean(
                overlap_count.to(valid.dtype) / student_topk_indices.shape[-1]
            ),
            "self_distillation/student_topk_overlap_mass": masked_mean(
                (student_topk_logps.exp() * overlap_mask).sum(dim=-1)
            ),
            "self_distillation/teacher_topk_overlap_mass": masked_mean(
                (teacher_aligned_logps.exp() * overlap_mask).sum(dim=-1)
            ),
            "self_distillation/topk_overlap_advantage": masked_mean(overlap_advantage),
            "self_distillation/student_topk_entropy": masked_mean(student_entropy),
            "self_distillation/teacher_topk_entropy": masked_mean(teacher_entropy),
            "self_distillation/topk_entropy_gap": masked_mean((student_entropy - teacher_entropy).abs()),
        }

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)

        torch.save(
            {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            },
            save_path,
        )

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        return_native_topk: bool = False,
        return_response_logits: bool = False,
        pad_response_outputs: bool = True,
        return_rank_tie_logits: bool = False,
        return_reference_entropy: bool = False,
        entropy_target: Optional[dict] = None,
        return_grounding: bool = False,
        return_evidence_attention: bool = False,
        return_teacher_center: bool = False,
        return_gap_attention: bool = False,
        focus_context=None,
        focus_mode: str = "capture",
        grounding_view: str = "student",
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                if return_response_logits is True:
                    response_logits: (bs, response_len, vocab_size)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        return_response_logits = return_response_logits or return_rank_tie_logits
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk or return_response_logits) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")
        if return_response_logits and self.use_ulysses_sp:
            raise ValueError("return_response_logits is not implemented for Ulysses sequence parallelism.")
        if (return_grounding or return_evidence_attention or return_teacher_center or return_gap_attention or focus_context is not None) and (self.use_remove_padding or self.use_ulysses_sp or self.use_prefix_grouper):
            raise ValueError("Grounded OPD requires padded, non-Ulysses, non-PrefixGrouper actor forwards.")

        if (return_reference_entropy or entropy_target is not None) and (self.use_remove_padding or self.use_ulysses_sp or self.use_prefix_grouper or temperature != 1.0):
            raise ValueError("Correctness entropy requires padded, temperature-1, non-Ulysses forwards")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        original_response_length = micro_batch["responses"].size(-1)
        padding_fraction_removed = 0.0
        logit_position_fraction = None
        trim_padding = self.config.get("trim_padding", False) and not self.use_remove_padding
        if trim_padding:
            micro_batch, original_response_length, padding_fraction_removed = trim_padded_micro_batch(micro_batch)
            if topk_indices is not None:
                topk_indices = topk_indices[:, :micro_batch["responses"].shape[1]]
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = chunked_topk(logits_rmpad, topk)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = chunked_logsumexp(logits_rmpad)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad
                        if return_native_topk and topk_indices is not None:
                            native_topk_logits_rmpad, native_topk_indices_rmpad = chunked_topk(
                                logits_rmpad, topk
                            )
                            native_topk_logps_rmpad = native_topk_logits_rmpad - logsumexp_rmpad

                    if return_response_logits:
                        if response_start_idx is None:
                            response_positions = torch.arange(
                                seqlen - response_length - 1,
                                seqlen - 1,
                                device=input_ids.device,
                                dtype=torch.long,
                            ).unsqueeze(0).expand(batch_size, -1)
                        else:
                            response_positions = self._build_response_positions(
                                response_start_idx=response_start_idx.to(device=input_ids.device),
                                response_length=response_length,
                                seqlen=seqlen,
                            )
                        flat_positions = (
                            torch.arange(batch_size, device=input_ids.device).unsqueeze(1) * seqlen
                            + response_positions
                        )
                        flat_to_rmpad = torch.full(
                            (batch_size * seqlen,),
                            -1,
                            device=input_ids.device,
                            dtype=torch.long,
                        )
                        flat_to_rmpad[indices] = torch.arange(indices.numel(), device=input_ids.device)
                        response_rmpad_indices = flat_to_rmpad[flat_positions]
                        valid_response_positions = attention_mask.gather(1, response_positions).bool()
                        valid_response_positions = valid_response_positions & (response_rmpad_indices >= 0)
                        response_logits = logits_rmpad.new_zeros(batch_size, response_length, logits_rmpad.shape[-1])
                        response_logits[valid_response_positions] = logits_rmpad[
                            response_rmpad_indices[valid_response_positions]
                        ]

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        if return_native_topk and topk_indices is not None:
                            native_topk_logps_rmpad = gather_outputs_and_unpad(
                                native_topk_logps_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                            native_topk_indices_rmpad = gather_outputs_and_unpad(
                                native_topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]
                        if return_native_topk and topk_indices is not None:
                            native_topk_logps_rmpad = native_topk_logps_rmpad[:0]
                            native_topk_indices_rmpad = native_topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    if return_native_topk and topk_indices is not None:
                        full_native_topk_logps = pad_input(
                            hidden_states=native_topk_logps_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_native_topk_indices = pad_input(
                            hidden_states=native_topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if compute_all_logps:
                    all_logps = self._select_response_positions(
                        full_all_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                    if return_native_topk and topk_indices is not None:
                        native_topk_logps = self._select_response_positions(
                            full_native_topk_logps,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                        native_topk_indices = self._select_response_positions(
                            full_native_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                response_logit_indices = None
                if self.config.get("response_only_logits", False) and not self.use_fused_kernels:
                    starts = response_start_idx
                    if starts is None:
                        starts = input_ids.new_full((batch_size,), seqlen - response_length)
                    response_positions = self._build_response_positions(starts, response_length, seqlen)
                    keep_positions = torch.unique(response_positions, sorted=True)
                    logit_position_fraction = keep_positions.numel() / seqlen
                    extra_args["logits_to_keep"] = keep_positions
                    response_logit_indices = torch.searchsorted(keep_positions, response_positions)

                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                grounding_capture = None
                evidence_capture = None
                capture_context = nullcontext()
                if return_grounding:
                    from verl.trainer.ppo.grounded_opd import (
                        GroundedAttentionCapture,
                        apply_grounded_attention_patch,
                        capture_grounding,
                        select_grounded_attention_layer,
                    )

                    apply_grounded_attention_patch()
                    raw_model = getattr(model, "module", model)
                    model_config = raw_model.config
                    text_config = getattr(model_config, "text_config", model_config)
                    vision_config = getattr(model_config, "vision_config", None)
                    layer_start_fraction = float(
                        self.config.self_distillation.get("grounded_layer_start_fraction", 0.3)
                    )
                    layer_end_fraction = float(
                        self.config.self_distillation.get("grounded_layer_end_fraction", 0.6)
                    )
                    target_layer_idx = select_grounded_attention_layer(
                        list(getattr(text_config, "layer_types", []) or []),
                        layer_start_fraction=layer_start_fraction,
                        layer_end_fraction=layer_end_fraction,
                    )
                    response_positions = self._build_response_positions(
                        response_start_idx=(
                            response_start_idx
                            if response_start_idx is not None
                            else torch.full(
                                (batch_size,),
                                seqlen - response_length,
                                device=input_ids.device,
                                dtype=torch.long,
                            )
                        ),
                        response_length=response_length,
                        seqlen=seqlen,
                    )
                    grounding_capture = GroundedAttentionCapture(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        response_positions=response_positions,
                        image_grid_thw=multi_modal_inputs["image_grid_thw"],
                        image_token_id=int(model_config.image_token_id),
                        spatial_merge_size=int(vision_config.spatial_merge_size),
                        total_layers=int(text_config.num_hidden_layers),
                        grid_size=int(self.config.self_distillation.get("grounded_grid_size", 24)),
                        layer_start_fraction=layer_start_fraction,
                        layer_end_fraction=layer_end_fraction,
                        response_chunk_size=int(
                            self.config.self_distillation.get("grounded_response_chunk_size", 16)
                        ),
                        target_layer_idx=target_layer_idx,
                        view=grounding_view,
                        bboxes=micro_batch.get("grounded_bbox"),
                    )
                    capture_context = capture_grounding(grounding_capture)

                if return_evidence_attention:
                    from verl.trainer.ppo.evidence_caption import EvidenceAttentionCapture, evidence_attention_layer_types, select_evidence_attention_layers
                    from verl.trainer.ppo.grounded_opd import apply_grounded_attention_patch, capture_grounding
                    apply_grounded_attention_patch()
                    cfg = getattr(model, "module", model).config
                    layers = select_evidence_attention_layers(
                        evidence_attention_layer_types(cfg), self.config.self_distillation.evidence_attention_layer)
                    starts = response_start_idx
                    if starts is None:
                        starts = input_ids.new_full((batch_size,), seqlen - response_length)
                    evidence_capture = EvidenceAttentionCapture(
                        input_ids=input_ids, attention_mask=attention_mask,
                        response_positions=self._build_response_positions(starts, response_length, seqlen),
                        image_grid_thw=multi_modal_inputs["image_grid_thw"],
                        image_token_id=int(cfg.image_token_id),
                        spatial_merge_size=int(cfg.vision_config.spatial_merge_size),
                        total_layers=int(cfg.text_config.num_hidden_layers),
                        target_layer_idx=layers[0] if len(layers) == 1 else None,
                        layer_start_fraction=0.0, layer_end_fraction=1.0, expected_layer_indices=layers,
                        bboxes=micro_batch["evidence_bbox"], response_mask=micro_batch["response_mask"],
                        anchor_centers=micro_batch.get("evidence_centers"),
                        anchor_mask=micro_batch.get("evidence_center_mask"),
                        max_tokens=int(self.config.self_distillation.evidence_attention_max_tokens),
                        background_alpha=float(self.config.self_distillation.evidence_attention_alpha),
                        detached_scores=bool(self.config.self_distillation.get("attention_target_enabled", False)),
                        center_attention_requires_grad=(
                            self.config.self_distillation.get("attention_target_map", "gaussian") == "teacher_center"
                            and float(self.config.self_distillation.get("evidence_attention_lambda", 0.0)) > 0),
                        local_crop_footprints=(micro_batch.get("attention_crop_footprint") if
                            self.config.self_distillation.get("attention_target_map", "gaussian") == "local_occlusion" else None),
                        collect_competitor_attention=(self.config.self_distillation.get(
                            "attention_target_negative_mode", "mean_erased") == "student_competitor"),
                        target_map=self.config.self_distillation.get("attention_target_map", "gaussian"),
                        region_minimum_mass=float(self.config.self_distillation.get("attention_target_region_mass", 0.9)),
                    )
                    capture_context = capture_grounding(evidence_capture)

                teacher_center_capture = None
                if return_teacher_center:
                    from verl.trainer.ppo.teacher_center_target import TeacherCenterCapture
                    from verl.trainer.ppo.evidence_caption import evidence_attention_layer_types, select_evidence_attention_layers
                    from verl.trainer.ppo.grounded_opd import apply_grounded_attention_patch, capture_grounding
                    apply_grounded_attention_patch()
                    cfg = getattr(model, "module", model).config
                    layers = select_evidence_attention_layers(
                        evidence_attention_layer_types(cfg), self.config.self_distillation.evidence_attention_layer)
                    teacher_center_capture = TeacherCenterCapture(
                        input_ids=input_ids, attention_mask=attention_mask,
                        response_positions=self._build_response_positions(response_start_idx, response_length, seqlen),
                        image_grid_thw=multi_modal_inputs["image_grid_thw"],
                        image_token_id=int(cfg.image_token_id),
                        spatial_merge_size=int(cfg.vision_config.spatial_merge_size),
                        total_layers=int(cfg.text_config.num_hidden_layers),
                        target_layer_idx=layers[0] if len(layers) == 1 else None,
                        layer_start_fraction=0.0, layer_end_fraction=1.0,
                        expected_layer_indices=layers, response_mask=micro_batch["response_mask"],
                        crop_footprints=micro_batch["attention_crop_footprint"],
                        max_tokens=int(self.config.self_distillation.evidence_attention_max_tokens),
                    )
                    capture_context = capture_grounding(teacher_center_capture)

                gap_capture = None
                if return_gap_attention:
                    from verl.trainer.ppo.attention_gap import PromptAttentionCapture
                    from verl.trainer.ppo.grounded_opd import apply_grounded_attention_patch, capture_grounding
                    apply_grounded_attention_patch()
                    cfg = getattr(model, "module", model).config
                    starts = response_start_idx
                    if starts is None:
                        starts = input_ids.new_full((batch_size,), seqlen - response_length)
                    gap_capture = PromptAttentionCapture(
                        input_ids, attention_mask, starts - 1, multi_modal_inputs["image_grid_thw"],
                        int(cfg.image_token_id), int(cfg.vision_config.spatial_merge_size),
                        tuple(i for i, kind in enumerate(cfg.text_config.layer_types) if kind == "full_attention"),
                    )
                    capture_context = capture_grounding(gap_capture)
                if focus_context is not None:
                    from verl.trainer.ppo.attention_focus import install_attention_focus, use_attention_focus
                    install_attention_focus()
                    cfg = getattr(model, "module", model).config
                    starts = response_start_idx
                    if starts is None:
                        starts = input_ids.new_full((batch_size,), seqlen - response_length)
                    focus_context.configure(
                        input_ids, attention_mask,
                        self._build_response_positions(starts, response_length, seqlen),
                        multi_modal_inputs["image_grid_thw"], int(cfg.image_token_id),
                        int(cfg.vision_config.spatial_merge_size), focus_mode)
                    capture_context = use_attention_focus(focus_context)
                with capture_context:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                if grounding_capture is not None:
                    grounding_probs, grounding_layers = grounding_capture.finish()
                if evidence_capture is not None:
                    evidence_loss, evidence_metrics = evidence_capture.finish()
                    evidence_token_kl, evidence_token_selected = evidence_capture.token_kl, evidence_capture.token_selected
                    local_evidence_attention = evidence_capture.local_attention
                    competitor_attention = evidence_capture.competitor_attention
                    center_attention = evidence_capture.center_attention
                    del evidence_capture
                if teacher_center_capture is not None:
                    teacher_centers, teacher_center_selected = teacher_center_capture.finish()
                    del teacher_center_capture

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    del output

                    logits.div_(temperature)
                    if response_logit_indices is None:
                        logits = self._select_response_positions(
                            logits, response_length=response_length, response_start_idx=response_start_idx,
                        )
                    else:
                        logits = select_response_logits(logits, response_logit_indices)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if return_response_logits:
                        response_logits = logits
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = chunked_topk(logits, topk)
                        else:
                            topk = topk_indices.size(-1)
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = chunked_logsumexp(logits)
                        topk_logps = topk_logits - logsumexp
                        if return_native_topk and topk_indices is not None:
                            native_topk_logits, native_topk_indices = chunked_topk(logits, topk)
                            native_topk_logps = native_topk_logits - logsumexp
                    entropy_stats = {}
                    if return_reference_entropy or entropy_target is not None:
                        from verl.trainer.ppo.correctness_entropy import reference_entropy, soften_teacher, student_topk_logps
                        if return_reference_entropy:
                            entropy_stats["reference_entropy"] = reference_entropy(logits)
                            topk_logps = student_topk_logps(logits, topk_indices)
                        if entropy_target is not None:
                            topk_logps, entropy_stats = soften_teacher(
                                logits, topk_indices, **entropy_target)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if return_reference_entropy or entropy_target is not None:
                outputs.update(entropy_stats)
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
                if return_native_topk and topk_indices is not None:
                    outputs["native_topk_logps"] = native_topk_logps
                    outputs["native_topk_indices"] = native_topk_indices
            if return_response_logits and not return_rank_tie_logits:
                outputs["response_logits"] = response_logits
            if return_grounding:
                outputs["grounding_probs"] = grounding_probs
                outputs["grounding_layer_count"] = len(set(grounding_layers))
            if logit_position_fraction is not None:
                outputs["logit_position_fraction"] = logit_position_fraction
            if return_evidence_attention and evidence_token_kl is not None:
                outputs["attention_token_kl"] = evidence_token_kl
                outputs["attention_token_selected"] = evidence_token_selected
            if return_teacher_center:
                outputs["teacher_attention_centers"] = teacher_centers
                outputs["teacher_center_selected"] = teacher_center_selected
            if trim_padding and pad_response_outputs:
                outputs = restore_response_width(outputs, original_response_length)
                outputs["padding_fraction_removed"] = padding_fraction_removed
            if return_gap_attention:
                outputs["gap_attention"] = gap_capture.finish()
            if return_evidence_attention:
                outputs["evidence_attention_loss"] = evidence_loss
                outputs["evidence_attention_metrics"] = evidence_metrics
                if competitor_attention is not None:
                    outputs["competitor_attention"] = competitor_attention
                if local_evidence_attention is not None:
                    outputs["local_evidence_attention"] = local_evidence_attention
                if center_attention is not None:
                    outputs["center_attention"] = center_attention
            if return_rank_tie_logits:
                # Keep the compact response width; never pad vocabulary logits.
                outputs["rank_tie_logits"] = response_logits
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        # Adam state first exists after step 1. Loading it before the next
        # backward unnecessarily overlaps two FP32 state tensors with FSDP's
        # full-layer gradients (about 46 GiB per rank for 397B / 64 ranks).
        if self.optimizer_step_only_offload:
            load_fsdp_optimizer(self.actor_optimizer, device_id=get_device_id())
        try:
            if self.scaler is not None:
                self.scaler.step(self.actor_optimizer)
                self.scaler.update()
            else:
                self.actor_optimizer.step()
        finally:
            if self.optimizer_step_only_offload:
                offload_fsdp_optimizer(self.actor_optimizer)
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    def _precompute_va_tensors(self, data: DataProto, temperature: float, pad_token_id: int) -> dict:
        """VA-OPD (arXiv:2605.21924) pre-pass.

        Scores each student-rollout token under the teacher with the
        evidence-present view (teacher_* inputs) and the evidence-degraded view
        (teacher_minus_* inputs), computes the per-token visual advantage

            a_t = max(log p_T(y_t | x+) - log p_T(y_t | x-), 0),

        then derives (i) the per-rollout high-VA-group token mask (top
        ``va_high_fraction`` of valid tokens by VA) and (ii) rollout-level
        softmax weights from trajectory-averaged VA, z-score normalized within
        each uid sibling group across all data-parallel ranks. Results are
        attached to ``data.batch`` as ``va_high_group_mask`` (B, T) and
        ``va_rollout_weights`` (B,) so they flow through mini/micro splits.

        Returns a dict of diagnostic metrics.
        """
        cfg = self.config.self_distillation
        teacher_model = self.teacher_module or self.actor_module
        # "teacher": probe = evidence-present teacher view vs degraded teacher view.
        # "student": probe = student's own (full-image) inputs scored by the teacher
        # module vs degraded full image (teacher_minus_* inputs must then carry the
        # degraded full image so both probe conditions share the deployment view).
        probe_view = cfg.get("va_probe_view", "teacher")

        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
            batch_idx_list = None

        plus_log_probs_lst = []
        minus_log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            if probe_view == "student":
                # Evidence-present probe condition = the student's own inputs
                # (full image, raw prompt), scored by the teacher module.
                teacher_inputs = {
                    "responses": model_inputs["responses"],
                    "input_ids": model_inputs["input_ids"],
                    "attention_mask": model_inputs["attention_mask"],
                    "position_ids": model_inputs["position_ids"],
                }
                if "multi_modal_inputs" in model_inputs:
                    teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
            else:
                teacher_inputs = {
                    "responses": model_inputs["responses"],
                    "input_ids": model_inputs["teacher_input_ids"],
                    "attention_mask": model_inputs["teacher_attention_mask"],
                    "position_ids": model_inputs["teacher_position_ids"],
                    "response_start_idx": model_inputs["teacher_response_start_idx"],
                }
                if "teacher_multi_modal_inputs" in model_inputs:
                    teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
            teacher_minus_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_minus_input_ids"],
                "attention_mask": model_inputs["teacher_minus_attention_mask"],
                "position_ids": model_inputs["teacher_minus_position_ids"],
                "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
            }
            if "teacher_minus_multi_modal_inputs" in model_inputs:
                teacher_minus_inputs["multi_modal_inputs"] = model_inputs["teacher_minus_multi_modal_inputs"]
            with torch.no_grad():
                plus_outputs = self._forward_micro_batch(
                    teacher_inputs, temperature=temperature, calculate_entropy=False, module=teacher_model
                )
                minus_outputs = self._forward_micro_batch(
                    teacher_minus_inputs, temperature=temperature, calculate_entropy=False, module=teacher_model
                )
            plus_log_probs_lst.append(plus_outputs["log_probs"].float().cpu())
            minus_log_probs_lst.append(minus_outputs["log_probs"].float().cpu())

        plus_log_probs = torch.concat(plus_log_probs_lst, dim=0)
        minus_log_probs = torch.concat(minus_log_probs_lst, dim=0)
        if batch_idx_list is not None:
            plus_log_probs = restore_dynamic_batch(plus_log_probs, batch_idx_list)
            minus_log_probs = restore_dynamic_batch(minus_log_probs, batch_idx_list)

        response_mask = data.batch["response_mask"].float().cpu()
        sd_mask = data.batch["self_distillation_mask"].float().cpu()
        loss_mask = response_mask * sd_mask.unsqueeze(1)

        token_va = torch.clamp(plus_log_probs - minus_log_probs, min=0.0) * response_mask
        valid_counts = loss_mask.sum(dim=-1)
        traj_va = (token_va * loss_mask).sum(dim=-1) / valid_counts.clamp(min=1.0)
        active = (valid_counts > 0).float()

        if cfg.get("va_token_grouping", True):
            high_group_mask = compute_va_high_group_mask(
                token_va, loss_mask, high_fraction=float(cfg.get("va_high_fraction", 0.2))
            )
        else:
            high_group_mask = torch.zeros_like(loss_mask)

        if cfg.get("va_rollout_reweight", True):
            if "uid" not in data.non_tensor_batch:
                raise ValueError(
                    "VA-OPD rollout reweighting requires `uid` in the batch to group sibling rollouts. "
                    "Set va_rollout_reweight=False to disable, or ensure uids are propagated."
                )
            uids = [str(u) for u in data.non_tensor_batch["uid"]]
            if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                world_size = torch.distributed.get_world_size()
                gathered: list = [None] * world_size
                torch.distributed.all_gather_object(gathered, (uids, traj_va.tolist(), active.tolist()))
                global_uids = [u for part in gathered for u in part[0]]
                global_traj_va = [v for part in gathered for v in part[1]]
                global_active = [a for part in gathered for a in part[2]]
            else:
                global_uids, global_traj_va, global_active = uids, traj_va.tolist(), active.tolist()
            rollout_weights = compute_va_rollout_weights(
                local_uids=uids,
                local_traj_va=traj_va,
                local_active=active,
                global_uids=global_uids,
                global_traj_va=global_traj_va,
                global_active=global_active,
                softmax_temp=float(cfg.get("va_rollout_softmax_temp", 1.0)),
                zscore_epsilon=float(cfg.get("va_zscore_epsilon", 1e-6)),
            )
        else:
            rollout_weights = active.clone()

        data.batch["va_rollout_weights"] = rollout_weights
        data.batch["va_high_group_mask"] = high_group_mask

        num_valid_tokens = loss_mask.sum().clamp(min=1.0)
        num_active = active.sum().clamp(min=1.0)
        high_token_count = (high_group_mask * loss_mask).sum().clamp(min=1.0)
        va_mass_total = (token_va * loss_mask).sum().clamp(min=1e-12)
        va_mass_high = (token_va * high_group_mask * loss_mask).sum()
        metrics = {
            "self_distillation/va_token_mean": ((token_va * loss_mask).sum() / num_valid_tokens).item(),
            "self_distillation/va_traj_mean": ((traj_va * active).sum() / num_active).item(),
            "self_distillation/va_high_group_token_frac": (high_token_count / num_valid_tokens).item(),
            "self_distillation/va_high_group_mass_frac": (va_mass_high / va_mass_total).item(),
            "self_distillation/va_rollout_weight_local_max": float(rollout_weights.max().item()),
            "self_distillation/va_active_rollout_frac": float(active.mean().item()),
        }
        return metrics

    def _precompute_vzero_tensors(self, data: DataProto, temperature: float, pad_token_id: int) -> dict:
        """Replay positive/negative teacher views and attach V-Zero gates.

        The positive sampled-token log-probabilities are retained for the OPD
        target, so the main update loop does not repeat the positive teacher
        forward.  Sibling z-score statistics are gathered across data-parallel
        ranks because a rollout group is not guaranteed to stay rank-local.
        """

        cfg = self.config.self_distillation
        teacher_model = self.teacher_module or self.actor_module

        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.ppo_micro_batch_size_per_gpu)
            batch_idx_list = None

        plus_log_probs_list = []
        minus_log_probs_list = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            teacher_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_input_ids"],
                "attention_mask": model_inputs["teacher_attention_mask"],
                "position_ids": model_inputs["teacher_position_ids"],
                "response_start_idx": model_inputs["teacher_response_start_idx"],
            }
            if "teacher_multi_modal_inputs" in model_inputs:
                teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]

            teacher_minus_inputs = {
                "responses": model_inputs["responses"],
                "input_ids": model_inputs["teacher_minus_input_ids"],
                "attention_mask": model_inputs["teacher_minus_attention_mask"],
                "position_ids": model_inputs["teacher_minus_position_ids"],
                "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
            }
            if "teacher_minus_multi_modal_inputs" in model_inputs:
                teacher_minus_inputs["multi_modal_inputs"] = model_inputs["teacher_minus_multi_modal_inputs"]

            with torch.no_grad():
                plus_outputs = self._forward_micro_batch(
                    teacher_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    module=teacher_model,
                )
                minus_outputs = self._forward_micro_batch(
                    teacher_minus_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    module=teacher_model,
                )
            plus_log_probs_list.append(plus_outputs["log_probs"].float().cpu())
            minus_log_probs_list.append(minus_outputs["log_probs"].float().cpu())

        plus_log_probs = torch.concat(plus_log_probs_list, dim=0)
        minus_log_probs = torch.concat(minus_log_probs_list, dim=0)
        if batch_idx_list is not None:
            plus_log_probs = restore_dynamic_batch(plus_log_probs, batch_idx_list)
            minus_log_probs = restore_dynamic_batch(minus_log_probs, batch_idx_list)

        response_mask = data.batch["response_mask"].float().cpu()
        self_distillation_mask = data.batch["self_distillation_mask"].float().cpu()
        loss_mask = response_mask * self_distillation_mask.unsqueeze(1)
        active = (loss_mask.sum(dim=-1) > 0).float()
        if "uid" not in data.non_tensor_batch:
            raise ValueError("V-Zero requires `uid` in the actor batch for sibling rollout grouping.")
        uids = [str(uid) for uid in data.non_tensor_batch["uid"]]

        local_scores, _ = compute_vzero_trajectory_scores(
            teacher_logprobs=plus_log_probs,
            teacher_neg_logprobs=minus_log_probs,
            response_mask=loss_mask,
            gamma=float(cfg.get("vzero_gate_gamma", 1.0)),
            eps=float(cfg.get("vzero_gate_epsilon", 1e-6)),
        )
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            world_size = torch.distributed.get_world_size()
            gathered: list = [None] * world_size
            torch.distributed.all_gather_object(gathered, (uids, local_scores.tolist(), active.tolist()))
            global_uids = [uid for part in gathered for uid in part[0]]
            global_scores = [score for part in gathered for score in part[1]]
            global_active = [item for part in gathered for item in part[2]]
        else:
            global_uids = uids
            global_scores = local_scores.tolist()
            global_active = active.tolist()

        weights, metrics, diagnostics = compute_vzero_evidence_weights(
            teacher_logprobs=plus_log_probs,
            teacher_neg_logprobs=minus_log_probs,
            response_mask=loss_mask,
            uids=uids,
            gamma=float(cfg.get("vzero_gate_gamma", 1.0)),
            alpha=float(cfg.get("vzero_gate_alpha", 0.5)),
            w_min=float(cfg.get("vzero_gate_min", 0.0)),
            w_max=float(cfg.get("vzero_gate_max", 2.0)),
            eps=float(cfg.get("vzero_gate_epsilon", 1e-6)),
            active=active,
            global_uids=global_uids,
            global_scores=global_scores,
            global_active=global_active,
        )
        data.batch["vzero_teacher_log_probs"] = plus_log_probs
        data.batch["vzero_rollout_weights"] = weights
        data.batch["vzero_scores"] = diagnostics["scores"]
        data.batch["vzero_advantages"] = diagnostics["advantages"]
        metrics.update(
            {
                "rew_opd/active_rollout_frac": active.mean().item(),
                "rew_opd/public_code_gamma": float(cfg.get("vzero_gate_gamma", 1.0)),
                "rew_opd/public_code_alpha": float(cfg.get("vzero_gate_alpha", 0.5)),
            }
        )
        return metrics

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        correctness_entropy_enabled = bool(self_distillation_enabled and self_distillation_cfg.get("correctness_entropy_enabled", False))
        vd_enabled = bool(self_distillation_enabled and self_distillation_cfg.get("visual_dependency_enabled", False))
        eg_enabled = bool(self_distillation_enabled and self_distillation_cfg.get("evidence_gap_enabled", False))
        vc_target_enabled = False
        va_target_enabled = False
        vzero_target_enabled = False
        opdv_target_enabled = False
        er_target_enabled = False
        ssp_target_enabled = False
        ssp_v2_target_enabled = False
        ver_target_enabled = False
        sa_target_enabled = False
        uni_target_enabled = False
        svr_target_enabled = False
        dsvr_target_enabled = False
        vgr_target_enabled = False
        grounded_target_enabled = False
        rank_tie_lambda = 0.0
        gap_lambda = float(self_distillation_cfg.get("attention_gap_lambda", 0.0)) if self_distillation_cfg else 0.0
        gap_cache = {}
        focus_enabled = bool(self_distillation_cfg.get("attention_focus_enabled", False)) if self_distillation_cfg else False
        at_enabled = bool(self_distillation_cfg.get("attention_target_enabled", False)) if self_distillation_cfg else False
        union_teacher_topk = int(self_distillation_cfg.get("attention_target_teacher_topk", 0)) if at_enabled else 0
        union_statistics = None
        if union_teacher_topk:
            from verl.trainer.ppo.teacher_union import SupportStatistics
            union_statistics = SupportStatistics(get_device_id())
        local_target_enabled = bool(at_enabled and self_distillation_cfg.get("attention_target_map", "gaussian") == "local_occlusion")
        teacher_center_enabled = bool(at_enabled and self_distillation_cfg.get("attention_target_map", "gaussian") == "teacher_center")
        competitor_enabled = bool(at_enabled and self_distillation_cfg.get(
            "attention_target_negative_mode", "mean_erased") == "student_competitor")
        competitor_image_cache = {}
        if at_enabled and self.config.ppo_epochs != 1:
            raise ValueError("Attention target currently requires one optimizer pass per rollout batch")
        evidence_lambda = float(self_distillation_cfg.get("evidence_attention_lambda", 0.0)) if self_distillation_cfg else 0.0
        if evidence_lambda > 0:
            from verl.trainer.ppo.evidence_caption import attention_weight_at_step
            evidence_lambda = attention_weight_at_step(self_distillation_cfg, self._current_global_steps)
        caption_lambda = float(self_distillation_cfg.get("caption_kl_lambda", 0.0)) if self_distillation_cfg else 0.0
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            rank_tie_lambda = float(self_distillation_cfg.get("rank_tie_lambda", 0.0))
            target_mode = self_distillation_cfg.get("target_mode", "full_teacher")
            if gap_lambda and (target_mode != "full_teacher" or rank_tie_lambda or not self_distillation_cfg.distillation_topk):
                raise ValueError("Attention-gap requires top-k JSD without rank-tie")
            if rank_tie_lambda and (target_mode != "full_teacher" or not self_distillation_cfg.distillation_topk):
                raise ValueError("Rank-tie correction requires the ordinary full_teacher top-k JSD path.")
            vc_target_enabled = target_mode == "vc"
            va_target_enabled = target_mode == "va"
            vzero_target_enabled = target_mode == "vzero"
            opdv_target_enabled = target_mode == "opdv"
            er_target_enabled = target_mode == "er_opd"
            ssp_target_enabled = target_mode == "ssp_opd"
            ssp_v2_target_enabled = target_mode == "ssp_opd_v2"
            ver_target_enabled = target_mode == "ver_opd"
            sa_target_enabled = target_mode == "sa"
            uni_target_enabled = target_mode == "uni_opd"
            svr_target_enabled = target_mode == "svr_opd"
            dsvr_target_enabled = target_mode == "dsvr_opd"
            vgr_target_enabled = target_mode in ("vgr_relay", "vrbr_opd")
            grounded_target_enabled = target_mode == "grounded_opd"
            raw_counterfactual_keys = self_distillation_cfg.get("counterfactual_teacher_image_keys", None)
            if raw_counterfactual_keys is None:
                counterfactual_view_count = 1
            elif isinstance(raw_counterfactual_keys, str):
                counterfactual_view_count = len(
                    [item.strip() for item in raw_counterfactual_keys.split(",") if item.strip()]
                )
            else:
                counterfactual_view_count = len(list(raw_counterfactual_keys))
            vc_direction_mode = self_distillation_cfg.get("vc_direction_mode", "single")
            use_mean_vc_direction = vc_target_enabled and vc_direction_mode == "mean" and counterfactual_view_count > 1
            if local_target_enabled:
                counterfactual_view_count = 5  # fully erased crop, then four local occlusions
            need_extra_teacher_views = use_mean_vc_direction or local_target_enabled
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            if focus_enabled:
                self_distillation_required_keys.add("attention_focus_bbox")
            if evidence_lambda > 0 or at_enabled:
                self_distillation_required_keys.add("evidence_bbox")
                if "evidence_centers" in data.batch:
                    self_distillation_required_keys.update({"evidence_centers", "evidence_center_mask"})
            if local_target_enabled or teacher_center_enabled:
                self_distillation_required_keys.add("attention_crop_footprint")
            if caption_lambda > 0:
                self_distillation_required_keys.update(
                    "caption_" + key for key in ("input_ids", "attention_mask", "position_ids", "response_start_idx"))
            if correctness_entropy_enabled:
                self_distillation_required_keys.add("question_success")
            if vd_enabled:
                self_distillation_required_keys.add("vd_is_degraded")
                for view in ("original", "degraded"):
                    self_distillation_required_keys.update(
                        f"teacher_vd_{view}_{name}" for name in
                        ("input_ids", "attention_mask", "position_ids", "response_start_idx")
                    )
            if eg_enabled:
                self_distillation_required_keys.add("eg_is_degraded")
                for view in ("original", "erased"):
                    self_distillation_required_keys.update(f"teacher_eg_{view}_{name}" for name in
                        ("input_ids", "attention_mask", "position_ids", "response_start_idx"))
            if grounded_target_enabled:
                self_distillation_required_keys.add("grounded_bbox")
            if self_distillation_cfg.get("visual_repair_token_weight", 1.0) != 1.0:
                self_distillation_required_keys.add("repair_token_mask")
            if uni_target_enabled:
                self_distillation_required_keys.update(
                    {"uni_response_correct", "uni_sample_keep_mask", "uni_group_index"}
                )
            if ver_target_enabled:
                self_distillation_required_keys.update({"ver_rollout_weights", "ver_loss_scale"})
            if svr_target_enabled or dsvr_target_enabled or vgr_target_enabled:
                self_distillation_required_keys.add("distillation_token_mask")
            if ssp_v2_target_enabled:
                self_distillation_required_keys.add("distillation_token_mask")
            if dsvr_target_enabled:
                self_distillation_required_keys.add("dsvr_suffix_mask")
            if vgr_target_enabled:
                self_distillation_required_keys.update({"vgr_bridge_mask", "vgr_suffix_mask"})
            if at_enabled or vc_target_enabled or va_target_enabled or vzero_target_enabled or opdv_target_enabled or er_target_enabled or ssp_target_enabled or ssp_v2_target_enabled or ver_target_enabled or dsvr_target_enabled or vgr_target_enabled:
                self_distillation_required_keys.update(
                    {
                        "teacher_minus_input_ids",
                        "teacher_minus_attention_mask",
                        "teacher_minus_position_ids",
                        "teacher_minus_response_start_idx",
                    }
                )
                if need_extra_teacher_views:
                    for view_idx in range(1, counterfactual_view_count):
                        self_distillation_required_keys.update(
                            {
                                f"teacher_minus_{view_idx}_input_ids",
                                f"teacher_minus_{view_idx}_attention_mask",
                                f"teacher_minus_{view_idx}_position_ids",
                                f"teacher_minus_{view_idx}_response_start_idx",
                            }
                        )
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
        ]
        if "old_log_probs" in data.batch:
            select_keys.append("old_log_probs")
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        has_teacher_minus_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_minus_multi_modal_inputs")
        )
        non_tensor_select_keys = ["gap_metadata"] if gap_lambda else []
        if union_teacher_topk:
            non_tensor_select_keys.append("world_side")
        if competitor_enabled:
            non_tensor_select_keys.append("competitor_metadata")
        if eg_enabled:
            non_tensor_select_keys.extend(f"teacher_eg_{view}_multi_modal_inputs" for view in ("original", "erased"))
        if vd_enabled:
            non_tensor_select_keys.extend(f"teacher_vd_{view}_multi_modal_inputs" for view in ("original", "degraded"))
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if has_teacher_minus_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_minus_multi_modal_inputs")
        if self_distillation_enabled and need_extra_teacher_views:
            for view_idx in range(1, counterfactual_view_count):
                key = f"teacher_minus_{view_idx}_multi_modal_inputs"
                if self._has_non_empty_multi_modal_inputs(data.non_tensor_batch.get(key)):
                    non_tensor_select_keys.append(key)
        if (eg_enabled or correctness_entropy_enabled or self.use_prefix_grouper or va_target_enabled or vzero_target_enabled or ssp_target_enabled or ssp_v2_target_enabled) and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        eg_precompute_metrics = None
        if eg_enabled:
            from verl.trainer.ppo.evidence_gap import precompute_weights
            eg_precompute_metrics = precompute_weights(self, data, get_device_id(), pad_token_id)
            for view in ("original", "erased"):
                for name in ("input_ids", "attention_mask", "position_ids", "response_start_idx"):
                    data.batch.pop(f"teacher_eg_{view}_{name}")
                data.non_tensor_batch.pop(f"teacher_eg_{view}_multi_modal_inputs")

        vd_precompute_metrics = None
        if vd_enabled:
            from verl.trainer.ppo.visual_dependency import precompute_scores
            vd_precompute_metrics = precompute_scores(self, data, get_device_id(), pad_token_id)
            # Probe payloads are no longer needed during the gradient pass.
            for view in ("original", "degraded"):
                for name in ("input_ids", "attention_mask", "position_ids", "response_start_idx"):
                    data.batch.pop(f"teacher_vd_{view}_{name}")
                data.non_tensor_batch.pop(f"teacher_vd_{view}_multi_modal_inputs")

        va_precompute_metrics = None
        if va_target_enabled:
            # VA-OPD pre-pass: score every rollout token under the teacher's
            # evidence-present and evidence-degraded views, then attach
            # va_rollout_weights / va_high_group_mask tensors to the batch so
            # they flow through the mini/micro-batch splits below.
            va_precompute_metrics = self._precompute_va_tensors(
                data, temperature=temperature, pad_token_id=pad_token_id
            )

        vzero_precompute_metrics = None
        if vzero_target_enabled:
            vzero_precompute_metrics = self._precompute_vzero_tensors(
                data, temperature=temperature, pad_token_id=pad_token_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        if eg_enabled and not on_policy:
            raise ValueError("Evidence-gap weights require one optimizer step per frozen rollout batch")
        if vd_enabled and not on_policy:
            raise ValueError("Visual dependency requires one optimizer step per frozen rollout batch")
        if correctness_entropy_enabled and (not on_policy or self.config.loss_agg_mode != "token-mean"):
            raise ValueError("Correctness entropy requires one token-mean optimizer step per rollout batch")

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        if vd_precompute_metrics:
            metrics.update(vd_precompute_metrics)
        if eg_precompute_metrics:
            metrics.update(eg_precompute_metrics)
        if va_precompute_metrics:
            metrics.update(va_precompute_metrics)
        if vzero_precompute_metrics:
            metrics.update(vzero_precompute_metrics)
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/teacher_minus_forward": 0.0,
                "timing_s/update_actor/student_minus_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                eg_denominator = None
                if eg_enabled:
                    from verl.trainer.ppo.correctness_entropy import token_denominator
                    eg_denominator = token_denominator(
                        mini_batch.batch["evidence_gap_weights"] * mini_batch.batch["response_mask"]
                        * mini_batch.batch["self_distillation_mask"][:, None], get_device_id())
                vd_denominator = None
                if vd_enabled:
                    from verl.trainer.ppo.correctness_entropy import token_denominator
                    vd_denominator = token_denominator(mini_batch.batch["distillation_token_mask"], get_device_id())
                correctness_denominator = None
                if correctness_entropy_enabled:
                    from verl.trainer.ppo.correctness_entropy import token_denominator
                    correctness_denominator = token_denominator(mini_batch.batch["response_mask"], get_device_id())
                repair_batch_denominator = None
                if "repair_token_mask" in mini_batch.batch:
                    from verl.trainer.ppo.visual_repair import repair_weight_denominator
                    if self.ulysses_sequence_parallel_size != 1 or self.config.loss_agg_mode != "token-mean":
                        raise ValueError("Repair token weighting requires token-mean and sequence parallel size 1")
                    repair_batch_denominator = repair_weight_denominator(
                        mini_batch.batch["response_mask"], mini_batch.batch["repair_token_mask"],
                        float(self_distillation_cfg.visual_repair_token_weight), get_device_id())
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    if self.config.get("sort_microbatches_by_length", False):
                        # Reorder the complete DataProto so images, prefixes and
                        # Teacher inputs stay aligned. Long pairs go first.
                        lengths = mini_batch.batch["attention_mask"].sum(-1)
                        mini_batch.reorder(torch.argsort(lengths, descending=True, stable=True))
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                defer_sync = self.config.get("defer_gradient_sync", False)
                if defer_sync and (not isinstance(self.actor_module, FSDP) or self.config.fsdp_config.param_offload):
                    raise ValueError("Deferred gradient sync requires FSDP1 without parameter offload")
                append_to_dict(metrics, {
                    "perf/actor_microbatches": len(micro_batches),
                    "perf/actor_gradient_sync_rounds": 1 if defer_sync else len(micro_batches),
                })
                for micro_idx, micro_batch in enumerate(micro_batches):
                    with gradient_sync_context(self.actor_module, defer_sync, micro_idx == len(micro_batches) - 1):
                        original_micro_size = len(micro_batch)
                        if correctness_entropy_enabled:
                            from verl.trainer.ppo.correctness_entropy import collapse_identical
                            micro_batch = collapse_identical(micro_batch)
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        if correctness_entropy_enabled:
                            micro_batch_metrics["correctness/distillation_duplicate_fraction"] = 1 - len(micro_batch) / original_micro_size
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs.get("old_log_probs")
                        advantages = model_inputs.get("advantages")

                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                        self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                        distillation_token_mask = model_inputs.get("distillation_token_mask")
                        policy_fallback_mask = None
                        if self_distillation_enabled and self_distillation_mask is not None:
                            if not ver_target_enabled:
                                policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                            micro_batch_metrics["actor/policy_fallback_fraction"] = (
                                policy_fallback_mask.float().mean().detach().item()
                                if policy_fallback_mask is not None
                                else 0.0
                            )

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation
                        if repair_batch_denominator is not None or correctness_denominator is not None or vd_denominator is not None:
                            # The denominator spans all microbatches and DP ranks.
                            loss_scale_factor = 1.0

                        teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                        teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                        use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                        if use_trust_region_teacher and self.use_fused_kernels:
                            raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                        # all return: (bsz, response_length)
                        return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                        distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                        vc_support_mode = (
                            self_distillation_cfg.get("vc_support_mode", "student_topk")
                            if vc_target_enabled or ver_target_enabled
                            else "student_topk"
                        )
                        use_three_view_union = er_target_enabled or (
                            (vc_target_enabled or ver_target_enabled) and vc_support_mode == "three_view_union"
                        )
                        distill_support_mask = None
                        student_forward_start = time.perf_counter()
                        outputs = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            return_all_logps=return_all_logps,
                            distill_topk=distill_topk,
                            return_response_logits=use_three_view_union or bool(union_teacher_topk),
                            return_rank_tie_logits=rank_tie_lambda > 0,
                            return_reference_entropy=correctness_entropy_enabled,
                            return_grounding=grounded_target_enabled,
                            return_evidence_attention=evidence_lambda > 0 or at_enabled,
                            grounding_view="student",
                            return_gap_attention=gap_lambda > 0,
                        )
                        if self_distillation_enabled:
                            student_forward_time = time.perf_counter() - student_forward_start
                            stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                            if "padding_fraction_removed" in outputs:
                                micro_batch_metrics["perf/student_padding_removed_fraction"] = outputs["padding_fraction_removed"]
                        if "logit_position_fraction" in outputs:
                            micro_batch_metrics["perf/student_logit_position_fraction"] = outputs["logit_position_fraction"]
                        log_prob = outputs["log_probs"]
                        entropy = outputs["entropys"] if calculate_entropy else None
                        student_all_logps = outputs.get("all_logps") if return_all_logps else None
                        student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                        student_topk_indices = outputs.get("topk_indices") if distill_topk else None
                        if distill_topk and self_distillation_cfg.get("visual_repair_enabled", False):
                            from verl.trainer.ppo.visual_repair import include_response_in_support
                            student_topk_indices, student_topk_logps = include_response_in_support(
                                student_topk_indices, student_topk_logps, model_inputs["responses"], log_prob)

                        # for fully_async_policy
                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs.get("old_log_probs")

                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                        # Extract pre-computed rollout correction weights if present
                        # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        if self_distillation_enabled:
                            teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids"],
                                "attention_mask": model_inputs["teacher_attention_mask"],
                                "position_ids": model_inputs["teacher_position_ids"],
                                "response_start_idx": model_inputs["teacher_response_start_idx"],
                            }
                            if grounded_target_enabled:
                                teacher_inputs["grounded_bbox"] = model_inputs["grounded_bbox"]
                            if teacher_center_enabled:
                                teacher_inputs["response_mask"] = response_mask
                                teacher_inputs["attention_crop_footprint"] = model_inputs["attention_crop_footprint"]
                            if "teacher_multi_modal_inputs" in model_inputs:
                                teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                            elif (
                                va_target_enabled
                                and self_distillation_cfg.get("va_probe_view", "teacher") == "student"
                                and "multi_modal_inputs" in model_inputs
                            ):
                                teacher_inputs["multi_modal_inputs"] = model_inputs["multi_modal_inputs"]
                            teacher_model = self.teacher_module or self.actor_module
                            if use_trust_region_teacher and (
                                self.teacher_module is None or self.teacher_module is self.actor_module
                            ):
                                raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                            if vzero_target_enabled:
                                # The V-Zero pre-pass already replayed both evidence
                                # views. Reuse its positive sampled-token log-probs as
                                # the detached OPD target instead of forwarding it a
                                # second time for every PPO micro-batch.
                                teacher_outputs = None
                                teacher_log_prob = model_inputs["vzero_teacher_log_probs"]
                                teacher_all_logps = None
                                teacher_topk_logps = None
                            else:
                                with torch.no_grad():
                                    teacher_forward_start = time.perf_counter()
                                    focus_context = None
                                    if focus_enabled:
                                        from verl.trainer.ppo.attention_focus import AttentionFocus
                                        focus_context = AttentionFocus(
                                            int(self_distillation_cfg.attention_focus_layer) - 1,
                                            float(self_distillation_cfg.attention_focus_rho),
                                            model_inputs["attention_focus_bbox"].tolist())
                                        teacher_inputs["response_mask"] = model_inputs["response_mask"]
                                        crop_started = time.perf_counter()
                                        self._forward_micro_batch(
                                            teacher_inputs, temperature=temperature,
                                            module=teacher_model, focus_context=focus_context)
                                        micro_batch_metrics["attention_focus/crop_forward_s"] = time.perf_counter()-crop_started
                                        teacher_inputs = {key: model_inputs[key] for key in
                                                          ("responses", "response_mask", "input_ids", "attention_mask", "position_ids", "multi_modal_inputs")}
                                    teacher_outputs = self._forward_micro_batch(
                                        teacher_inputs,
                                        focus_context=focus_context,
                                        focus_mode="route",
                                        temperature=temperature,
                                        calculate_entropy=False,
                                        return_all_logps=return_all_logps,
                                        distill_topk=distill_topk,
                                        topk_indices=None if use_three_view_union else student_topk_indices,
                                        return_native_topk=bool(distill_topk and not use_three_view_union and not correctness_entropy_enabled),
                                        entropy_target=({
                                            "student_entropy": outputs["reference_entropy"],
                                            "success": model_inputs["question_success"],
                                            "response_mask": response_mask,
                                            "tolerance": float(self_distillation_cfg.correctness_entropy_tolerance),
                                        } if correctness_entropy_enabled else None),
                                        return_response_logits=use_three_view_union,
                                        return_grounding=grounded_target_enabled,
                                        grounding_view="teacher",
                                        return_teacher_center=teacher_center_enabled,
                                        return_gap_attention=gap_lambda > 0,
                                        module=teacher_model,
                                    )
                                    teacher_forward_time = time.perf_counter() - teacher_forward_start
                                    if focus_context is not None:
                                        micro_batch_metrics.update(focus_context.metrics())
                                        del focus_context
                                stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                                if teacher_center_enabled:
                                    from verl.trainer.ppo.teacher_center_target import replace_attention_distance
                                    micro_batch_metrics.update(replace_attention_distance(
                                        outputs, teacher_outputs, model_inputs["evidence_bbox"],
                                        response_mask, float(self_distillation_cfg.evidence_attention_alpha),
                                        compute_auxiliary_loss=evidence_lambda > 0))
                                if "padding_fraction_removed" in teacher_outputs:
                                    micro_batch_metrics["perf/teacher_padding_removed_fraction"] = teacher_outputs["padding_fraction_removed"]
                                if "logit_position_fraction" in teacher_outputs:
                                    micro_batch_metrics["perf/teacher_logit_position_fraction"] = teacher_outputs["logit_position_fraction"]
                                teacher_log_prob = teacher_outputs["log_probs"]
                                teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                                teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                                if caption_lambda > 0:
                                    caption_start = time.perf_counter()
                                    caption_inputs = {
                                        key: model_inputs["caption_" + key]
                                        for key in ("input_ids", "attention_mask", "position_ids", "response_start_idx")
                                    }
                                    caption_inputs["responses"] = model_inputs["responses"]
                                    caption_inputs["response_mask"] = response_mask
                                    with torch.no_grad():
                                        caption_outputs = self._forward_micro_batch(
                                            caption_inputs, temperature=temperature, module=teacher_model,
                                            distill_topk=distill_topk, topk_indices=student_topk_indices,
                                        )
                                    caption_topk_logps = caption_outputs["topk_logps"]
                                    del caption_outputs
                                    caption_seconds = time.perf_counter() - caption_start
                                    stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += caption_seconds
                                    micro_batch_metrics["evidence/caption_forward_s"] = caption_seconds
                                teacher_native_topk_indices = (
                                    teacher_outputs.get("topk_indices")
                                    if use_three_view_union
                                    else teacher_outputs.get("native_topk_indices")
                                )
                                teacher_native_topk_logps = (
                                    teacher_outputs.get("topk_logps")
                                    if use_three_view_union
                                    else teacher_outputs.get("native_topk_logps")
                                )
                                if (
                                    distill_topk
                                    and student_topk_indices is not None
                                    and student_topk_logps is not None
                                    and teacher_native_topk_indices is not None
                                    and teacher_native_topk_logps is not None
                                ):
                                    alignment_mask = response_mask
                                    if self_distillation_mask is not None:
                                        alignment_mask = alignment_mask * self_distillation_mask.unsqueeze(1)
                                    if distillation_token_mask is not None:
                                        alignment_mask = alignment_mask * distillation_token_mask
                                    micro_batch_metrics.update(
                                        self._compute_opd_alignment_metrics(
                                            student_topk_indices,
                                            student_topk_logps,
                                            teacher_native_topk_indices,
                                            teacher_native_topk_logps,
                                            alignment_mask,
                                        )
                                    )
                            if union_teacher_topk:
                                from verl.trainer.ppo.teacher_union import positive_union
                                native_ids = teacher_native_topk_indices[..., :union_teacher_topk]
                                native_logps = teacher_native_topk_logps[..., :union_teacher_topk]
                                union_ids, distill_support_mask, union_positive = positive_union(
                                    student_topk_indices, native_ids, teacher_topk_logps, native_logps)
                                union_statistics.add(
                                    student_topk_indices, native_ids, teacher_topk_logps,
                                    union_positive, distill_support_mask, response_mask,
                                    model_inputs["world_side"])
                                # This gather carries Student gradients, outside Teacher no_grad.
                                student_topk_logps = self._gather_logps_from_response_logits(
                                    outputs.pop("response_logits"), union_ids, distill_support_mask)
                                teacher_topk_logps = union_positive
                                student_topk_indices = union_ids
                            teacher_minus_topk_logps = None
                            student_minus_topk_logps = None
                            opdv_visual_advantage_weights = None
                            if at_enabled or vc_target_enabled or opdv_target_enabled or er_target_enabled or ssp_target_enabled or ssp_v2_target_enabled or ver_target_enabled or dsvr_target_enabled or vgr_target_enabled:
                                if distill_topk is None:
                                    if vc_target_enabled or er_target_enabled or ver_target_enabled or dsvr_target_enabled or vgr_target_enabled:
                                        raise ValueError("visual target construction requires distillation_topk to be set.")
                                teacher_minus_inputs = {
                                    "responses": model_inputs["responses"],
                                    "input_ids": model_inputs["teacher_minus_input_ids"],
                                    "attention_mask": model_inputs["teacher_minus_attention_mask"],
                                    "position_ids": model_inputs["teacher_minus_position_ids"],
                                    "response_start_idx": model_inputs["teacher_minus_response_start_idx"],
                                }
                                if "teacher_minus_multi_modal_inputs" in model_inputs:
                                    teacher_minus_inputs["multi_modal_inputs"] = model_inputs[
                                        "teacher_minus_multi_modal_inputs"
                                    ]
                                if competitor_enabled:
                                    from verl.trainer.ppo.student_competitor import build_competitor_inputs
                                    competitor_start = time.perf_counter()
                                    negative_mm, available, competitor_metrics = build_competitor_inputs(
                                        self.processor, model_inputs["teacher_multi_modal_inputs"],
                                        model_inputs["competitor_metadata"], outputs.pop("competitor_attention"),
                                        model_inputs["evidence_bbox"], competitor_image_cache,
                                        device=model_inputs["input_ids"].device)
                                    teacher_minus_inputs["multi_modal_inputs"] = negative_mm
                                    outputs["attention_token_selected"] &= available[:, None]
                                    micro_batch_metrics.update(competitor_metrics)
                                    micro_batch_metrics["competitor/preparation_s"] = time.perf_counter() - competitor_start
                                with torch.no_grad():
                                    teacher_minus_forward_start = time.perf_counter()
                                    teacher_minus_outputs = self._forward_micro_batch(
                                        teacher_minus_inputs,
                                        temperature=temperature,
                                        calculate_entropy=False,
                                        return_all_logps=False,
                                        distill_topk=distill_topk if (at_enabled or vc_target_enabled or er_target_enabled or ver_target_enabled or dsvr_target_enabled or vgr_target_enabled) else None,
                                        topk_indices=(None if use_three_view_union else student_topk_indices)
                                        if (at_enabled or vc_target_enabled or er_target_enabled or ver_target_enabled or dsvr_target_enabled or vgr_target_enabled)
                                        else None,
                                        return_response_logits=use_three_view_union if (vc_target_enabled or er_target_enabled or ver_target_enabled) else False,
                                        module=teacher_model,
                                    )
                                    teacher_minus_forward_time = time.perf_counter() - teacher_minus_forward_start
                                    stage_wall_time_totals["timing_s/update_actor/teacher_minus_forward"] += (
                                        teacher_minus_forward_time
                                    )
                                    teacher_minus_outputs_list = [teacher_minus_outputs]
                                    if need_extra_teacher_views:
                                        for view_idx in range(1, counterfactual_view_count):
                                            teacher_minus_inputs = {
                                                "responses": model_inputs["responses"],
                                                "input_ids": model_inputs[f"teacher_minus_{view_idx}_input_ids"],
                                                "attention_mask": model_inputs[f"teacher_minus_{view_idx}_attention_mask"],
                                                "position_ids": model_inputs[f"teacher_minus_{view_idx}_position_ids"],
                                                "response_start_idx": model_inputs[
                                                    f"teacher_minus_{view_idx}_response_start_idx"
                                                ],
                                            }
                                            extra_mm_key = f"teacher_minus_{view_idx}_multi_modal_inputs"
                                            if extra_mm_key in model_inputs:
                                                teacher_minus_inputs["multi_modal_inputs"] = model_inputs[extra_mm_key]
                                            with torch.no_grad():
                                                teacher_minus_forward_start = time.perf_counter()
                                                extra_teacher_minus_outputs = self._forward_micro_batch(
                                                    teacher_minus_inputs,
                                                    temperature=temperature,
                                                    calculate_entropy=False,
                                                    return_all_logps=False,
                                                    distill_topk=distill_topk,
                                                    topk_indices=None if use_three_view_union else student_topk_indices,
                                                    return_response_logits=use_three_view_union,
                                                    module=teacher_model,
                                                )
                                                teacher_minus_forward_time = (
                                                    time.perf_counter() - teacher_minus_forward_start
                                                )
                                            stage_wall_time_totals["timing_s/update_actor/teacher_minus_forward"] += (
                                                teacher_minus_forward_time
                                            )
                                            teacher_minus_outputs_list.append(extra_teacher_minus_outputs)
                                    if local_target_enabled:
                                        from verl.trainer.ppo.local_evidence_target import replace_attention_distance
                                        micro_batch_metrics.update(replace_attention_distance(
                                            outputs, teacher_topk_logps,
                                            [view["topk_logps"] for view in teacher_minus_outputs_list[1:]],
                                            response_mask, float(self_distillation_cfg.evidence_attention_alpha)))
                                    teacher_minus_topk_logps = teacher_minus_outputs.get("topk_logps")
                                    student_minus_outputs = None
                                    if er_target_enabled:
                                        student_minus_forward_start = time.perf_counter()
                                        student_minus_outputs = self._forward_micro_batch(
                                            teacher_minus_inputs,
                                            temperature=temperature,
                                            calculate_entropy=False,
                                            return_all_logps=False,
                                            distill_topk=distill_topk,
                                            topk_indices=None,
                                            return_response_logits=True,
                                            module=self.actor_module,
                                        )
                                        stage_wall_time_totals["timing_s/update_actor/student_minus_forward"] += (
                                            time.perf_counter() - student_minus_forward_start
                                        )
                                    if opdv_target_enabled:
                                        opdv_visual_advantage_weights = torch.relu(
                                            (teacher_log_prob - teacher_minus_outputs["log_probs"]).detach()
                                        )
                                        opdv_mask = response_mask
                                        if self_distillation_mask is not None:
                                            opdv_mask = opdv_mask * self_distillation_mask.unsqueeze(1)
                                        opdv_count = opdv_mask.sum().clamp(min=1.0)
                                        micro_batch_metrics.update(
                                            {
                                                "self_distillation/opdv_margin_mean": (
                                                    (opdv_visual_advantage_weights * opdv_mask).sum() / opdv_count
                                                ).detach().item(),
                                                "self_distillation/opdv_active_token_fraction": (
                                                    ((opdv_visual_advantage_weights > 0).to(opdv_mask.dtype) * opdv_mask).sum()
                                                    / opdv_count
                                                ).detach().item(),
                                            }
                                        )
                                    if use_mean_vc_direction and not use_three_view_union:
                                        teacher_minus_topk_logps = torch.stack(
                                            [outputs_item.get("topk_logps") for outputs_item in teacher_minus_outputs_list],
                                            dim=0,
                                        )
                                    if use_three_view_union:
                                        teacher_topk_indices = teacher_outputs.get("topk_indices")
                                        student_minus_topk_indices = (
                                            student_minus_outputs.get("topk_indices")
                                            if student_minus_outputs is not None
                                            else None
                                        )
                                        teacher_minus_topk_indices_list = [
                                            outputs_item.get("topk_indices") for outputs_item in teacher_minus_outputs_list
                                        ]
                                        if (
                                            student_topk_indices is None
                                            or teacher_topk_indices is None
                                            or (er_target_enabled and student_minus_topk_indices is None)
                                            or any(item is None for item in teacher_minus_topk_indices_list)
                                        ):
                                            raise ValueError("three_view_union support requires top-k indices from all views.")
                                        union_indices, union_mask, union_size = self._build_topk_union_indices(
                                            student_topk_indices,
                                            teacher_topk_indices,
                                            *([student_minus_topk_indices] if student_minus_topk_indices is not None else []),
                                            *teacher_minus_topk_indices_list,
                                        )
                                        distill_support_mask = union_mask
                                        student_topk_logps = self._gather_logps_from_response_logits(
                                            outputs["response_logits"],
                                            union_indices,
                                            union_mask,
                                        )
                                        teacher_topk_logps = self._gather_logps_from_response_logits(
                                            teacher_outputs["response_logits"],
                                            union_indices,
                                            union_mask,
                                        ).detach()
                                        teacher_minus_topk_logps_list = [
                                            self._gather_logps_from_response_logits(
                                                outputs_item["response_logits"],
                                                union_indices,
                                                union_mask,
                                            ).detach()
                                            for outputs_item in teacher_minus_outputs_list
                                        ]
                                        if student_minus_outputs is not None:
                                            student_minus_topk_logps = self._gather_logps_from_response_logits(
                                                student_minus_outputs["response_logits"],
                                                union_indices,
                                                union_mask,
                                            ).detach()
                                        if use_mean_vc_direction:
                                            teacher_minus_topk_logps = torch.stack(
                                                teacher_minus_topk_logps_list,
                                                dim=0,
                                            )
                                        else:
                                            teacher_minus_topk_logps = teacher_minus_topk_logps_list[0]
                                        support_loss_mask = response_mask
                                        if self_distillation_mask is not None:
                                            support_loss_mask = support_loss_mask * self_distillation_mask.unsqueeze(1)
                                        support_loss_mask = support_loss_mask.to(torch.float32)
                                        valid_support_count = support_loss_mask.sum().clamp(min=1)
                                        union_size_for_metrics = union_size.to(support_loss_mask.dtype)
                                        support_prefix = "er" if er_target_enabled else "vc"
                                        micro_batch_metrics[
                                            f"self_distillation/{support_prefix}_support_union"
                                        ] = 1.0
                                        micro_batch_metrics[f"self_distillation/{support_prefix}_support_size.mean"] = (
                                            (union_size_for_metrics * support_loss_mask).sum() / valid_support_count
                                        ).detach().item()
                                        micro_batch_metrics[f"self_distillation/{support_prefix}_support_size.max"] = (
                                            torch.where(
                                                support_loss_mask > 0,
                                                union_size_for_metrics,
                                                torch.zeros_like(union_size_for_metrics),
                                            )
                                            .max()
                                            .detach()
                                            .item()
                                        )
                                        if use_mean_vc_direction:
                                            micro_batch_metrics["self_distillation/vc_direction_mean_view_count"] = float(
                                                counterfactual_view_count
                                            )
                            if (vc_target_enabled or er_target_enabled or ver_target_enabled) and use_three_view_union:
                                # The union support is built under teacher no-grad above, but student log-probs
                                # must be gathered outside that context so the visual loss can backpropagate.
                                student_topk_logps = self._gather_logps_from_response_logits(
                                    outputs["response_logits"],
                                    union_indices,
                                    union_mask,
                                )
                            if self_distillation_cfg.get("log_prob_dump_dir", None):
                                if distill_topk:
                                    student_distill_log_probs = student_topk_logps
                                    teacher_distill_log_probs = teacher_topk_logps
                                    if self_distillation_cfg.distillation_add_tail:
                                        student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                        teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                                else:
                                    student_distill_log_probs = student_all_logps
                                    teacher_distill_log_probs = teacher_all_logps

                                if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                    raise ValueError("Missing distillation log_probs for dump.")

                                loss_mask = response_mask
                                if self_distillation_mask is not None:
                                    loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                                if distillation_token_mask is not None:
                                    loss_mask = loss_mask * distillation_token_mask
                                valid_rows = loss_mask > 0
                                if valid_rows.any():
                                    distill_dump_chunks.append(
                                        {
                                            "student_log_probs": student_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                            "teacher_log_probs": teacher_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        }
                                    )
                            sa_filter_mask = None
                            if sa_target_enabled:
                                model_config = getattr(getattr(self.actor_module, "module", self.actor_module), "config", None)
                                dummy_token_id = getattr(model_config, "bos_token_id", None)
                                if dummy_token_id is None:
                                    dummy_token_id = getattr(model_config, "eos_token_id", None)
                                if isinstance(dummy_token_id, (list, tuple)):
                                    dummy_token_id = dummy_token_id[0]
                                if dummy_token_id is None:
                                    dummy_token_id = pad_token_id
                                dummy = torch.full(
                                    (model_inputs["responses"].shape[0], 1),
                                    int(dummy_token_id),
                                    dtype=model_inputs["responses"].dtype,
                                    device=model_inputs["responses"].device,
                                )
                                residual_input_ids = torch.cat([dummy, model_inputs["responses"]], dim=-1)
                                residual_attention_mask = torch.cat(
                                    [torch.ones_like(dummy, dtype=response_mask.dtype), response_mask],
                                    dim=-1,
                                )
                                residual_position_ids = residual_attention_mask.long().cumsum(dim=-1) - 1
                                residual_position_ids.masked_fill_(residual_attention_mask == 0, 0)
                                residual_inputs = {
                                    "responses": model_inputs["responses"],
                                    "input_ids": residual_input_ids,
                                    "attention_mask": residual_attention_mask,
                                    "position_ids": residual_position_ids,
                                    "response_start_idx": torch.ones(
                                        residual_input_ids.shape[0],
                                        device=residual_input_ids.device,
                                        dtype=torch.long,
                                    ),
                                }
                                with torch.no_grad():
                                    student_residual = self._forward_micro_batch(
                                        residual_inputs,
                                        temperature=temperature,
                                        module=self.actor_module,
                                    )["log_probs"]
                                    teacher_residual = self._forward_micro_batch(
                                        residual_inputs,
                                        temperature=temperature,
                                        module=teacher_model,
                                    )["log_probs"]
                                sa_filter_mask, sa_metrics = compute_sa_filter_mask(
                                    real_k1=(log_prob - teacher_log_prob).detach(),
                                    residual_k1=(student_residual - teacher_residual).detach(),
                                    response_mask=response_mask,
                                    low_sensitivity_fraction=float(
                                        self_distillation_cfg.get("sa_low_sensitivity_fraction", 0.2)
                                    ),
                                    extreme_loss_fraction=float(
                                        self_distillation_cfg.get("sa_extreme_loss_fraction", 0.3)
                                    ),
                                )
                                micro_batch_metrics.update(sa_metrics)

                            loss_compute_start = time.perf_counter()
                            if ssp_v2_target_enabled:
                                vopd_loss, vopd_metrics = compute_ssp_v2_listwise_loss(
                                    student_log_probs=log_prob,
                                    old_log_probs=old_log_prob,
                                    teacher_crop_log_probs=teacher_log_prob,
                                    teacher_full_log_probs=teacher_minus_outputs["log_probs"],
                                    response_mask=response_mask,
                                    branch_mask=model_inputs["distillation_token_mask"],
                                    group_ids=list(model_inputs["uid"]),
                                    candidate_count=int(self_distillation_cfg.get("ssp_candidate_count", 8)),
                                    beta=float(self_distillation_cfg.get("ssp_beta", 0.5)),
                                    trust_lambda=float(self_distillation_cfg.get("ssp_trust_lambda", 0.1)),
                                    candidate_token_ids=model_inputs["responses"],
                                )
                            elif ssp_target_enabled:
                                vopd_loss, vopd_metrics = compute_ssp_listwise_loss(
                                    student_log_probs=log_prob,
                                    old_log_probs=old_log_prob,
                                    teacher_crop_log_probs=teacher_log_prob,
                                    teacher_full_log_probs=teacher_minus_outputs["log_probs"],
                                    response_mask=response_mask,
                                    group_ids=list(model_inputs["uid"]),
                                    candidate_count=int(
                                        self_distillation_cfg.get("ssp_candidate_count", 4)
                                    ),
                                    chunk_tokens=int(self_distillation_cfg.get("ssp_chunk_tokens", 64)),
                                    beta=float(self_distillation_cfg.get("ssp_beta", 1.0)),
                                    reference_lambda=float(
                                        self_distillation_cfg.get("ssp_reference_lambda", 0.02)
                                    ),
                                )
                            elif correctness_entropy_enabled:
                                from verl.trainer.ppo.correctness_entropy import multiplicity_jsd
                                multiplicity = model_inputs["response_multiplicity"]
                                vopd_loss = multiplicity_jsd(student_topk_logps, teacher_topk_logps,
                                                           response_mask, multiplicity, correctness_denominator)
                                weights = response_mask * multiplicity[:, None]
                                vopd_metrics = {"correctness/jsd": vopd_loss.detach().item()}
                                for name in ("teacher_entropy", "target_entropy", "actual_entropy", "temperature", "entropy_residual"):
                                    vopd_metrics[f"correctness/{name}"] = ((teacher_outputs[name] * weights).sum() / weights.sum().clamp_min(1)).item()
                                vopd_metrics["correctness/entropy_residual_max"] = teacher_outputs["entropy_residual"][response_mask.bool()].max().item()
                                vopd_metrics["correctness/temperature_max"] = teacher_outputs["temperature"][response_mask.bool()].max().item()
                                vopd_metrics["correctness/student_entropy"] = ((outputs["reference_entropy"] * weights).sum() / weights.sum().clamp_min(1)).item()
                                vopd_metrics["correctness/softened_token_fraction"] = (((teacher_outputs["temperature"] > 1) * weights).sum() / weights.sum().clamp_min(1)).item()
                            else:
                                vopd_loss, vopd_metrics = compute_self_distillation_loss(
                                    student_log_probs=log_prob,
                                    teacher_log_probs=teacher_log_prob,
                                    response_mask=response_mask,
                                    self_distillation_config=self_distillation_cfg,
                                    old_log_probs=old_log_prob,
                                    student_all_log_probs=student_all_logps,
                                    teacher_all_log_probs=teacher_all_logps,
                                    student_topk_log_probs=student_topk_logps,
                                    student_minus_topk_log_probs=student_minus_topk_logps,
                                    teacher_topk_log_probs=teacher_topk_logps,
                                    teacher_minus_topk_log_probs=teacher_minus_topk_logps,
                                    dsvr_suffix_mask=model_inputs.get("dsvr_suffix_mask"),
                                    vgr_bridge_mask=model_inputs.get("vgr_bridge_mask"),
                                    vgr_suffix_mask=model_inputs.get("vgr_suffix_mask"),
                                    distill_support_mask=distill_support_mask,
                                    self_distillation_mask=self_distillation_mask,
                                    distillation_token_mask=distillation_token_mask,
                                    repair_token_mask=model_inputs.get("repair_token_mask"),
                                    repair_batch_denominator=repair_batch_denominator,
                                    loss_agg_mode=loss_agg_mode,
                                    rollout_is_weights=None if (svr_target_enabled or dsvr_target_enabled or vgr_target_enabled) else rollout_is_weights,
                                    va_rollout_weights=model_inputs.get("va_rollout_weights"),
                                    va_high_group_mask=model_inputs.get("va_high_group_mask"),
                                    vzero_rollout_weights=model_inputs.get("vzero_rollout_weights"),
                                    opdv_visual_advantage_weights=opdv_visual_advantage_weights,
                                    evidence_gap_weights=model_inputs.get("evidence_gap_weights"),
                                    attention_token_kl=outputs.get("attention_token_kl"),
                                    attention_token_selected=outputs.get("attention_token_selected"),
                                    ver_rollout_weights=model_inputs.get("ver_rollout_weights"),
                                    ver_loss_scale=model_inputs.get("ver_loss_scale"),
                                    sa_filter_mask=sa_filter_mask,
                                    uni_response_correct=model_inputs.get("uni_response_correct"),
                                    uni_sample_keep_mask=model_inputs.get("uni_sample_keep_mask"),
                                    uni_group_index=model_inputs.get("uni_group_index"),
                                    policy_config=self.config,
                                    batch_num_tokens=(eg_denominator if eg_enabled else
                                        vd_denominator if vd_enabled else self.config.global_batch_info.get("batch_num_tokens")),
                                    global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                                    loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                                )
                            # Main gap loss already has the global weighted-token
                            # denominator. Undo only its later microbatch scale;
                            # auxiliary Attention/Caption reductions stay unchanged.
                            if eg_enabled:
                                vopd_loss = vopd_loss / loss_scale_factor
                            if evidence_lambda > 0 or caption_lambda > 0:
                                alpha = float(self_distillation_cfg.alpha)
                                divergence_name = "fkl" if alpha == 0.0 else "rkl" if alpha == 1.0 else "jsd"
                                target_label = "reconstructed" if (at_enabled and
                                    self_distillation_cfg.get("attention_target_gate_mode", "adaptive") != "zero") else "crop"
                                vopd_metrics[f"evidence/{target_label}_{divergence_name}"] = vopd_loss.detach().item()
                                vopd_metrics[f"evidence/{target_label}_divergence"] = vopd_loss.detach().item()
                                vopd_metrics["evidence/distillation_alpha"] = alpha
                            opd_lambda = float(self_distillation_cfg.get("opd_loss_lambda", 1.0))
                            vopd_metrics["evidence/opd_lambda"] = opd_lambda
                            vopd_metrics["evidence/main_weighted"] = (opd_lambda * vopd_loss).detach().item()
                            vopd_loss = opd_lambda * vopd_loss
                            if at_enabled:
                                vopd_metrics.update(outputs["evidence_attention_metrics"])
                                vopd_metrics["evidence/attention_lambda"] = 0.0
                            if evidence_lambda > 0:
                                aux_attention = outputs["evidence_attention_loss"]
                                vopd_loss = vopd_loss + evidence_lambda * aux_attention
                                vopd_metrics["evidence/attention_lambda"] = evidence_lambda
                                vopd_metrics.update(outputs["evidence_attention_metrics"])
                                vopd_metrics["evidence/attention_weighted"] = (evidence_lambda * aux_attention).detach().item()
                            if caption_lambda > 0:
                                from verl.trainer.ppo.evidence_caption import caption_forward_kl
                                cap_tokens = caption_forward_kl(student_topk_logps, caption_topk_logps)
                                cap_loss = agg_loss(
                                    loss_mat=cap_tokens, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
                                    batch_num_tokens=self.config.global_batch_info.get("batch_num_tokens"),
                                    global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                                    loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                                )
                                vopd_loss = vopd_loss + caption_lambda * cap_loss
                                vopd_metrics["evidence/caption_kl"] = cap_loss.detach().item()
                                vopd_metrics["evidence/caption_weighted"] = (caption_lambda * cap_loss).detach().item()
                            if rank_tie_lambda > 0:
                                rank_logits = outputs["rank_tie_logits"]
                                width = rank_logits.shape[1]
                                rank_mask = response_mask[:, :width] * self_distillation_mask.unsqueeze(1)
                                if distillation_token_mask is not None:
                                    rank_mask = rank_mask * distillation_token_mask[:, :width]
                                rank_losses, rank_metrics = rank_tie_token_loss(
                                    rank_logits,
                                    teacher_outputs["native_topk_indices"][:, :width, 0],
                                    rank_mask,
                                )
                                rank_loss = agg_loss(
                                    loss_mat=rank_losses, loss_mask=rank_mask, loss_agg_mode=loss_agg_mode,
                                    batch_num_tokens=self.config.global_batch_info.get("batch_num_tokens"),
                                    global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                                    loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                                )
                                rank_metrics.update({
                                    "rank_tie/jsd_loss": vopd_loss.detach().item(),
                                    "rank_tie/aux_loss": rank_loss.detach().item(),
                                    "rank_tie/lambda": rank_tie_lambda,
                                })
                                vopd_loss = vopd_loss + rank_tie_lambda * rank_loss
                                vopd_metrics.update(rank_metrics)
                            if grounded_target_enabled:
                                from verl.trainer.ppo.grounded_opd import compute_grounding_loss

                                grounding_mask = response_mask
                                if self_distillation_mask is not None:
                                    grounding_mask = grounding_mask * self_distillation_mask.unsqueeze(1)
                                grounding_loss, grounding_metrics = compute_grounding_loss(
                                    student_probs=outputs["grounding_probs"],
                                    teacher_probs=teacher_outputs["grounding_probs"],
                                    response_mask=grounding_mask,
                                    bboxes=model_inputs["grounded_bbox"],
                                    grid_size=int(self_distillation_cfg.get("grounded_grid_size", 24)),
                                )
                                grounding_beta = float(self_distillation_cfg.get("grounded_beta", 1.0))
                                vopd_loss = vopd_loss + grounding_beta * grounding_loss
                                grounding_metrics.update(
                                    {
                                        "self_distillation/grounded_beta": grounding_beta,
                                        "self_distillation/grounded_selected_layer_count_student": float(
                                            outputs["grounding_layer_count"]
                                        ),
                                        "self_distillation/grounded_selected_layer_count_teacher": float(
                                            teacher_outputs["grounding_layer_count"]
                                        ),
                                    }
                                )
                                vopd_metrics.update(grounding_metrics)
                            loss_compute_time = time.perf_counter() - loss_compute_start
                            stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                            vopd_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                            micro_batch_metrics.update(vopd_metrics)

                            if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                                if advantages is None:
                                    raise ValueError(
                                        "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                    )
                                policy_loss_fn = get_policy_loss_fn("vanilla")
                                grpo_loss, grpo_metrics = policy_loss_fn(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=rollout_is_weights,
                                )
                                pg_loss = vopd_loss + grpo_loss
                                micro_batch_metrics.update(
                                    {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                                )
                            else:
                                grpo_loss = None
                                pg_loss = vopd_loss
                        else:
                            # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                            # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                            policy_loss_fn = get_policy_loss_fn(loss_mode)

                            # Compute policy loss (any function is expected to return 2 values)
                            pg_loss, pg_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            micro_batch_metrics.update(pg_metrics)

                        # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            # Compute metrics using CURRENT policy π_θ vs π_rollout
                            # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                        policy_loss = pg_loss
                        if calculate_entropy and entropy is not None:
                            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                            if entropy_coeff != 0:
                                policy_loss -= entropy_agg * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor
                        backward_start = time.perf_counter()
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        if self_distillation_enabled:
                            backward_time = time.perf_counter() - backward_start
                            stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                        if gap_lambda > 0:
                            from verl.trainer.ppo.attention_gap import build_auxiliary_batch, auxiliary_jsd
                            gap_start = time.perf_counter()
                            aux_inputs, active, gap_metrics = build_auxiliary_batch(
                                self.processor, model_inputs, outputs["gap_attention"], teacher_outputs["gap_attention"],
                                gap_cache, crop_size=int(self_distillation_cfg.attention_gap_crop_size),
                                crop_scale=float(self_distillation_cfg.attention_gap_crop_scale),
                            )
                            # Reuse q on the same original-Student top-100 support.
                            aux_outputs = self._forward_micro_batch(aux_inputs, temperature=temperature,
                                distill_topk=distill_topk, topk_indices=student_topk_indices.detach())
                            aux_mask = response_mask * self_distillation_mask[:, None] * active[:, None]
                            aux_tokens = auxiliary_jsd(aux_outputs["topk_logps"], teacher_topk_logps)
                            aux_loss = agg_loss(loss_mat=aux_tokens, loss_mask=aux_mask, loss_agg_mode=loss_agg_mode,
                                batch_num_tokens=self.config.global_batch_info.get("batch_num_tokens"),
                                global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                                loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"))
                            aux_scaled = gap_lambda * aux_loss * loss_scale_factor
                            if self.scaler is not None:
                                self.scaler.scale(aux_scaled).backward()
                            else:
                                aux_scaled.backward()
                            gap_metrics.update({"gap/main_jsd": vopd_loss.detach().item(),
                                "gap/aux_jsd": aux_loss.detach().item(), "gap/lambda": gap_lambda,
                                "timing_s/update_actor/gap_branch": time.perf_counter() - gap_start})
                            micro_batch_metrics.update(gap_metrics)
                            vopd_loss = vopd_loss.detach() + gap_lambda * aux_loss.detach()
                            pg_loss = vopd_loss

                        metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                        if self_distillation_enabled:
                            metrics["actor/vopd_loss"] += vopd_loss.detach().item() * loss_scale_factor
                            metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item() * loss_scale_factor
                            if grpo_loss is not None:
                                metrics["actor/grpo_loss"] += grpo_loss.detach().item() * loss_scale_factor
                        append_to_dict(metrics, micro_batch_metrics)


                from verl.trainer.ppo.counterfactual_pair_aux import backward_pair_aux
                append_to_dict(metrics, backward_pair_aux(self))

                optimizer_step_start = time.perf_counter()
                if self.config.get("train_vision_only", False) and not getattr(self, "_visual_grad_checked", False):
                    from verl.workers.visual_only import visual_gradient_stats

                    visual_metrics = visual_gradient_stats(self.actor_module)
                    if visual_metrics["visual_only/frozen_grad_tensors"]:
                        raise RuntimeError(f"Frozen language parameters received gradients: {visual_metrics}")
                    append_to_dict(metrics, visual_metrics)
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        print(f"[visual-only] first_backward={visual_metrics}", flush=True)
                    self._visual_grad_checked = True
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            teacher_update_rate = self._update_teacher()
            if teacher_update_rate is not None:
                metrics["actor/teacher_ema_update_rate"] = [teacher_update_rate]
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        if union_statistics is not None:
            metrics.update(union_statistics.metrics())
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics

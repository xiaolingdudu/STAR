# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode is "vopd".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        gamma (float): Weight applied to the SDPO loss.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "progressive".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        teacher_update_interval (Optional[int]): Hard-sync the teacher to the current student every N actor updates
            when teacher_regularization="progressive".
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
        teacher_always_on (bool): Whether to distill every sample directly from a teacher input instead of selecting successful samples by reward.
        teacher_model_source (str): Teacher source. Options: "legacy", "current" or "fixed".
        teacher_model_path (Optional[str]): Fixed teacher model path when teacher_model_source="fixed".
        teacher_image_key (Optional[str]): Dataset column holding teacher-side images for multimodal distillation.
        target_mode (str): Distillation target mode. "full_teacher" uses the original teacher distribution;
            "vc" builds the Visual-Correction OPD target q_T^VC; "va" keeps the full teacher target and
            applies VA-OPD (arXiv:2605.21924) visual-advantage weighting: rollout-level softmax
            reweighting by trajectory-averaged VA and token-level grouped KL (high-/low-VA groups);
            "vzero" applies V-Zero contrastive evidence gates to sampled-token reverse-KL OPD;
            "opdv" applies OPD-V positive/negative visual-teacher margin weighting; "ssp_opd"
            performs Student-branch listwise posterior distillation; "ver_opd"
            applies outcome-conditioned crop-vs-full visual evidence relay; "sa" applies SA-OPD
            residual-input filtering to sampled-token k1 OPD.
        counterfactual_teacher_image_key (Optional[str]): Dataset column holding x^- images for VC-OPD/VA-OPD.
        counterfactual_teacher_image_keys (Optional[list[str]]): Multiple x^- image columns for VC-OPD
            multi-direction variants. The first key is also used as the legacy single counterfactual key.
        vc_support_mode (str): Candidate support for VC-OPD. "student_topk" uses the existing
            student top-k support; "three_view_union" uses the union of student, teacher-plus,
            and teacher-minus top-k supports.
        vc_direction_mode (str): Visual direction construction for VC-OPD. "single" uses the legacy
            single counterfactual direction; "mean" averages multiple counterfactual directions after
            centering, yielding \bar u_t.
        vc_target_variant (str): VC-OPD target reconstruction variant. "single" uses the legacy
            one-direction projection; "budgeted_asvc" uses a budgeted asymmetric signed projection.
        vc_positive_budget_cap (float): Positive-branch budget cap for vc_target_variant="budgeted_asvc".
        va_probe_view (str): Which pair of teacher-scored conditions defines the VA probe.
            "teacher": evidence-present teacher view vs its degraded version (teacher_* vs
            teacher_minus_* inputs, e.g. bbox crop vs degraded crop) — consistent with the
            privileged-teacher pipeline. "student": the student's own deployment view vs its
            degraded version (student inputs scored by the teacher module vs teacher_minus_*
            inputs, which should then hold the degraded full image) — literal to the VA-OPD
            paper where the probe's evidence-present condition is the student's inference input.
        va_high_fraction (float): VA-OPD p_v, fraction of tokens per rollout assigned to the high-VA group.
        va_group_lambda (float): VA-OPD lambda, weight of the high-VA group's within-group KL mean.
        va_rollout_softmax_temp (float): VA-OPD tau, softmax temperature for rollout-level reweighting.
        va_rollout_reweight (bool): Enable VA-OPD rollout-level reweighting (ablation switch).
        va_token_grouping (bool): Enable VA-OPD token-level grouped KL (ablation switch).
        va_zscore_epsilon (float): Numerical-stability epsilon for VA z-score normalization.
        vzero_gate_gamma (float): Negative evidence-gap multiplier from the released V-Zero code.
        vzero_gate_alpha (float): Sibling z-score multiplier used by the released V-Zero gate.
            This is distinct from ``alpha``, which selects the distillation divergence.
        vzero_gate_min (float): Minimum V-Zero trajectory gate.
        vzero_gate_max (float): Maximum V-Zero trajectory gate.
        vzero_gate_epsilon (float): Numerical-stability epsilon for V-Zero sibling z-scores.
        fallback_to_policy_loss_on_missing_teacher (bool): When teacher_always_on=True, fall back to vanilla
            policy loss for samples whose teacher_image_key column is empty.
        log_prob_dump_dir (Optional[str]): Optional directory used to dump student/teacher log-prob tensors for each step.
    """

    visual_repair_enabled: bool = False
    visual_repair_max_tokens: int = 16
    visual_repair_token_weight: float = 1.0
    visual_repair_probe_batch_size: int = 1
    visual_repair_preference_margin: float = 1.098612289
    visual_repair_visual_margin: float = 0.693147181
    rank_tie_lambda: float = 0.0
    visual_dependency_enabled: bool = False
    visual_dependency_keep_fraction: float = 0.3
    visual_dependency_probe_scale: float = 0.5
    visual_dependency_chunk_tokens: int = 64
    correctness_entropy_enabled: bool = False
    correctness_entropy_tolerance: float = 1e-3
    attention_gap_lambda: float = 0.0
    attention_gap_crop_size: int = 1024
    attention_gap_crop_scale: float = 0.25
    attention_focus_enabled: bool = False
    attention_focus_layer: int = 12
    attention_focus_rho: float = 0.5
    attention_target_enabled: bool = False
    attention_target_beta: float = 0.25
    attention_target_gate_mode: str = "adaptive"
    attention_external_weight_enabled: bool = False
    attention_target_teacher_topk: int = 0
    attention_target_map: str = "gaussian"
    attention_target_negative_mode: str = "mean_erased"
    attention_target_region_mass: float = 0.9
    evidence_attention_lambda: float = 0.0
    evidence_attention_decay_start_step: Optional[int] = None
    evidence_attention_decay_end_step: Optional[int] = None
    evidence_attention_final_lambda: Optional[float] = None
    pair_aux_weight: float = 0.0
    pair_aux_manifest: str = ""
    evidence_attention_layer: int = 20  # one-based; 0 averages every full-attention layer
    evidence_attention_max_tokens: int = 64  # 0 includes all valid response tokens
    evidence_attention_alpha: float = 0.1
    caption_kl_lambda: float = 0.0
    opd_loss_lambda: float = 1.0
    evidence_gap_enabled: bool = False
    evidence_gap_token_beta: float = 0.5
    evidence_gap_trajectory_beta: float = 0.5
    full_logit_distillation: bool = True
    alpha: float = 0.0
    gamma: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    teacher_ema_freeze_after_step: Optional[int] = None
    teacher_update_interval: Optional[int] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_always_on: bool = False
    teacher_model_source: str = "legacy"
    teacher_model_path: Optional[str] = None
    teacher_image_key: Optional[str] = None
    target_mode: str = "full_teacher"
    counterfactual_teacher_image_key: Optional[str] = None
    counterfactual_teacher_image_keys: Optional[list[str]] = None
    vc_support_mode: str = "student_topk"
    vc_direction_mode: str = "single"
    vc_target_variant: str = "single"
    vc_positive_budget_cap: float = 0.8
    vc_projection_ridge: float = 1e-3
    vc_step_alpha: float = 1.0
    vc_max_logit_shift: Optional[float] = 20.0
    vc_max_shift_norm: Optional[float] = None
    vc_use_rho_gate: bool = False
    vc_rho_threshold: float = 0.0
    vc_rho_temperature: float = 1.0
    vc_nonvisual_anchor_eta: float = 0.0
    va_probe_view: str = "teacher"
    va_high_fraction: float = 0.2
    va_group_lambda: float = 0.5
    va_rollout_softmax_temp: float = 1.0
    va_rollout_reweight: bool = True
    va_token_grouping: bool = True
    va_zscore_epsilon: float = 1e-6
    vzero_gate_gamma: float = 1.0
    vzero_gate_alpha: float = 0.5
    vzero_gate_min: float = 0.0
    vzero_gate_max: float = 2.0
    vzero_gate_epsilon: float = 1e-6
    er_response_lambda: float = 1.0
    ssp_candidate_count: int = 4
    ssp_chunk_tokens: int = 64
    ssp_beta: float = 1.0
    ssp_reference_lambda: float = 0.02
    ssp_trunk_depths: list[int] = field(default_factory=lambda: [16, 32, 48, 64])
    ssp_trust_lambda: float = 0.1
    grounded_beta: float = 1.0
    grounded_grid_size: int = 24
    grounded_layer_start_fraction: float = 0.3
    grounded_layer_end_fraction: float = 0.6
    grounded_response_chunk_size: int = 16
    opdv_negative_mode: str = "random-mask"
    ver_success_threshold: float = 1.0
    ver_beta: float = 1.0
    ver_min_evidence: float = 0.02
    ver_relay_span: int = 64
    sa_low_sensitivity_fraction: float = 0.2
    sa_extreme_loss_fraction: float = 0.3
    uni_balance_ratio: float = 1.0
    uni_margin_delta: float = 0.4
    uni_margin_direction: str = "both"
    svr_prefix_ratio: float = 0.5
    dsvr_prefix_alpha: float = 0.5
    dsvr_planner_urls: Optional[list[str]] = None
    vgr_student_alpha: float = 1.0
    vgr_chunk_tokens: int = 128
    vgr_max_checkpoints: int = 2
    vgr_min_prefix_steps: int = 2
    vgr_bridge_max_tokens: int = 64
    vrbr_student_alpha: float = 1.0
    vrbr_chunk_tokens: int = 128
    vrbr_max_checkpoints: int = 2
    vrbr_min_prefix_tokens: int = 32
    vrbr_bridge_max_tokens: int = 64
    vrbr_boundary_window: int = 4
    vrbr_boundary_threshold: float = 0.35
    teacher_prompt_mode: Optional[str] = None
    answer_hint_template: str = (
        "\n\nHere is a reference solution to this problem:\n"
        "{answer}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
    )
    fallback_to_policy_loss_on_missing_teacher: bool = False
    log_prob_dump_dir: Optional[str] = None

    def __post_init__(self):
        if min(self.opd_loss_lambda, self.evidence_gap_token_beta, self.evidence_gap_trajectory_beta) < 0:
            raise ValueError("OPD coefficient and evidence-gap strengths must be nonnegative")
        if self.evidence_gap_enabled:
            if (self.target_mode != "full_teacher" or not self.teacher_always_on
                    or self.teacher_image_key != "bbox_images" or self.visual_dependency_enabled
                    or self.correctness_entropy_enabled or self.attention_focus_enabled):
                raise ValueError("Evidence-gap weighting keeps the always-on crop Teacher and cannot stack token filtering/target reconstruction")
        if self.attention_target_teacher_topk and (not self.attention_target_enabled
                or not 0 < self.attention_target_teacher_topk <= self.distillation_topk):
            raise ValueError("Teacher union requires attention target and 0 < Teacher k <= Student k")
        if self.attention_external_weight_enabled and (
                not self.attention_target_enabled or self.attention_target_gate_mode != "zero"):
            raise ValueError("External attention weighting requires the unreconstructed crop target (gate=zero)")
        if self.attention_target_enabled:
            if self.attention_target_gate_mode not in ("adaptive", "zero", "one", "shuffle_response"):
                raise ValueError("Unknown attention_target_gate_mode")
            if self.attention_target_negative_mode not in ("mean_erased", "student_competitor"):
                raise ValueError("Unknown attention_target_negative_mode")
            if self.attention_target_negative_mode == "student_competitor" and self.attention_target_map != "gaussian":
                raise ValueError("Student competitor uses the Gaussian attention gate")
            if self.attention_target_map not in ("gaussian", "local_occlusion", "region_projection", "teacher_center"):
                raise ValueError("Unknown attention_target_map")
            if self.attention_target_map == "region_projection" and not 0 < self.attention_target_region_mass < 1:
                raise ValueError("Region attention mass must lie strictly between zero and one")
            if (self.target_mode != "full_teacher" or not self.teacher_always_on
                    or self.teacher_image_key != "bbox_images" or not self.full_logit_distillation
                    or not self.distillation_topk or not self.distillation_add_tail
                    or (self.evidence_attention_lambda != 0 and not (
                        self.attention_target_map == "teacher_center"
                        and self.attention_target_gate_mode == "zero"
                        and not self.attention_external_weight_enabled))
                    or self.attention_target_beta < 0
                    or self.evidence_gap_enabled or self.visual_dependency_enabled
                    or self.correctness_entropy_enabled or self.attention_focus_enabled):
                raise ValueError("Attention target requires crop top-k+tail; its auxiliary control uses Teacher-center, gate=zero and no external weight")
        if self.evidence_attention_decay_start_step is not None:
            if (self.evidence_attention_decay_end_step is None
                    or self.evidence_attention_decay_end_step <= self.evidence_attention_decay_start_step
                    or self.evidence_attention_final_lambda is None
                    or not 0 <= self.evidence_attention_final_lambda <= self.evidence_attention_lambda):
                raise ValueError("Attention decay requires end_step > start_step and final_lambda in [0, initial_lambda]")
        if self.evidence_attention_lambda > 0 or self.caption_kl_lambda > 0:
            if self.target_mode != "full_teacher" or not self.teacher_always_on or not self.distillation_topk:
                raise ValueError("Evidence/caption auxiliaries require always-on full_teacher top-k distillation")
            if self.attention_focus_enabled:
                raise ValueError("Evidence supervision preserves the unmodified crop Teacher")
        if self.attention_focus_enabled:
            if self.target_mode != "full_teacher" or not 0 <= self.attention_focus_rho <= 1:
                raise ValueError("Attention-focus requires full_teacher target and rho in [0,1]")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        if self.gamma < 0.0:
            raise ValueError(f"self_distillation.gamma must be non-negative, got {self.gamma}")
        valid_teacher_regularization = ["ema", "trust-region", "progressive"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.teacher_update_interval is not None and self.teacher_update_interval <= 0:
            raise ValueError(
                "self_distillation.teacher_update_interval must be a positive integer "
                f"when set, got {self.teacher_update_interval}"
            )
        if self.teacher_ema_freeze_after_step is not None and self.teacher_ema_freeze_after_step < 0:
            raise ValueError("self_distillation.teacher_ema_freeze_after_step must be nonnegative")
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.teacher_prompt_mode is not None and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                f"self_distillation.teacher_prompt_mode must be None or 'answer_hint', got {self.teacher_prompt_mode}"
            )
        if self.teacher_always_on and not self.teacher_image_key and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                "self_distillation.teacher_image_key is required when teacher_always_on=True "
                "(unless teacher_prompt_mode='answer_hint')"
            )
        if self.visual_dependency_enabled:
            if self.target_mode != "full_teacher" or not self.teacher_always_on or not self.full_logit_distillation:
                raise ValueError("Visual dependency requires full_teacher distributional distillation")
            if not 0 < self.visual_dependency_keep_fraction <= 1 or not 0 < self.visual_dependency_probe_scale < 1:
                raise ValueError("Visual dependency requires a positive keep ratio and a degraded probe")
            if self.visual_dependency_chunk_tokens < 1:
                raise ValueError("Visual dependency chunk size must be positive")
        valid_target_modes = ["full_teacher", "vc", "va", "vzero", "opdv", "er_opd", "ssp_opd", "ssp_opd_v2", "grounded_opd", "ver_opd", "sa", "uni_opd", "svr_opd", "dsvr_opd", "vgr_relay", "vrbr_opd"]
        if self.target_mode not in valid_target_modes:
            raise ValueError(
                "self_distillation.target_mode must be one of "
                f"{valid_target_modes}, got {self.target_mode}"
            )
        if self.opdv_negative_mode != "random-mask":
            raise ValueError(
                "self_distillation.opdv_negative_mode currently supports only 'random-mask', "
                f"got {self.opdv_negative_mode}"
            )
        for name, value in (
            ("sa_low_sensitivity_fraction", self.sa_low_sensitivity_fraction),
            ("sa_extreme_loss_fraction", self.sa_extreme_loss_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"self_distillation.{name} must be in (0,1], got {value}")
        if self.target_mode == "vc":
            if not self.counterfactual_teacher_image_key:
                raise ValueError(
                    "self_distillation.counterfactual_teacher_image_key is required when target_mode='vc'"
                )
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError(
                    "VC-OPD requires full_logit_distillation=True and a finite distillation_topk "
                    "for the first top-k implementation."
                )
            valid_vc_support_modes = ["student_topk", "three_view_union"]
            if self.vc_support_mode not in valid_vc_support_modes:
                raise ValueError(
                    "self_distillation.vc_support_mode must be one of "
                    f"{valid_vc_support_modes}, got {self.vc_support_mode}"
                )
            valid_vc_direction_modes = ["single", "mean"]
            if self.vc_direction_mode not in valid_vc_direction_modes:
                raise ValueError(
                    "self_distillation.vc_direction_mode must be one of "
                    f"{valid_vc_direction_modes}, got {self.vc_direction_mode}"
                )
            if self.vc_direction_mode == "mean":
                keys = self.counterfactual_teacher_image_keys or []
                if len(keys) < 2:
                    raise ValueError(
                        "self_distillation.vc_direction_mode='mean' requires at least two "
                        "counterfactual_teacher_image_keys."
                    )
            valid_vc_target_variants = ["single", "budgeted_asvc"]
            if self.vc_target_variant not in valid_vc_target_variants:
                raise ValueError(
                    "self_distillation.vc_target_variant must be one of "
                    f"{valid_vc_target_variants}, got {self.vc_target_variant}"
                )
            if not 0.0 <= self.vc_positive_budget_cap <= 1.0:
                raise ValueError(
                    "self_distillation.vc_positive_budget_cap must be in [0,1], "
                    f"got {self.vc_positive_budget_cap}"
                )
        if self.target_mode == "va":
            if not self.counterfactual_teacher_image_key:
                raise ValueError(
                    "self_distillation.counterfactual_teacher_image_key is required when target_mode='va' "
                    "(it provides the degraded view used to compute visual advantage)."
                )
            if self.va_probe_view not in ("teacher", "student"):
                raise ValueError(
                    f"self_distillation.va_probe_view must be 'teacher' or 'student', got {self.va_probe_view}"
                )
            if not 0.0 < self.va_high_fraction < 1.0:
                raise ValueError(
                    f"self_distillation.va_high_fraction must be in (0,1), got {self.va_high_fraction}"
                )
            if not 0.0 <= self.va_group_lambda <= 1.0:
                raise ValueError(
                    f"self_distillation.va_group_lambda must be in [0,1], got {self.va_group_lambda}"
                )
            if self.va_rollout_softmax_temp <= 0.0:
                raise ValueError(
                    "self_distillation.va_rollout_softmax_temp must be positive, "
                    f"got {self.va_rollout_softmax_temp}"
                )
            if self.va_zscore_epsilon <= 0.0:
                raise ValueError(
                    f"self_distillation.va_zscore_epsilon must be positive, got {self.va_zscore_epsilon}"
                )
        if self.target_mode == "vzero":
            if not self.teacher_always_on:
                raise ValueError("V-Zero requires self_distillation.teacher_always_on=True.")
            if not self.counterfactual_teacher_image_key:
                raise ValueError(
                    "self_distillation.counterfactual_teacher_image_key is required when target_mode='vzero'."
                )
            if self.full_logit_distillation or self.distillation_topk is not None:
                raise ValueError(
                    "V-Zero uses the released sampled-token reverse-KL estimator and therefore requires "
                    "full_logit_distillation=False and distillation_topk=null."
                )
            if self.alpha != 1.0:
                raise ValueError(
                    "V-Zero sampled-token reverse-KL requires self_distillation.alpha=1.0. "
                    "Use vzero_gate_alpha to control the evidence-gate scale."
                )
        if self.target_mode == "opdv" and not self.full_logit_distillation:
            raise ValueError("OPD-V requires full_logit_distillation=True for its weighted JSD target.")
        if self.target_mode == "er_opd":
            if not self.counterfactual_teacher_image_key:
                raise ValueError("Evidence-Response OPD requires counterfactual_teacher_image_key.")
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("Evidence-Response OPD requires full-logit top-k distillation.")
            if self.distillation_add_tail:
                raise ValueError("Evidence-Response OPD requires distillation_add_tail=False.")
            if self.er_response_lambda < 0.0:
                raise ValueError("self_distillation.er_response_lambda must be non-negative.")
        if self.target_mode in {"ssp_opd", "ssp_opd_v2"}:
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError("SSP-OPD requires an always-on privileged crop Teacher view.")
            if self.full_logit_distillation or self.distillation_topk is not None:
                raise ValueError("SSP-OPD uses sequence scores and requires full_logit_distillation=False and distillation_topk=null.")
            if self.ssp_candidate_count < 2 or self.ssp_chunk_tokens <= 0:
                raise ValueError("SSP-OPD candidate_count must be at least two and chunk_tokens must be positive.")
            if self.ssp_beta < 0.0 or self.ssp_reference_lambda < 0.0:
                raise ValueError("SSP-OPD beta and reference_lambda must be non-negative.")
            if self.target_mode == "ssp_opd_v2":
                if not self.ssp_trunk_depths or min(self.ssp_trunk_depths) <= 0:
                    raise ValueError("SSP-OPD v2 requires positive trunk depths.")
                if self.ssp_trust_lambda < 0.0:
                    raise ValueError("SSP-OPD v2 trust_lambda must be non-negative.")
        if self.target_mode == "grounded_opd":
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError("Grounded OPD requires an always-on privileged crop teacher view.")
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("Grounded OPD requires OPSD top-k logit distillation.")
            if self.grounded_beta < 0.0 or self.grounded_grid_size <= 1:
                raise ValueError("Grounded OPD beta must be non-negative and grid_size must exceed one.")
            if not 0.0 <= self.grounded_layer_start_fraction < self.grounded_layer_end_fraction <= 1.0:
                raise ValueError("Grounded OPD layer fractions must satisfy 0 <= start < end <= 1.")
            if self.grounded_response_chunk_size <= 0:
                raise ValueError("Grounded OPD response_chunk_size must be positive.")
        if self.target_mode == "ver_opd":
            if not self.counterfactual_teacher_image_key:
                raise ValueError(
                    "VER-OPD requires counterfactual_teacher_image_key for the full-image teacher view."
                )
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("VER-OPD requires full-logit top-k distillation.")
            if self.ver_success_threshold <= 0.0:
                raise ValueError("self_distillation.ver_success_threshold must be positive.")
            if self.ver_beta < 0.0 or self.ver_min_evidence < 0.0:
                raise ValueError("VER-OPD beta and minimum evidence must be non-negative.")
            if self.ver_relay_span <= 0:
                raise ValueError("self_distillation.ver_relay_span must be positive.")
        if self.target_mode == "sa":
            if self.full_logit_distillation or self.distillation_topk is not None or self.alpha != 1.0:
                raise ValueError(
                    "SA-OPD uses sampled-token k1 policy-gradient distillation and requires "
                    "full_logit_distillation=False, distillation_topk=null, and alpha=1.0."
                )
        if self.target_mode == "uni_opd":
            if self.full_logit_distillation or self.distillation_topk is not None or self.alpha != 1.0:
                raise ValueError(
                    "Uni-OPD uses sampled-token on-policy distillation and requires "
                    "full_logit_distillation=False, distillation_topk=null, and alpha=1.0."
                )
            if self.uni_balance_ratio <= 0:
                raise ValueError("self_distillation.uni_balance_ratio must be positive.")
            if self.uni_margin_delta < 0:
                raise ValueError("self_distillation.uni_margin_delta must be non-negative.")
            if self.uni_margin_direction not in {"correct_up", "incorrect_down", "both"}:
                raise ValueError(
                    "self_distillation.uni_margin_direction must be correct_up, incorrect_down, or both."
                )
        if self.target_mode == "svr_opd":
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("SVR-OPD requires full-logit top-k distillation.")
            if self.alpha != 0.0:
                raise ValueError("SVR-OPD privilege erasure uses Forward KL and requires alpha=0.0.")
            if self.is_clip is not None:
                raise ValueError("SVR-OPD crop-assisted suffix is off-policy for the no-crop student; is_clip must be null.")
            if not 0.0 < self.svr_prefix_ratio < 1.0:
                raise ValueError("self_distillation.svr_prefix_ratio must be in (0,1).")
        if self.target_mode == "dsvr_opd":
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("DSVR-OPD requires full-logit top-k distillation.")
            if self.alpha != 0.0 or self.is_clip is not None:
                raise ValueError("DSVR-OPD uses ordinary OPD on the prefix and Forward KL on the suffix without IS.")
            if not 0.0 <= self.dsvr_prefix_alpha <= 1.0:
                raise ValueError("self_distillation.dsvr_prefix_alpha must be in [0,1].")
            if not self.dsvr_planner_urls:
                raise ValueError("DSVR-OPD requires at least one dsvr_planner_urls endpoint.")
        if self.target_mode == "vgr_relay":
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("VGR-Relay requires full-logit top-k distillation.")
            if self.alpha != 0.0 or self.is_clip is not None:
                raise ValueError("VGR-Relay uses segment-specific KL without rollout importance weighting.")
            if not 0.0 <= self.vgr_student_alpha <= 1.0:
                raise ValueError("self_distillation.vgr_student_alpha must be in [0,1].")
            if self.vgr_chunk_tokens <= 0 or self.vgr_max_checkpoints <= 0:
                raise ValueError("VGR-Relay chunk_tokens and max_checkpoints must be positive.")
            if self.vgr_min_prefix_steps <= 0 or self.vgr_bridge_max_tokens <= 0:
                raise ValueError("VGR-Relay min_prefix_steps and bridge_max_tokens must be positive.")
            if not self.dsvr_planner_urls:
                raise ValueError("VGR-Relay requires at least one dsvr_planner_urls endpoint.")
        if self.target_mode == "vrbr_opd":
            if not self.full_logit_distillation or self.distillation_topk is None:
                raise ValueError("VRBR-OPD requires full-logit top-k distillation.")
            if self.alpha != 0.0 or self.is_clip is not None:
                raise ValueError("VRBR-OPD uses segment-specific KL without rollout importance weighting.")
            if not 0.0 <= self.vrbr_student_alpha <= 1.0:
                raise ValueError("self_distillation.vrbr_student_alpha must be in [0,1].")
            if self.vrbr_chunk_tokens <= 0 or self.vrbr_max_checkpoints <= 0:
                raise ValueError("VRBR-OPD chunk_tokens and max_checkpoints must be positive.")
            if self.vrbr_min_prefix_tokens <= 0 or self.vrbr_bridge_max_tokens <= 0:
                raise ValueError("VRBR-OPD min_prefix_tokens and bridge_max_tokens must be positive.")
            if self.vrbr_boundary_window <= 0 or self.vrbr_boundary_threshold < 0.0:
                raise ValueError("VRBR-OPD boundary window must be positive and threshold non-negative.")
            if not self.dsvr_planner_urls:
                raise ValueError("VRBR-OPD requires at least one 9B scoring endpoint.")
        if self.vzero_gate_gamma < 0.0:
            raise ValueError(
                f"self_distillation.vzero_gate_gamma must be non-negative, got {self.vzero_gate_gamma}"
            )
        if self.vzero_gate_alpha < 0.0:
            raise ValueError(
                f"self_distillation.vzero_gate_alpha must be non-negative, got {self.vzero_gate_alpha}"
            )
        if self.vzero_gate_min < 0.0:
            raise ValueError(
                f"self_distillation.vzero_gate_min must be non-negative, got {self.vzero_gate_min}"
            )
        if self.vzero_gate_max < self.vzero_gate_min:
            raise ValueError(
                "self_distillation.vzero_gate_max must be greater than or equal to vzero_gate_min, "
                f"got {self.vzero_gate_max} < {self.vzero_gate_min}"
            )
        if self.vzero_gate_epsilon <= 0.0:
            raise ValueError(
                f"self_distillation.vzero_gate_epsilon must be positive, got {self.vzero_gate_epsilon}"
            )
        if self.vc_projection_ridge <= 0.0:
            raise ValueError(f"self_distillation.vc_projection_ridge must be positive, got {self.vc_projection_ridge}")
        if self.vc_step_alpha < 0.0:
            raise ValueError(f"self_distillation.vc_step_alpha must be non-negative, got {self.vc_step_alpha}")
        if self.vc_max_logit_shift is not None and self.vc_max_logit_shift <= 0.0:
            raise ValueError(f"self_distillation.vc_max_logit_shift must be positive, got {self.vc_max_logit_shift}")
        if self.vc_max_shift_norm is not None and self.vc_max_shift_norm <= 0.0:
            raise ValueError(f"self_distillation.vc_max_shift_norm must be positive, got {self.vc_max_shift_norm}")
        if self.vc_rho_temperature <= 0.0:
            raise ValueError(f"self_distillation.vc_rho_temperature must be positive, got {self.vc_rho_temperature}")
        if self.vc_nonvisual_anchor_eta < 0.0:
            raise ValueError(
                "self_distillation.vc_nonvisual_anchor_eta must be non-negative, "
                f"got {self.vc_nonvisual_anchor_eta}"
            )
        valid_teacher_model_source = ["legacy", "current", "fixed"]
        if self.teacher_model_source not in valid_teacher_model_source:
            raise ValueError(
                "self_distillation.teacher_model_source must be one of "
                f"{valid_teacher_model_source}, got {self.teacher_model_source}"
            )
        if self.teacher_model_source == "fixed" and not self.teacher_model_path:
            raise ValueError("self_distillation.teacher_model_path is required when teacher_model_source='fixed'")
        if self.teacher_regularization == "progressive":
            if self.teacher_model_source != "legacy":
                raise ValueError(
                    "self_distillation.teacher_regularization='progressive' requires "
                    "teacher_model_source='legacy'"
                )
            if self.teacher_update_interval is None:
                raise ValueError(
                    "self_distillation.teacher_update_interval is required when "
                    "teacher_regularization='progressive'"
                )


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'vopd'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    prune_update_payload: bool = False
    share_multimodal_inputs: bool = False
    defer_gradient_sync: bool = False
    trim_padding: bool = False
    sort_microbatches_by_length: bool = False
    response_only_logits: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    train_vision_only: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )

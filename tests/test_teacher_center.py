"""CPU checks of native-QK centroids, the detached gate, and actor forward wiring."""
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.evidence_caption import gaussian_target
from verl.trainer.ppo.teacher_center_target import (
    TeacherCenterCapture, centered_gaussians, map_centroids, replace_attention_distance,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor


def test_centroid_matches_per_head_softmax_and_registered_crop_mapping():
    torch.manual_seed(17)
    ids = torch.tensor([[0, 0, 99, 99, 99, 99, 3, 4, 5, 6]])
    mask = torch.tensor([[1, 1, 0]])
    # Two response prediction rows; the final padded position must stay unselected.
    positions = torch.tensor([[6, 7, 8]])
    q = torch.randn(1, 4, 10, 3, requires_grad=True)
    k = torch.randn(1, 2, 10, 3, requires_grad=True)
    footprint = torch.tensor([[.1, .2, .9, .6]])
    capture = TeacherCenterCapture(input_ids=ids, attention_mask=ids.ne(0),
        response_positions=positions, image_grid_thw=torch.tensor([[1, 2, 2]]),
        image_token_id=99, spatial_merge_size=1, total_layers=20,
        target_layer_idx=19, expected_layer_indices=(19,), response_mask=mask,
        crop_footprints=footprint, response_chunk_size=1)
    capture.add(SimpleNamespace(layer_idx=19), q, k, scaling=.7, softcap=None)
    centers, selected = capture.finish()
    for t in range(2):
        # Explicit head loop avoids accidentally mixing heads before softmax.
        probs = torch.stack([(q[0, h, positions[0, t]] @ k[0, h // 2, 2:6].T * .7)
                             .softmax(-1) for h in range(4)]).mean(0)
        expected = map_centroids(probs[None], 2, 2, footprint[0])[0]
        torch.testing.assert_close(centers[0, t], expected)
    assert selected.tolist() == [[True, True, False]]
    assert not centers.requires_grad
    assert centers[0, 2].count_nonzero() == 0


def test_gaussian_mass_center_and_gate_match_explicit_formula():
    box = torch.tensor([.2, .2, .8, .8])
    means = torch.tensor([[.5, .5], [.7, .3]], requires_grad=True)
    targets = centered_gaussians(5, 7, box, means.detach(), .2)
    torch.testing.assert_close(targets.sum(-1), torch.ones(2))
    assert (targets >= .2 / 35).all()
    torch.testing.assert_close(targets[0], gaussian_target(5, 7, box, alpha=.2))
    assert targets[0].argmax() != targets[1].argmax()
    attention = torch.arange(1, 36, dtype=torch.float32).repeat(2, 1)
    attention = attention / attention.sum(-1, keepdim=True)
    offsets = torch.tensor([0, 2])
    outputs = dict(center_attention=[(offsets, attention, 5, 7)],
        attention_token_kl=torch.ones(1, 4), evidence_attention_metrics={})
    tc = torch.zeros(1, 4, 2)
    tc[0, offsets] = means
    teacher = dict(teacher_attention_centers=tc, teacher_center_selected=torch.tensor([[1, 0, 1, 0]]))
    metrics = replace_attention_distance(outputs, teacher, box[None], torch.tensor([[1, 0, 1, 0]]), .2)
    expected = (attention * (attention.log() - targets.log())).sum(-1)
    torch.testing.assert_close(outputs['attention_token_kl'][0, offsets], expected)
    assert not outputs['attention_token_kl'].requires_grad
    assert outputs['attention_token_kl'][0, [1, 3]].count_nonzero() == 0
    assert metrics['teacher_center/background_alpha'] == .2


class ToyForward(torch.nn.Module):
    """Exercise production actor trimming/capture/output contracts, without loading a model."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(157, 8)
        self.head = torch.nn.Linear(8, 157)
        self.calls = 0
        self.config = SimpleNamespace(model_type="qwen3_5", image_token_id=99,
            text_config=SimpleNamespace(layer_types=['full_attention'] * 20, num_hidden_layers=20),
            vision_config=SimpleNamespace(spatial_merge_size=1))

    def forward(self, input_ids, attention_mask, position_ids, image_grid_thw, use_cache, logits_to_keep):
        from verl.trainer.ppo.grounded_opd import _grounded_wrapper
        self.calls += 1
        hidden = self.embed(input_ids)
        q = hidden[:, None].repeat(1, 2, 1, 1)
        k = hidden[:, None]
        # Same QK interception used by the deployed attention patch.
        _grounded_wrapper(lambda *args, **kwargs: None)(
            SimpleNamespace(layer_idx=19), q, k, k, None, scaling=8**-.5)
        return SimpleNamespace(logits=self.head(hidden[:, logits_to_keep]))


def test_actor_positive_only_capture_trimming_and_gate_dataflow():
    torch.manual_seed(42)
    model = ToyForward()
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.config = OmegaConf.create(dict(trim_padding=True, response_only_logits=True,
        self_distillation=dict(evidence_attention_layer=20, evidence_attention_max_tokens=0,
            evidence_attention_alpha=.2, attention_target_enabled=True,
            attention_target_map='teacher_center')))
    actor.device_name, actor.param_dtype = 'cpu', torch.bfloat16
    actor.use_fused_kernels = actor.use_remove_padding = actor.use_ulysses_sp = False
    actor.use_prefix_grouper = False
    ids = torch.tensor([[0, 0, 99, 99, 99, 99, 7, 8, 9, 10, 0, 0]])
    mm = np.empty(1, dtype=object)
    mm[0] = {'image_grid_thw': torch.tensor([[1, 2, 2]])}
    mask = torch.tensor([[1, 1, 0, 0]])
    batch = dict(input_ids=ids, attention_mask=ids.ne(0), position_ids=torch.arange(12)[None],
        responses=ids[:, 8:], response_mask=mask, response_start_idx=torch.tensor([8]),
        multi_modal_inputs=mm, evidence_bbox=torch.tensor([[.2, .2, .8, .8]]),
        attention_crop_footprint=torch.tensor([[.1, .1, .9, .9]]))
    student = actor._forward_micro_batch(batch, temperature=1., distill_topk=100, return_evidence_attention=True)
    with torch.no_grad():
        teacher = actor._forward_micro_batch(batch, temperature=1., distill_topk=100, return_teacher_center=True)
        negative = actor._forward_micro_batch(batch, temperature=1., distill_topk=100)
    assert model.calls == 3
    assert 'teacher_attention_centers' not in negative
    assert teacher['teacher_attention_centers'].shape == (1, 4, 2)
    assert teacher['teacher_center_selected'].tolist() == [[True, True, False, False]]
    assert 'local_evidence_attention' not in student
    metrics = replace_attention_distance(student, teacher, batch['evidence_bbox'], mask, .2)
    assert metrics['teacher_center/extra_model_forwards'] == 0
    assert student['attention_token_kl'].shape == mask.shape
    assert not student['attention_token_kl'].requires_grad
    assert student['topk_logps'].requires_grad
    student['topk_logps'][0, :2].sum().backward()
    assert model.head.weight.grad is not None

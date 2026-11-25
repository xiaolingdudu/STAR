import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.core_algos import compute_self_distillation_loss
from verl.trainer.ppo.teacher_union import positive_union, SupportStatistics
from verl.workers.actor.dp_actor import DataParallelPPOActor


def example():
    torch.manual_seed(42)
    student = torch.randn(2, 3, 157)
    student[..., 156] = -7
    teacher = student.clone()
    teacher[1, :, 156] = 5
    negative = torch.randn_like(student)
    return student.requires_grad_(), teacher, negative


def test_union_matches_full_vocabulary_gather_and_keeps_missing_teacher_token():
    s, t, _ = example()
    si, ti = s.topk(100).indices, t.topk(20).indices
    tq = t.log_softmax(-1)
    ui, mask, uq = positive_union(si, ti, tq.gather(-1, si), tq.gather(-1, ti))
    torch.testing.assert_close(uq[mask], tq.gather(-1, ui)[mask])
    assert not uq.requires_grad
    assert (mask[0].sum(-1) == 100).all()
    assert (mask[1].sum(-1) == 101).all()
    assert not (si[1] == 156).any()
    for b in range(2):
        for j in range(3):
            assert set(ui[b, j, mask[b, j]].tolist()) == set(si[b, j].tolist()) | set(ti[b, j].tolist())


def test_production_js_value_and_student_gradient_match_dense_coarsening():
    s, t, n = example()
    si, ti = s.detach().topk(100).indices, t.topk(20).indices
    tq, nq = t.log_softmax(-1), n.log_softmax(-1)
    ui, mask, uq = positive_union(si, ti, tq.gather(-1, si), tq.gather(-1, ti))
    sq = DataParallelPPOActor._gather_logps_from_response_logits(s, ui, mask)
    negative = nq.gather(-1, ui).masked_fill(~mask, -1e9)
    cfg = OmegaConf.create(dict(full_logit_distillation=True, distillation_topk=100,
        distillation_add_tail=True, target_mode='full_teacher', alpha=.5,
        attention_target_enabled=True, attention_target_beta=1., is_clip=None))
    response = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    distance = torch.full_like(response, 2.)
    loss, _ = compute_self_distillation_loss(
        torch.zeros_like(response), torch.zeros_like(response), response, cfg,
        student_topk_log_probs=sq, teacher_topk_log_probs=uq,
        teacher_minus_topk_log_probs=negative, distill_support_mask=mask,
        attention_token_kl=distance, attention_token_selected=response.bool())
    actual_grad = torch.autograd.grad(loss, s)[0]
    terms = []
    for b, j in response.nonzero().tolist():
        ids = ui[b, j, mask[b, j]]
        def coarsen(prob):
            head = prob[ids]
            return torch.cat((head, (1 - head.sum()).reshape(1)))
        p = coarsen(s[b, j].softmax(-1))
        qp, qm = coarsen(t[b, j].softmax(-1)), coarsen(n[b, j].softmax(-1))
        target = (qp.log() + (2 / 3) * (qp.log() - qm.log())).softmax(-1).detach()
        mean = (p + target) / 2
        terms.append(.5 * ((p * (p.log() - mean.log())).sum() +
                            (target * (target.log() - mean.log())).sum()))
    expected = torch.stack(terms).mean()
    expected_grad = torch.autograd.grad(expected, s)[0]
    torch.testing.assert_close(loss, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-5)
    assert (actual_grad[1, :, 156] < 0).all()
    assert actual_grad[0, 2].abs().sum() == 0


def test_diagnostics_count_valid_tokens_across_microbatch_splits():
    s, t, _ = example()
    si, ti = s.detach().topk(100).indices, t.topk(20).indices
    tq = t.log_softmax(-1)
    ui, mask, uq = positive_union(si, ti, tq.gather(-1, si), tq.gather(-1, ti))
    response = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    full, split = SupportStatistics('cpu'), SupportStatistics('cpu')
    args = [si, ti, tq.gather(-1, si), uq, mask, response]
    full.add(*args, ['original', 'edited'])
    for i, side in enumerate(['original', 'edited']):
        split.add(*(v[i:i+1] for v in args), [side])
    assert full.metrics() == split.metrics()
    m = full.metrics()
    assert m['support_union/original/all/tokens'] == 1
    assert m['support_union/edited/all/tokens'] == 3
    assert m['support_union/original/all/teacher_top1_miss_rate'] == 0
    assert m['support_union/edited/all/teacher_top1_miss_rate'] == 1
    assert m['support_union/edited/all/teacher_coverage_union'] > m['support_union/edited/all/teacher_coverage_student100']

"""Student candidates plus positive-crop Teacher candidates, with a shared tail.

The positive Teacher already scores Student candidates and returns native top-k.
Merge those small tensors; neither an extra forward nor full Teacher logits are
needed. The Student gather must remain outside the frozen Teacher context.
"""
import torch
import torch.distributed as dist


@torch.no_grad()
def positive_union(student_ids, teacher_ids, teacher_on_student, teacher_native):
    ids, order = torch.cat((student_ids, teacher_ids), -1).sort(-1)
    logps = torch.cat((teacher_on_student, teacher_native), -1).gather(-1, order)
    unique = torch.ones_like(ids, dtype=torch.bool)
    unique[..., 1:] = ids[..., 1:] != ids[..., :-1]
    size = unique.sum(-1)
    width = int(size.max())
    ranks = unique.cumsum(-1) - 1
    indices = ids.new_zeros(*ids.shape[:-1], width)
    positive = logps.new_full(indices.shape, -1e9)
    # Duplicate entries have the same ID and the same Teacher probability.
    indices.scatter_(-1, ranks, ids)
    positive.scatter_(-1, ranks, logps)
    mask = torch.arange(width, device=ids.device) < size[..., None]
    return indices, mask, positive.masked_fill(~mask, -1e9)


class SupportStatistics:
    """Token-weighted diagnostics, summed over microbatches and all actor ranks."""
    sides = ('original', 'edited')
    scopes = ('all', 'top1_disagreement', 'last16_tokens')

    def __init__(self, device):
        # count, teacher-top1 misses, old coverage, union coverage, support size
        self.sums = torch.zeros(2, 3, 5, device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, student_ids, teacher_ids, teacher_on_student, union_positive,
            union_mask, response_mask, world_sides):
        valid = response_mask.bool()
        miss = ~(student_ids == teacher_ids[..., :1]).any(-1)
        disagreement = student_ids[..., 0] != teacher_ids[..., 0]
        remaining = valid.long().flip(-1).cumsum(-1).flip(-1)
        scopes = (valid, valid & disagreement, valid & (remaining <= 16))
        values = torch.stack((torch.ones_like(response_mask), miss,
                              teacher_on_student.exp().sum(-1),
                              (union_positive.exp() * union_mask).sum(-1),
                              union_mask.sum(-1)), -1).double()
        for i, side in enumerate(self.sides):
            rows = torch.tensor([x == side for x in world_sides], device=valid.device)
            for j, scope in enumerate(scopes):
                self.sums[i, j] += (values * (scope & rows[:, None])[..., None]).sum((0, 1))

    def metrics(self):
        if dist.is_initialized():
            dist.all_reduce(self.sums)
        result = {}
        for i, side in enumerate(self.sides):
            for j, scope in enumerate(self.scopes):
                count, miss, old, union, size = self.sums[i, j].tolist()
                prefix = f'support_union/{side}/{scope}'
                result[f'{prefix}/tokens'] = count
                if count:
                    result.update({f'{prefix}/teacher_top1_miss_rate': miss / count,
                                   f'{prefix}/teacher_coverage_student100': old / count,
                                   f'{prefix}/teacher_coverage_union': union / count,
                                   f'{prefix}/support_size': size / count})
        return result

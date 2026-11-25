"""Independent paired-answer supervision; the original OPD/RGB target is unchanged."""
from pathlib import Path
import json
import random

from PIL import Image
import torch
import torch.nn.functional as F


def candidate_loss(scores, correct_index):
    """Exact two-candidate logistic loss, using complete answer log probabilities."""
    return F.softplus(scores[1 - correct_index] - scores[correct_index])


def candidate_batch(processor, item, view, device):
    tokenizer = processor.tokenizer
    content = []
    for index, part in enumerate(item['original_problem'].split('<image>')):
        if index:
            content.append({'type': 'image'})
        if part:
            content.append({'type': 'text', 'text': part})
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    with Image.open(item['original_path']) as image:
        original_size = image.size
    with Image.open(item[f'{view}_path']) as image:
        image = image.convert('RGB')
        if image.size != original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)
        inputs = processor(text=[text], images=[image], return_tensors='pt')
    prefix = inputs['input_ids'][0]
    mm = {k: inputs[k] for k in ('pixel_values', 'image_grid_thw')}
    image_id = int(processor.image_token_id)
    positions = processor.get_rope_index(input_ids=prefix[None],
        mm_token_type_ids=prefix.eq(image_id).long()[None],
        image_grid_thw=mm['image_grid_thw'], attention_mask=torch.ones_like(prefix)[None])
    if isinstance(positions, tuple):
        positions = positions[0]
    positions = positions.squeeze(1)
    if positions.shape[0] == 3:
        positions = torch.cat([torch.arange(len(prefix))[None], positions])
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    answers = [tokenizer.encode(letter, add_special_tokens=False) + [eos]
               for letter in (item['original_answer'], item['verified_edited_answer'])]
    width = max(map(len, answers))
    responses = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(responses)
    for i, tokens in enumerate(answers):
        responses[i, :len(tokens)] = torch.tensor(tokens)
        mask[i, :len(tokens)] = 1
    suffix_pos = positions[:, -1:] + 1 + torch.arange(width)[None]
    batch = {
        'input_ids': torch.cat([prefix[None].repeat(2, 1), responses], -1).to(device),
        'attention_mask': torch.cat([torch.ones(2, len(prefix), dtype=torch.long), mask], -1).to(device),
        'position_ids': torch.cat([positions, suffix_pos], -1)[None].repeat(2, 1, 1).to(device),
        'response_start_idx': torch.tensor([len(prefix)] * 2, device=device),
        'responses': responses.to(device), 'response_mask': mask.to(device),
        'multi_modal_inputs': [{k: v.to(device) for k, v in mm.items()} for _ in range(2)],
        'pad_token_id': tokenizer.pad_token_id,
    }
    return batch


def backward_pair_aux(actor):
    cfg = actor.config.self_distillation
    weight = float(cfg.get('pair_aux_weight', 0.0))
    if not weight:
        return {}
    if not hasattr(actor, '_pair_aux_items'):
        items = json.loads(Path(cfg.pair_aux_manifest).read_text())
        items = [x for x in items if x['split'] == 'train']
        random.Random(42).shuffle(items)
        actor._pair_aux_items = items
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    step = int(actor._current_global_steps)
    item = actor._pair_aux_items[((step - 1) * world + rank) % len(actor._pair_aux_items)]
    device = torch.cuda.current_device()
    losses, margins = [], []
    for view, correct in [('original', 0), ('edited', 1)]:
        batch = candidate_batch(actor.processor, item, view, device)
        outputs = actor._forward_micro_batch(batch, temperature=1.0)
        scores = (outputs['log_probs'].float() * batch['response_mask']).sum(-1)
        side_loss = candidate_loss(scores, correct)
        loss = 0.5 * weight * side_loss
        if actor.scaler is not None:
            actor.scaler.scale(loss).backward()
        else:
            loss.backward()
        losses.append(float(side_loss.detach()))
        margins.append(float((scores[correct] - scores[1 - correct]).detach()))
    metrics = {'pair_aux/loss': sum(losses) / 2,
        'pair_aux/weighted_loss': weight * sum(losses) / 2,
        'pair_aux/original_margin': margins[0], 'pair_aux/edited_margin': margins[1],
        'pair_aux/pair_correct': float(min(margins) > 0), 'pair_aux/weight': weight,
        'pair_aux/global_pairs_per_step': world}
    if rank == 0:
        print('[pair-aux-backward] ' + json.dumps({'step': step, 'sample': item['id'], **metrics}), flush=True)
    return metrics

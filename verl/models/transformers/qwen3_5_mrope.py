"""Keep Qwen3.5 MRoPE coordinates out of FlashAttention packing detection."""
from functools import wraps


def without_mrope_packing_positions(forward):
    @wraps(forward)
    def wrapped(self, *args, **kwargs):
        positions = kwargs.get("position_ids")
        if positions is not None and positions.ndim == 3:
            # MRoPE is already represented by position_embeddings in attention.
            # Flattening [3, batch, tokens] as sequence boundaries would address
            # three times as many tokens as the actual Q/K/V tensors contain.
            kwargs.pop("position_ids")
        return forward(self, *args, **kwargs)
    return wrapped


def patch_qwen3_5_mrope_positions(model_type="qwen3_5"):
    if model_type == "qwen3_5_moe":
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeAttention as attention_class
    else:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention as attention_class

    if getattr(attention_class.forward, "_verl_mrope_positions_fixed", False):
        return
    forward = without_mrope_packing_positions(attention_class.forward)
    forward._verl_mrope_positions_fixed = True
    attention_class.forward = forward
    print("[qwen3.5] MRoPE coordinates excluded from FlashAttention packing detection", flush=True)

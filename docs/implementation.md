# Implementation map

The method code is retained from the experiment runtime. Public changes are
confined to local entrypoints, data import, configuration and removal of
internal storage integration.

| Component | File |
|---|---|
| Two worlds, matching Teacher inputs, independent 8+8 rollout allocation | `verl/trainer/ppo/counterfactual_world.py` |
| Training orchestration, Teacher prompt and image preparation | `verl/trainer/ppo/ray_trainer.py` |
| Actor / frozen Teacher forward and JS update | `verl/workers/actor/dp_actor.py` |
| Teacher attention centroid and full-image mapping | `verl/trainer/ppo/teacher_center_target.py` |
| Student attention capture and Gaussian reference | `verl/trainer/ppo/evidence_caption.py` |
| Native attention collection | `verl/trainer/ppo/grounded_opd.py` |
| Positive-Teacher support union | `verl/trainer/ppo/teacher_union.py` |
| Attention-conditioned target distribution | `verl/trainer/ppo/attention_target.py` |
| Tokenized image dataset | `verl/utils/dataset/rl_dataset.py` |
| FSDP workers and Teacher initialization | `verl/workers/fsdp_workers.py` |

The selected recipe uses the historical `teacher_model_source=legacy` branch,
`teacher_regularization=ema`, and **`teacher_update_rate=0.0`**. This initializes
the matching Teacher and freezes it. The name `ema` alone does not imply that
the Teacher changes in the reported runs.

Layer20 is one-based; the captured full-attention module index is 19 for these
Qwen3.5 dense backbones. `evidence_attention_max_tokens=0` includes all valid
response positions. Teacher centers and token gates are detached. The Student
distribution receives gradients through the JS objective; the default recipe
does not backpropagate an attention auxiliary loss.

The historical reduction is valid-token averaging within each microbatch,
scaled by microbatch/rank sequence count, followed by the FSDP rank mean. It is
not replaced here by a globally token-weighted denominator. The original EOS,
padding masks and tail construction remain in the bundled implementation.

Data and FSDP initialization seeds are 42. The historical vLLM async-engine
default is seed0, with a separate rollout RNG initialized by rank. These seeds
do not guarantee bitwise reproducibility across GPU backends or kernels.

Actor and frozen Teacher use FSDP1 FULL_SHARD; rollout uses vLLM replicas with
TP1. Actor parameter / optimizer / activation offload and Teacher parameter
offload are off in the selected dense-model recipes. vLLM sleeping and cache
release between rollout and optimization are separate from FSDP CPU offload.

Processor area limits for the three selected models were 65,536 to 16,777,216
pixels, patch size16, merge size2. The 8192-token prompt budget still applies
after image and text processing. Preserve the downloaded model's processor
configuration and chat template.

## Released 27B checkpoint

The public 27B training recipe inherits the same learning rate as 4B and 9B:
10 warmup steps, then a constant 2e-6. The released 27B step80 weights came
from an earlier run: update 61 used 5e-7, followed by a linear decay to 1e-7
at update 130. To replay that schedule, append these Hydra overrides to
`bash scripts/train.sh 27b`:

```text
actor_rollout_ref.actor.optim.lr_drop_after_step=60
actor_rollout_ref.actor.optim.lr_drop_ratio=0.25
actor_rollout_ref.actor.optim.lr_drop_decay_end_step=130
```

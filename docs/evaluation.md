# Evaluation inputs and protocol

`scripts/evaluate.py` expects prepared JSON arrays with these filenames:

| Benchmark | File | Questions |
|---|---|---:|
| VStar | `vstar.json` | 191 |
| ZoomBench | `zoombench.json` | 845 |
| HRBench-4K | `hr_bench_4k.json` | 800 |
| HRBench-8K | `hr_bench_8k.json` | 800 |
| MME-RealWorld EN | `MME_RealWorld.json` | 23609 |
| MME-RealWorld CN | `MME_RealWorld_CN.json` | 5917 |
| MMVP, optional | `mmvp.json` | 300 |

Each record carries `images` (one local path), `query` (the actual question and
options), and `response` (ground truth). Keep category fields for breakdowns.
MMVP additionally needs its original integer `index` from1 to300. Paths may be
absolute, or relative to the prepared JSON's directory.

The retained `evaluation/vision_opd_official/prepare_data.py` contains the
upstream dataset-specific conversion routines. For example:

```bash
python evaluation/vision_opd_official/prepare_data.py \
  --benchmark vstar --data_dir /tmp/benchmarks
```

It uses a locally downloaded source snapshot when present, otherwise the
configured public dataset repository. Review the selected release/split and
the expected question count before a formal comparison. The training data
importer and the benchmark preparer serve different datasets.

The wrapper assigns stable row IDs, resolves local image paths, and saves the
exact input records under the run's `inputs/` directory. Existing run inputs
must remain unchanged. No source images or model weights are uploaded.

Answer extraction, the math/MCQ rules and the fallback Judge prompt are in
`evaluation/vision_opd_official/judge_qwenlm.py`. VStar's image transport uses
the historical PNG conversion and 20MiB limit from `infer.py`. The other
datasets preserve the prepared source images. Changing these preprocessing or
prompt rules changes the protocol.

The local GPT-OSS-120B Judge receives the question, reference and extracted
answer for unresolved cases. Its output must be `Yes` or `No`; an empty answer,
failed generation or unresolved Judge label is not silently dropped from the
denominator. Valid outputs are reused on the next invocation. Length-limited
but nonempty responses remain scored under the original decoding budget.

Per-benchmark accuracy uses the full question denominator. Avg6 gives each of
the six benchmarks equal weight. MMVP uses 150 pairs: indices(1,2), (3,4), ...,
(299,300), requiring both answers correct. Avg7 adds that pair accuracy as the
seventh equally weighted score. A partial run has no complete aggregate.

The two-phase commands in the README permit sequential Student and Judge
loading on the same GPUs. `--phase all` instead expects both endpoints to be
available simultaneously. Defaults point to localhost; no remote paid API is
configured.

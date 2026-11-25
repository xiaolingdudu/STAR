#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || $# == 0 ]]; then
  echo 'Usage: bash scripts/train.sh {4b|9b|27b} [Hydra overrides or --cfg job --resolve]'
  echo 'Required: STAR_MODEL, STAR_TRAIN_FILE, STAR_OUTPUT. Optional: RAY_ADDRESS.'
  exit 0
fi
star_scale="$1"
shift
case "$star_scale" in 4b|9b|27b) ;; *) echo 'Choose 4b, 9b or 27b.' >&2; exit 2;; esac
: "${STAR_MODEL:?Set the downloaded Base model directory}"
: "${STAR_TRAIN_FILE:?Set the prepared worlds.parquet path}"
: "${STAR_OUTPUT:?Set the local checkpoint directory}"
export STAR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${STAR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export VERL_REUSE_SINGLE_TURN_INPUTS=1
export TOKENIZERS_PARALLELISM=false
cd "$STAR_ROOT"
star_ray_args=()
if [[ -n "${RAY_ADDRESS:-}" ]]; then
  star_ray_args+=("+ray_kwargs.ray_init.address=${RAY_ADDRESS}")
fi
exec "${PYTHON_BIN:-python}" -m verl.trainer.main_ppo \
  --config-path "$STAR_ROOT/configs" --config-name "qwen35_${star_scale}" \
  "hydra.searchpath=[file://${STAR_ROOT}/verl/trainer/config]" \
  "${star_ray_args[@]+"${star_ray_args[@]}"}" "$@"

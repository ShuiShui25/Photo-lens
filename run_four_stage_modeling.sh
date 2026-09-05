#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="$ROOT/lens-modeling"
PY="${AUTOLENS_PYTHON:-python}"
JUPYTER="${JUPYTER:-jupyter}"
DEFAULT_SYSTEMS=(2 8 12 3110 3336 3441 13369 12971 13246)
if (( $# )); then SYSTEMS=("$@"); else SYSTEMS=("${DEFAULT_SYSTEMS[@]}"); fi

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp/matplotlib-photo-lens03

cd "$MODEL"
"$PY" autoarray_jax_crop_patch/apply_autoarray_crop_fix.py
"$PY" prepare_system_configs.py "${SYSTEMS[@]}"
"$PY" -c 'import jax; d=jax.devices(); print("JAX devices:",d); assert any(x.platform=="gpu" for x in d)'

mkdir -p "$MODEL/logs"
for system in "${SYSTEMS[@]}"; do
  config="$MODEL/system_${system}/config.json"
  echo "[$system] stages 1-3"
  "$PY" "$MODEL/run_three_stage.py" --config "$config" --jax-platform gpu 2>&1 | tee "$MODEL/logs/system_${system}_stages1-3.log"
  if [[ "$system" == 13369 ]]; then
    echo "[$system] finalized at stage 3 (stage 4 is intentionally disabled)"
    continue
  fi
  echo "[$system] stage 4 C+BrightnessZeroth"
  "$PY" "$MODEL/run_stage4.py" "$config" 2>&1 | tee "$MODEL/logs/system_${system}_stage4.log"
  "$PY" "$MODEL/create_result_notebook.py" "$system"
  "$JUPYTER" nbconvert --to notebook --execute \
    "$MODEL/${system}_four_stage_modeling.ipynb" \
    --output "$MODEL/${system}_four_stage_modeling.executed.ipynb" \
    --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.cwd="$MODEL"
  "$PY" -c "import json, pathlib; p=pathlib.Path('$MODEL/system_${system}/output/stage4_run_summary.json'); d=json.loads(p.read_text()); chi=float(d['reduced_chi_squared']); assert d['source']['regularization']=='ConstantPlusBrightnessZeroth'; assert d['jax_platform']=='gpu'; assert 0.05 < chi < 50.0, f'unphysical reduced chi-squared: {chi}'; print('[$system] validated', d['log_evidence'], 'chi2nu=', chi)"
done

# Build the complete nine-system report when the default sample was requested.
if (( ${#SYSTEMS[@]} == ${#DEFAULT_SYSTEMS[@]} )); then
  "$PY" "$MODEL/build_four_stage_summary.py"
fi

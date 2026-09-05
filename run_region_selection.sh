#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PHOTO_LENS_PYTHON:-python}"
AUTOLENS_PY="${AUTOLENS_PYTHON:-python}"
DEFAULT_SYSTEMS=(2 8 12 3110 3336 3441 13369 12971 13246)
if (( $# )); then SYSTEMS=("$@"); else SYSTEMS=("${DEFAULT_SYSTEMS[@]}"); fi

for system in "${SYSTEMS[@]}"; do
  config="$ROOT/lens-modeling/system_${system}/config.json"
  # System 13369 is deliberately finalized at stage 3: its stage-4 pixel-source
  # fit has a visibly worse residual map and is archived in PHOTO-LENS03-test.
  if [[ "$system" == 13369 ]]; then
    products="$ROOT/lens-modeling/system_${system}/output/wcs_products"
    product_label="stage-3"
  else
    products="$ROOT/lens-modeling/system_${system}/output/wcs_products_stage4"
    product_label="stage-4"
  fi
  positions="$ROOT/data/bb/${system}/positions_gui.txt"
  survey="$($PY -c "import json; print(json.load(open('$config'))['bb_survey'])")"
  if [[ "$survey" == HSC ]]; then lens_scale=1.0; else lens_scale=0.85; fi
  echo "[$system] $survey $product_label products; lens_scale=$lens_scale"
  # Rebuild the cached maximum-likelihood FitImaging without rerunning the
  # search. Export its native foreground plane and lens-light-subtracted arrays
  # directly, plus exact per-galaxy satellite images for exclusion masks.
  "$AUTOLENS_PY" "$ROOT/region-detection/export_satellite_products.py" \
    --config "$config" --products-dir "$products" \
    --stage "${product_label#stage-}"
  "$PY" "$ROOT/region-detection/region_selection_v2.py" \
    --system "$system" --products-dir "$products" \
    --positions-file "$positions" --background-centre positions-gui \
    --lens-scale "$lens_scale" \
    --output-dir "$ROOT/region-detection/outputs/system_${system}"
done

"$PY" "$ROOT/region-detection/build_region_report.py" \
  --system "${SYSTEMS[@]}" --output "$ROOT/region-detection/region_report.html"

# The stable masks above are intermediate detections.  The production masks
# assign pixels in the source/lens overlap with the point-reflection source
# fraction and preserve the satellite/secondary-source exclusions.
"$PY" "$ROOT/region-detection/apply_point_fraction_selection.py" \
  --system "${SYSTEMS[@]}"

#!/usr/bin/env bash
# Put the contrast channel through the same synthetic-recovery screen as anom and min,
# for every model. classify.py has always had --channels and the third channel was never
# run, on the assumption that min covered it; measuring what min resolves to showed it
# does not - along the recovered paths the anomaly is the cheaper branch in 94 per cent of
# cells, so min is anom with a three per cent discount and no geometry.
#
# RevealLO is already done (out/detection_RevealLO_contrast.csv). This does the other five,
# so that every model's configuration is chosen from the same screen and the cross-model
# comparison stays like-for-like. Each writes its own table and touches nothing existing.
#
#   bash sweep_contrast.sh
#
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction"
TOMO="$ROOT/REVEAL_mantle_tomography"
mkdir -p logs
MODELS=(
  "RevealLO_30km|$TOMO/RevealLO.nc|voigt|3|2880"
  "REVEAL|$ROOT/REVEAL_vs_full.nc|voigt|1|2880"
  "GLADM35|$TOMO/GLADM35.nc|voigt|2|2890"
  "SPiRaL|$TOMO/SPiRaL.nc|voigt|1|2891"
  "SEMUCB-WM1|$TOMO/SEMUCB-WM1.nc|vs|1|2891"
)
rc=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  if [ -f "out/detection_${tag}_contrast.csv" ]; then
    echo "== $tag already swept, skipping"; continue
  fi
  if [ ! -f "$file" ]; then
    echo "== $tag: $file not found, skipping"; rc=1; continue
  fi
  echo "== contrast screen: $tag"
  python3 classify.py --file "$file" --tag "$tag" --var "$var" --every "$every" \
      --depth-max "$zmax" --channels contrast --suffix _contrast \
      --lateral 0.60 --ncell 2 --jobs 8 2>&1 | tee "logs/detect_contrast_$tag.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
done
echo
echo "done. then: python3 channel_decision.py"
exit $rc

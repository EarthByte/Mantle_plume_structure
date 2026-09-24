#!/bin/zsh
# Root depth across all five models, then the cross-family classification.
# Model discovery matches run_morphology.sh so an external disk is found the
# same way, and a missing file is skipped rather than silently using a stale copy.
set -u
ROOT="${ROOT:-$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction}"

find_tomo () {
  local c
  for c in "${TOMO:-}" \
           "$ROOT/REVEAL_mantle_tomography" \
           "/Volumes/ZeitwissenDataSSD/archive/Muller_mantle_tomography_subduction/REVEAL_mantle_tomography" \
           "/Volumes/ZeitwissenDataSSD/REVEAL_mantle_tomography"; do
    [ -n "$c" ] && [ -f "$c/RevealLO.nc" ] && { echo "$c"; return; }
  done
  echo "$ROOT/REVEAL_mantle_tomography"
}
TOMO="$(find_tomo)"
echo "tomographic models: $TOMO"
mkdir -p out

MODELS=(
  "RevealLO|$TOMO/RevealLO.nc|voigt|2|2880"
  "REVEAL|$ROOT/REVEAL_vs_full.nc|voigt|2|2880"
  "GLADM35|$TOMO/GLADM35.nc|voigt|2|2890"
  "SPiRaL|$TOMO/SPiRaL.nc|voigt|1|2891"
  "SEMUCB-WM1|$TOMO/SEMUCB-WM1.nc|vs|1|2891"
)

for spec in "${MODELS[@]}"; do
  tag="${spec%%|*}"; rest="${spec#*|}"
  file="${rest%%|*}"; rest="${rest#*|}"
  var="${rest%%|*}"; rest="${rest#*|}"
  every="${rest%%|*}"; zmax="${rest##*|}"
  if [ ! -f "$file" ]; then
    echo "== $tag: $file not found, skipping"
    continue
  fi
  echo "== $tag"
  python3 root_depth.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" 2>&1 | tee "out/root_depth_${tag}.log"
done

echo
echo "== cross-family classification"
python3 root_classes.py 2>&1 | tee out/root_classes.log

# The calibration decides what of the above is reportable, so it runs in the same
# pass rather than being something to remember afterwards. Injected conduits
# terminate at known depths and the same walk is run at the axis, through
# root_core.py, so this calibrates the code actually used.
echo
echo "== injection calibration"
for spec in "${MODELS[@]}"; do
  tag="${spec%%|*}"; rest="${spec#*|}"
  file="${rest%%|*}"; rest="${rest#*|}"
  var="${rest%%|*}"; rest="${rest#*|}"
  every="${rest%%|*}"; zmax="${rest##*|}"
  if [ ! -f "$file" ]; then
    echo "== $tag: not found, skipping"
    continue
  fi
  echo "== $tag calibration"
  python3 root_depth_test.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --sites 25 \
      2>&1 | tee "out/root_depth_test_${tag}.log"
done

echo
echo "Read out/root_depth_test_RevealLO.log before believing any root depth."
echo "If the bias at 660 km is large and the interquartile range is comparable to"
echo "the class widths, only the binary reaches-the-base statistic is reportable."

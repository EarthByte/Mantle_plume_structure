#!/bin/zsh
# Three phases, in this order for a reason: the configuration is chosen and frozen
# on injected conduits with known truth BEFORE the statistic is pointed at real
# hotspots, and phase 2 refuses to run if phase 1 has not frozen a configuration.
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

echo
echo "== phase 1: calibrate and freeze a configuration FOR EACH MODEL"
echo "   The search radius is a physical distance and the same distance is a"
echo "   different number of grid cells in each model, so one configuration does"
echo "   not transfer. A model whose best configuration cannot detect the weakest"
echo "   required conduit is refused a configuration rather than given a bad one."
for spec in "${MODELS[@]}"; do
  tag="${spec%%|*}"; rest="${spec#*|}"
  file="${rest%%|*}"; rest="${rest#*|}"
  var="${rest%%|*}"; rest="${rest#*|}"
  every="${rest%%|*}"; zmax="${rest##*|}"
  if [ ! -f "$file" ]; then echo "== $tag: not found, skipping"; continue; fi
  if [ -f "out/axis_config_frozen_${tag}.json" ]; then
    echo "== $tag: already frozen, skipped"
    continue
  fi
  echo "== $tag calibration"
  python3 axis_coherence_test.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" \
      --radius 300,400,600,800,1000,1200 --conduit-radius 200,400,600 \
      --amps 0.3,0.6,1.2 --axial-tols 150,300,450 --detect-amp 0.6 --freeze \
      2>&1 | tee "out/axis_calibration_${tag}.log"
done

echo
echo "== phase 2: measure axis coherence at hotspots against matched and rotated nulls"
for spec in "${MODELS[@]}"; do
  tag="${spec%%|*}"; rest="${spec#*|}"
  file="${rest%%|*}"; rest="${rest#*|}"
  var="${rest%%|*}"
  if [ ! -f "$file" ]; then echo "== $tag: not found, skipping"; continue; fi
  if [ ! -f "out/axis_config_frozen_${tag}.json" ]; then
    echo "== $tag: no frozen configuration, so this model could not detect a conduit"
    echo "   at any configuration tried. Skipped, and that is a result about the model."
    continue
  fi
  echo "== $tag"
  python3 axis_coherence.py --file "$file" --tag "$tag" --var "$var" \
      2>&1 | tee "out/axis_coherence_${tag}.log"
done

echo
echo "== phase 3: continuity re-run with nearest-neighbour matching"
for spec in "${MODELS[@]}"; do
  tag="${spec%%|*}"; rest="${spec#*|}"
  file="${rest%%|*}"; rest="${rest#*|}"
  var="${rest%%|*}"; rest="${rest#*|}"
  every="${rest%%|*}"; zmax="${rest##*|}"
  if [ ! -f "$file" ]; then echo "== $tag: not found, skipping"; continue; fi
  echo "== $tag"
  python3 continuity_annulus_null.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" \
      2>&1 | tee "out/continuity_annulus_${tag}.log"
done

echo
echo "Phase 1 decides, per model, whether the statistic can see a conduit at all."
echo "A model refused a configuration is not a failure of the run: it is the finding"
echo "that the model cannot support the question. Phase 2 speaks only for models that"
echo "were frozen. Phase 3 re-tests continuity with nearest-neighbour matching, which"
echo "on the first pass left the result standing in the REVEAL family alone."

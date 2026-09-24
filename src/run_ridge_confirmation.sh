#!/bin/zsh
# The ridge association was read off the RevealLO output, so RevealLO cannot test
# it. This computes corridors for REVEAL and GLAD-M35 at each model's own
# calibrated configuration, then applies the rule fixed in RIDGE_PREREGISTRATION.md:
# distance to the nearest spreading centre of any age, per component, one-sided,
# P < 0.05. One measure, decided before these corridors existed.
#
# About twenty-five minutes for the two models together.
set -u
ROOT="${ROOT:-$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction}"
find_tomo () {
  local c
  for c in "${TOMO:-}" "$ROOT/REVEAL_mantle_tomography" \
           "/Volumes/ZeitwissenDataSSD/archive/Muller_mantle_tomography_subduction/REVEAL_mantle_tomography" \
           "/Volumes/ZeitwissenDataSSD/REVEAL_mantle_tomography"; do
    [ -n "$c" ] && [ -f "$c/RevealLO.nc" ] && { echo "$c"; return; }
  done
  echo "$ROOT/REVEAL_mantle_tomography"
}
TOMO="$(find_tomo)"
echo "tomographic models: $TOMO"
mkdir -p out data

if [ ! -f data/ridges_presentday_zahirovic2022.csv ]; then
  echo "extracting present-day ridges from Zahirovic 2022"
  python3 ridges_presentday.py --model "$ROOT/Paper_hydration/Zahirovic2022_plate_model"
fi

run_model () {
  local tag="$1" file="$2" every="$3" dmax="$4"
  echo
  echo "=============== $tag ==============="
  python3 corridor_all.py --file "$file" --tag "$tag" --every "$every" \
      --depth-max "$dmax" 2>&1 | tee "out/corridor_all_${tag}.log"
  python3 corridor_network.py --tag "$tag" 2>&1 | tee "out/corridor_network_${tag}.log"
  python3 corridor_ridge_association.py --tag "$tag" 2>&1 \
      | tee "out/corridor_ridge_${tag}.log"
}

run_model REVEAL  "$ROOT/REVEAL_vs_full.nc" 1 2880
run_model GLADM35 "$TOMO/GLADM35.nc"        2 2890

echo
echo "Read the self-check line for each model first. Then the per-component column"
echo "for distance to a ridge of any age: that one number, in each model, is the"
echo "whole test. Both under 0.05 is a result, one is a hint, neither withdraws it."

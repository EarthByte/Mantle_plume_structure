#!/bin/zsh
# Corridors for all 49 hotspots at the configuration the paths were traced under.
# About ten minutes. Produces the per-plume error bars and the isolated-versus-
# networked measurement in one pass.
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
mkdir -p out
python3 corridor_all.py --file "$TOMO/RevealLO.nc" --tag RevealLO \
    2>&1 | tee out/corridor_all.log
echo
echo "The width column is the error bar on the traced path and the isolation"
echo "measure at once. Check the self-check line first: a corridor that fails it"
echo "is not a corridor, and its width means nothing."

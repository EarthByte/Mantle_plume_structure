#!/bin/zsh
# Does corridor width measure the conduit, or the site? Conduits of known width
# are injected into ambient locations and the corridor is measured exactly as it
# is for a real hotspot. The decision rule is fixed in the script's docstring and
# is not to be revised after seeing the output.
#
# Run the pilot first. It prints how long the full run will take, so the cost is
# known before it is committed.
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
MODE="${1:-pilot}"
echo "tomographic models: $TOMO"
mkdir -p out
if [ "$MODE" = "full" ]; then
  python3 corridor_width_calibrate.py --file "$TOMO/RevealLO.nc" --tag RevealLO \
      --sites 12 --lateral 0.60 --ncell 2 2>&1 | tee out/corridor_width_calibration.log
else
  python3 corridor_width_calibrate.py --file "$TOMO/RevealLO.nc" --tag RevealLO \
      --pilot --lateral 0.60 --ncell 2 2>&1 | tee out/corridor_width_calibration_pilot.log
fi
echo
echo "The verification line comes first: the in-place injector must agree with"
echo "inject_tilted, or the calibration is not measuring the conduit it claims."

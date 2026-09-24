#!/bin/zsh
# Phase 1 alone for now. It decides whether the depth questions can be attempted:
# the stopping rule in PLUME_STRUCTURE_PLAN.md says an interquartile error worse
# than about 150 km means 660 and 1000 km cannot be separated and the programme
# stops at the catalogue. Nothing downstream is written until this has run.
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

echo
echo "== branch-depth calibration on RevealLO"
echo "   Y conduits injected at hotspot locations, at the widths real plumes have."
echo "   Limb separation is swept as a multiple of limb width, because two limbs"
echo "   closer than about twice their width are one blob and cannot be resolved"
echo "   in principle - that ratio may bind before depth precision does."
python3 plume_trace_test.py --file "$TOMO/RevealLO.nc" --tag RevealLO \
    --every 2 --depth-max 2880 \
    --radii 300,500,700 --amps 0.4,0.8,1.5 \
    --branches 660,1000,1500 --sep-ratios 1.5,2.5,4.0 \
    --sites 30 --suffix _paired 2>&1 | tee out/plume_trace_test.log

# The ambient field produces a spurious branch at about a fifth of sites, and those
# cluster near 1000 km - exactly where Rudolph's viscosity boundary is predicted. A
# real measurement finding plume branches there would sit on a background of the
# same shape, so the background is surveyed at sites away from hotspots and its
# depth distribution recorded. This is not a test of whether plumes differ from
# ambient mantle; it is the instrument's own false-positive spectrum.
echo
echo "== ambient branch background, away from hotspots"
python3 plume_ambient.py --file "$TOMO/RevealLO.nc" --tag RevealLO \
    --every 2 --depth-max 2880 --sites 200 2>&1 | tee out/plume_ambient.log

echo
echo "Read the table before anything else. Two numbers decide the programme:"
echo "  the interquartile branch-depth error, against the 150 km stopping rule,"
echo "  and the separation-to-width ratio at which two limbs become resolvable."

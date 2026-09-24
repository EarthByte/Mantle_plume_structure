#!/bin/zsh
# Everything the continuity result needs before it can be submitted.
#
# Each step writes its own log and the script does not stop on a failure, so one
# bad step does not cost the night. A summary is printed at the end.
#
# About four and a half hours.
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
echo "started $(date)"
mkdir -p out logs

DUTY="1.0,0.75,0.5,0.25,0.1"
SITES=12

step () {
  local name="$1"; shift
  echo
  echo "=============== $name  $(date +%H:%M) ==============="
  if "$@" > "logs/$name.log" 2>&1; then
    echo "OK  $name"
  else
    echo "FAILED  $name  (see logs/$name.log)"
  fi
  tail -14 "logs/$name.log"
}

# Corridors for the two models that do not have them yet, so each has its own
# observed hotspot width to invert against. RevealLO, REVEAL and GLAD-M35 already do.
step corridors_SEMUCB python3 corridor_all.py --file "$TOMO/SEMUCB-WM1.nc" \
     --tag SEMUCB-WM1 --var vs --every 1 --depth-max 2891
step corridors_RevealLO30 python3 corridor_all.py --file "$TOMO/RevealLO.nc" \
     --tag RevealLO_30km --every 3 --depth-max 2880

# The headline measurement at twelve sites instead of three, in each model. The
# occupancy is read off each model's OWN injection curve, so resolution divides out.
step duty_RevealLO python3 corridor_width_calibrate.py --file "$TOMO/RevealLO.nc" \
     --tag RevealLO --sites $SITES --duty "$DUTY"
step duty_GLADM35 python3 corridor_width_calibrate.py --file "$TOMO/GLADM35.nc" \
     --tag GLADM35 --every 2 --depth-max 2890 --sites $SITES --duty "$DUTY"
step duty_SEMUCB python3 corridor_width_calibrate.py --file "$TOMO/SEMUCB-WM1.nc" \
     --tag SEMUCB-WM1 --var vs --every 1 --depth-max 2891 --sites $SITES --duty "$DUTY"

# The resolution control: the same model and the same Earth at coarser sampling.
# A cross-model difference could always be the Earth or the method; this cannot.
step duty_RevealLO30 python3 corridor_width_calibrate.py --file "$TOMO/RevealLO.nc" \
     --tag RevealLO_30km --every 3 --depth-max 2880 --sites $SITES --duty "$DUTY"

# Does the arbitrary 400 km cycle length set the answer?
step segments_RevealLO python3 corridor_width_calibrate.py --file "$TOMO/RevealLO.nc" \
     --tag RevealLO --sites 6 --segments 200,400,800,1600 --sweep-duty 0.25

echo
echo "=============== finished $(date) ==============="
echo "Read in this order: the injector verification line in each log, then the"
echo "occupancy each model infers, then the RevealLO against RevealLO_30km pair."
echo "If coarser sampling infers HIGHER occupancy, that is smearing filling gaps"
echo "and the best-resolved model gives the better estimate. If it infers the same,"
echo "the measurement is resolution-independent and the cross-model numbers stand"
echo "on their own."

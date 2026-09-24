#!/usr/bin/env bash
# The occupancy sweeps behind section 3.7, for every model-sampling combination the
# section compares. run_width_calibration.sh does the RevealLO calibration alone; this
# does the four duty sweeps, which are what the occupancy bound rests on.
#
# They were run once each, on three different days, and each carries the configuration
# it was made under in its own cfg_ columns:
#
#   GLADM35        s=0.5  channel=min    radius=2.0  h_max=900
#   RevealLO_30km  s=0.4  channel=anom   radius=2.0  h_max=300
#   SEMUCB-WM1     s=0.5  channel=anom   radius=2.0  h_max=300
#   RevealLO       no cfg_ columns at all; it predates the stamping
#
# Not one of them is the frozen configuration, and two are not even the same channel as
# each other. Section 3.7 set all four beside corridors traced under contrast at s = 0.3.
#
# Each sweep is 72 corridors, about an hour. A sweep whose stored configuration already
# matches the frozen one is skipped, so an interrupted run resumes where it stopped and
# a finished one costs nothing to re-run.
#
#   nohup bash run_calibration_all.sh > logs/calibration_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
eval "$(grep -E '^(ROOT|TOMO)=' run_all.sh)"
eval "$(sed -n '/^MODELS=(/,/^)/p' run_all.sh)"
DUTY="1,0.75,0.5,0.25,0.1"
SUFFIX="_dutysweep_1-0.75-0.5-0.25-0.1"
rc=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  case "$tag" in RevealLO|RevealLO_30km|GLADM35|SEMUCB-WM1) ;; *) continue ;; esac
  out="out/corridor_width_calibration_${tag}${SUFFIX}.csv"
  if python3 - "$out" "$tag" <<'PY'
import sys, os, pandas as pd, path_config
out, tag = sys.argv[1], sys.argv[2]
if not os.path.exists(out):
    sys.exit(1)
d = pd.read_csv(out)
need = ('cfg_s', 'cfg_channel', 'cfg_z_target', 'cfg_radius', 'cfg_h_max',
        'lateral', 'ncell')
if any(k not in d.columns for k in need):
    sys.exit(1)
c = path_config.load(tag, 'out')
ok = (abs(float(d.lateral.iloc[0]) - 0.60) < 1e-9
      and int(d.ncell.iloc[0]) == 2
      and abs(float(d.cfg_s.iloc[0]) - float(c.s)) < 1e-9
      and str(d.cfg_channel.iloc[0]) == str(c.channel)
      and abs(float(d.cfg_z_target.iloc[0]) - float(c.z_target)) < 1e-9
      and abs(float(d.cfg_radius.iloc[0]) - float(c.radius)) < 1e-9)
sys.exit(0 if ok else 1)
PY
  then
    echo "== $tag: the stored sweep already matches the frozen configuration, skipping"
    continue
  fi
  echo
  echo "=============== $tag  $(date '+%Y-%m-%d %H:%M') ==============="
  python3 corridor_width_calibrate.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --sites 12 --duty "$DUTY" --lateral 0.60 --ncell 2 \
      2>&1 | tee "logs/calibration_${tag}.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!! $tag FAILED"; rc=1; }
done
echo
echo "duty sweeps done; rc=$rc"
exit $rc

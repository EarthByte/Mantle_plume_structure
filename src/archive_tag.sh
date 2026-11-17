#!/usr/bin/env bash
#
# Before a retrace caused by a change to a MODEL (a corrected file, or a corrected
# reading of it), move aside the outputs of that model's tag that would otherwise
# let a step resume or skip.
#
# WHY. Several steps decide whether to work by looking at the configuration recorded in
# their existing output: classify (via retrace.sh's guard), competing_routes.py,
# frontier.py, corridor_width_calibrate.py (via run_calibration_all.sh) and
# detection_veto.py. That is right when the move set or the frozen configuration
# changes, and wrong when the model changes underneath them, because nothing in a
# recorded configuration says which model or grid it was computed on. After RevealLO.nc
# was replaced on 28 September, the retrace skipped seven such files while every step
# reported success. Moving them aside makes those steps start from nothing.
#
# Only the resuming outputs move. Everything else a step rewrites unconditionally, and
# the audit's freshness report (audit_manuscript.py, freshness.py) lists any input it
# reads that is older than the newly traced paths - the check that this list is
# complete. Nothing is deleted; the frozen configuration (path_config_<tag>.json) stays.
#
#   bash archive_tag.sh <label> <tag> [<tag> ...]
#   bash archive_tag.sh meridian_fix_20261020 RevealLO RevealLO_30km
#
# then:  FROM=classify bash retrace.sh
set -euo pipefail
cd "$(dirname "$0")"
[ "$#" -ge 2 ] || { echo "usage: bash archive_tag.sh <label> <tag> [<tag> ...]" >&2; exit 2; }
label="$1"; shift
dest="out/archive_$label"
mkdir -p "$dest"
shopt -s nullglob
n=0
move () {
  local f
  for f in "$@"; do
    [ -f "$f" ] || continue
    if [ -e "$dest/$(basename "$f")" ]; then
      echo "already in $dest, left in place: $(basename "$f")" >&2; continue
    fi
    mv "$f" "$dest/"; n=$((n + 1)); echo "  moved $(basename "$f")"
  done
}
PAPER_MODEL=$(sed -n 's/^PAPER_MODEL=//p' retrace.sh)
for t in "$@"; do
  move out/classification_"$t".csv out/classification_"$t".csv.prov.json \
       out/competing_routes_"$t".csv out/competing_tracks_"$t".json \
       out/corridor_width_calibration_"$t"_dutysweep_*.csv
  # The continuity step regenerates these for the paper model only; for any other tag
  # they are left alone rather than moved to where nothing would replace them.
  if [ "$t" = "$PAPER_MODEL" ]; then
    move out/competing_routes_"$t"_null.csv out/frontier_"$t".csv \
         out/detection_veto_"$t".csv out/tilt_recovery_"$t"_levels.csv \
         out/tilt_recovery_"$t"_levels.csv.prov.json
  fi
done
# One row per model in a shared file: re-run for every model by the continuity step.
move out/route_anomaly_spread.csv
echo "moved $n files to $dest"

#!/usr/bin/env bash
#
# Retrace everything at the calibrated move set: lateral 0.60, two lateral cells
# per slanted move.
#
# slant_calibrate.py chose that setting on the criterion fixed before it ran: a
# hotspot path should find slower material than an ambient path traced at the
# same setting. Every number in the manuscript was computed with one lateral cell,
# where the steepest expressible tilt is about 64 degrees, so a conduit leaning
# harder than that could not be followed at any price - which is what the four
# named failures were.
#
# The move set enters at the cost field, so it is not only the traced paths that
# change. The classification is a hotspot-versus-null cost comparison built on
# that same field, and a setting that helps hotspots more than nulls - which is
# exactly what the calibration measured - moves it. The classification is
# therefore redone first, and everything else follows from it.
#
# 2026-09-15: this script is now also the retrace for a change to the FROZEN
# CONFIGURATION, not only to the move set. s was refrozen from 0.4 to 0.3 - see
# out/path_config_RevealLO.json, whose note records why, and STATE.md. When only the
# frozen configuration has changed, START AT paths:
#
#   FROM=paths bash retrace.sh
#
# The classification is an ensemble over every configuration that clears the detection
# floor, not over the frozen one, so a change to the frozen configuration does not move
# it and step_classify is right to skip. The move set does move it, which is what its
# idempotency guard checks. Everything from paths onward reads the frozen file and must
# be redone.
#
#   bash retrace.sh              run from the first unfinished step
#   bash retrace.sh --list       show the steps and what is already done
#   FROM=corridors bash retrace.sh   restart at a named step
#
# Each step writes a stamp when it succeeds and is skipped on a later run, so an
# interruption costs only the step it happened in. Nothing is deleted: the whole
# 1.0 run is copied to out/lateral_1.0/ before anything is overwritten.
set -uo pipefail
cd "$(dirname "$0")"

# The ambient control is the comparison every claim in the paper leans on, and its
# SIZE has decided three separate conclusions. Text S3 states 300 paths. This step
# ran at 30 and quietly replaced the 300-path file with it, which turned the
# directional test from R = 0.124 at P = 0.025 into R = 0.150 at P = 0.57 - the same
# control, with the power taken out. It is not a tuning knob.
export AMBIENT_SITES=300
export LATERAL=0.60
export NCELL=2
export JOBS="${JOBS:-8}"

ROOT="$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction"
TOMO="$ROOT/REVEAL_mantle_tomography"
STAMPS=out/retrace_stamps
mkdir -p "$STAMPS" logs out

# tag | file | var | every | depth-max, the same table run_all.sh carries
MODELS=(
  "RevealLO|$TOMO/RevealLO.nc|voigt|1|2880"
  "RevealLO_30km|$TOMO/RevealLO.nc|voigt|3|2880"
  "REVEAL|$ROOT/REVEAL_vs_full.nc|voigt|1|2880"
  "GLADM35|$TOMO/GLADM35.nc|voigt|2|2890"
  "SPiRaL|$TOMO/SPiRaL.nc|voigt|1|2891"
  "SEMUCB-WM1|$TOMO/SEMUCB-WM1.nc|vs|1|2891"
)
# The models that carry corridors. SPiRaL has never had them.
CORRIDOR_TAGS="RevealLO RevealLO_30km REVEAL GLADM35 SEMUCB-WM1"
PAPER_MODEL=RevealLO
# The models that carry the 300 ambient paths. The paper model has always had them;
# GLAD-M35 and SEMUCB-WM1 were added on 22 September 2026 so that the mid-mantle
# tilt excess of section 3.2 (hotspot paths against ambient paths at 1000-1500 km) is
# tested in two independent inversion families and not in one image. Each set is
# 300 traces at the production move set, about the cost of the paper model's own.
AMBIENT_TAGS="RevealLO GLADM35 SEMUCB-WM1"

# THE ORDER, AND THE ONLY PLACE IT IS WRITTEN DOWN. This list was maintained by hand
# beside the run_step calls at the bottom and the two drifted apart: nulls and
# calibration ran but were absent from the list, so FROM=nulls was rejected as "no such
# step" and a restart at corridors left their stamps in place and skipped them both,
# while provenance sat in the list and was never called at all. The bottom of the file
# now iterates this list, so the two cannot disagree again, and a name without a
# matching step_ function stops the run before any work is done.
STEPS="preflight classify paths ambient nulls columns corridors geometry calibration derived continuity provenance figures audit"

if [ "${1:-}" = "--list" ]; then
  for s in $STEPS; do
    if [ -f "$STAMPS/$s" ]; then echo "done     $s   ($(cat "$STAMPS/$s"))"
    else echo "to run   $s"; fi
  done
  exit 0
fi

# FROM=<step> restarts there: that step and every one after it lose their stamp,
# because a step that is re-run invalidates everything downstream of it.
FROM="${FROM:-}"
if [ -n "$FROM" ]; then
  case " $STEPS " in *" $FROM "*) ;; *) echo "no such step: $FROM" >&2; exit 2;; esac
  hit=0
  for s in $STEPS; do
    [ "$s" = "$FROM" ] && hit=1
    [ "$hit" = 1 ] && rm -f "$STAMPS/$s"
  done
  echo "restarting at $FROM"
fi

run_step () {                      # run_step <name> <command...>
  local name="$1"; shift
  if [ -f "$STAMPS/$name" ]; then
    echo "== $name: already done $(cat "$STAMPS/$name"), skipping"
    return 0
  fi
  echo
  echo "=============== $name  $(date '+%Y-%m-%d %H:%M') ==============="
  if "$@"; then
    date '+%Y-%m-%d %H:%M' > "$STAMPS/$name"
    echo "== $name OK"
  else
    echo "!! $name FAILED - stopping. Fix it, then re-run: bash retrace.sh" >&2
    exit 1
  fi
}

# ------------------------------------------------------------- 0. preflight
# Seconds. Checks that the corridor's forward and backward fields still price one
# physical move identically now that a move may cross more than one cell, that
# the old setting is reproduced bit for bit, and that both flags reach every cost
# field including inside the forked injection workers. A missing argument would
# otherwise surface hours into the classification.
step_preflight () {
  python3 test_moveset.py 2>&1 | tee logs/retrace_preflight.log
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------- 1. backup
step_backup () {
  local d=out/lateral_1.0
  mkdir -p "$d"
  local n=0 skipped=0
  for f in out/classification_*.csv out/detection_*.csv out/curves_*.csv \
           out/conduit_paths_all_*.json out/corridor_summary_*.csv \
           out/corridor_profiles_*.csv out/corridor_network_*.csv \
           out/corridor_overlap_*.csv out/corridor_ridge_association_*.csv \
           out/plume_*.csv out/crossmodel_lean.csv out/apm_crossmodel.csv \
           out/track_*.csv out/province_vote.npz sections_all.json sections.json; do
    [ -e "$f" ] || continue
    if [ -e "$d/$(basename "$f")" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    cp -p "$f" "$d/" && n=$((n + 1))
  done
  echo "archived $n files to $d ($skipped already there, left untouched)"
  # -n semantics by hand: an existing archive is never overwritten, so running
  # this again after the retrace cannot replace the 1.0 run with the new one.
  [ "$(ls -1 "$d" | wc -l)" -gt 40 ]
}

# ------------------------------------------------- 2. classification + paths
# run_all.sh reads LATERAL and NCELL and passes them to classify.py, paths.py,
# track_depth.py and null_snapshot.py, so the whole run is at one move set.
# FORCE=1 is required: without it every model with an existing classification
# is skipped and the retrace silently does nothing.
step_classify () {
  # Idempotent: if every model's classification already carries the target move
  # set, the expensive part is done and forcing it would throw away hours.
  local done_all=1 t
  for t in RevealLO RevealLO_30km REVEAL GLADM35 SPiRaL SEMUCB-WM1; do
    python3 - "$t" <<'PYCHK' || done_all=0
import pandas as pd, sys
sys.path.insert(0, '.')
import path_config
t = sys.argv[1]
# The guard has to see everything the classification depends on. It used to check the
# move set alone, so a change to the cost channel - or to the rule deciding which
# configurations the ensemble averages over - left it invisible and this step skipped,
# keeping a classification built under a configuration nobody was using.
try:
    d = pd.read_csv(f'out/classification_{t}.csv')
    want = path_config.ensemble_channel(t, 'out') or 'all'
    ok = ('lateral' in d and abs(d.lateral.iloc[0] - 0.60) < 1e-9
          and int(d.ncell.iloc[0]) == 2
          and 'ensemble_channel' in d
          and str(d.ensemble_channel.iloc[0]) == str(want))
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYCHK
  done
  if [ "$done_all" = 1 ]; then
    echo "all six classifications carry the current move set and ensemble channel; not re-running"
    return 0
  fi
  FORCE=1 bash run_all.sh 2>&1 | tee logs/retrace_run_all.log
  local rc="${PIPESTATUS[0]}"
  # run_all.sh runs its standalone figures LAST, so that a label collision - a figure
  # defect, nothing to do with the computation - cannot throw away the tracing. It still
  # sets a non-zero status, and treating that as a hard stop threw away the retrace
  # instead, which is the same mistake one level up. A figure failure is recorded and
  # surfaced at the end; only a computation failure stops the run.
  if [ "$rc" -ne 0 ] && grep -q 'done with figure failures' logs/retrace_run_all.log; then
    grep -A2 'these figures failed their own checks' logs/retrace_run_all.log \
        > out/figure_failures.txt
    echo
    echo "!! FIGURES FAILED THEIR CHECKS, recorded in out/figure_failures.txt:"
    sed 's/^/   /' out/figure_failures.txt
    echo "   The computation completed; continuing. These must be fixed before submission."
    return 0
  fi
  return "$rc"
}

# ------------------------------------------------------------------ 3. paths
# The all-hotspot path tracing, and the section planes derived from it. The paper
# model goes FIRST and alone carries --fix-azimuths: it writes sections_all.json,
# the nine hundred great circles every other model is then cut on, so that one
# hotspot is sectioned on the same plane in every model. Tracing the others first
# would leave them on the previous run's planes.
step_paths () {
  local m tag file var every zmax pass fix rc=0
  for pass in fix rest; do
    for m in "${MODELS[@]}"; do
      IFS='|' read -r tag file var every zmax <<< "$m"
      [ -f "$file" ] || continue
      [ -f "out/classification_$tag.csv" ] || continue
      if [ "$tag" = "$PAPER_MODEL" ]; then
        [ "$pass" = "fix" ] || continue
        fix="--fix-azimuths"
      else
        [ "$pass" = "rest" ] || continue
        fix=""
      fi
      echo "== all-hotspot paths $tag"
      python3 paths.py --file "$file" --tag "$tag" --var "$var" \
          --every "$every" --depth-max "$zmax" --all \
          --lateral "$LATERAL" --ncell "$NCELL" $fix \
          2>&1 | tee "logs/paths_$tag.log"
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    done
  done
  [ "$rc" -eq 0 ] || return 1
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    [ -f "out/conduit_paths_all_$tag.json" ] || continue
    if [ "$tag" = "$PAPER_MODEL" ]; then
      python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --all || rc=1
      python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --top 9 || rc=1
    else
      python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --top 9 || rc=1
    fi
  done
  return $rc
}

# ---------------------------------------------------------------- 4. ambient
# The ambient control, re-traced at the SAME move set as the hotspots. It was on
# lateral 1.0 / ncell 1 and nothing re-ran it, so a retrace that left it there
# would compare hotspot paths from one search against ambient paths from another -
# and this control is what section 3.2 rests on, the finding that displacement
# magnitude is not resolved while its direction is. Thirty sites, nothing injected.
# It now also records the offset over 660-1500 km, which is the window the
# Louisville sentence uses and the one the old file could not answer for.
step_ambient () {
  local m tag file var every zmax
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    case " $AMBIENT_TAGS " in *" $tag "*) ;; *) continue;; esac
    [ -f "$file" ] || { echo "!! $file missing"; return 1; }
    # --save-paths is not optional. track_support.py reads ambient_paths_<tag>.json to
    # set the ambient level every hotspot's track support is judged against, and so the
    # atlas grouping depends on it. Without this flag the ambient run recorded summary
    # columns only, that file stood at its 13 September version through two changes of
    # configuration, and the grouping compared current hotspots against superseded
    # ambient paths.
    # Move the previous ambient control aside under a name that says what it is.
    # It used to go to out/lateral_1.0/, which labelled a current-configuration
    # file as a superseded one and made the 300-path control look like an old
    # artefact of the 1.0 move set.
    mkdir -p out/previous
    [ -f "out/ambient_null_$tag.csv" ] && mv -f "out/ambient_null_$tag.csv" \
        "out/previous/ambient_null_${tag}_$(date '+%Y%m%d-%H%M').csv" 
    python3 tilt_recovery.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" \
        --amp 0 --tilts 0 --sites "$AMBIENT_SITES" --no-resume \
        --lateral "$LATERAL" --ncell "$NCELL" \
        --save-paths "ambient_paths_$tag.json" \
        --dir out --out "ambient_null_$tag.csv" 2>&1 | tee logs/ambient_$tag.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  done
  # ambient_null.py is the ANALYSIS of these traces, not the tracing, and its last
  # sentence compares the null against the hotspot statistic that lean_confirm.py
  # computes. It ran here, a whole step before lean_confirm.py, and so read the
  # PREVIOUS run's hotspot result: a current null printed beside a number from the
  # superseded cost channel, concluding that a claim was confirmed when the current
  # paths no longer support it. It now runs in step_derived, after lean_confirm.py,
  # and refuses to run if that file is older than the paths.
  return 0
}

# --------------------------------------------------------------- 6. corridors
step_corridors () {
  local m tag file var every zmax
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    case " $CORRIDOR_TAGS " in *" $tag "*) ;; *) continue;; esac
    [ -f "$file" ] || { echo "!! $tag: $file missing, skipping"; continue; }
    echo "== corridors $tag"
    python3 corridor_all.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" \
        --lateral "$LATERAL" --ncell "$NCELL" \
        ${CORRIDOR_DIAGNOSE:+--diagnose "$CORRIDOR_DIAGNOSE"} \
        2>&1 | tee "logs/corridors_$tag.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  done
  # The self-check is the corridor's own validity test: min(D + C) must equal the
  # hotspot's own cost. The forward field now prices a multi-cell move the same
  # way the backward field does, so this has to pass; if it does not, the two
  # fields have drifted apart again and the widths mean nothing.
  echo
  echo "--- corridor self-check lines ---"
  grep -i "self.check\|self check" logs/corridors_*.log || echo "(none printed)"
}

# ---------------------------------------------------------------- 7. geometry
step_geometry () {
  python3 corridor_ridge_association.py --tag RevealLO 2>&1 | tee logs/ridge_assoc.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  python3 plume_geometry.py --tag RevealLO 2>&1 | tee logs/plume_geometry.log
  return "${PIPESTATUS[0]}"
}

# ----------------------------------------------------------------- 9. derived
step_derived () {
  local rc=0
  # Plate kinematics first: absolute plate motion above each hotspot and the trench
  # tessellation the deep lean is measured against. plume_slant_apm.csv used to mix
  # these with path-derived columns (lean_az, tilt, offset) and stood eight days stale
  # in no runner while deformation_drivers.py read the stale half. It now writes the
  # kinematic columns only, from the plate model and the hotspot list alone, so it
  # depends on nothing else in this run and goes first. It needs pygplates and the
  # Zahirovic rotation model, so a missing dependency warns rather than failing the run
  # - the two files it writes do not move between runs.
  if python3 -c "import pygplates" 2>/dev/null; then
    python3 plume_slant_apm.py     2>&1 | tee logs/plume_slant_apm.log     || rc=1
  else
    echo "WARNING: pygplates unavailable, so out/plume_slant_apm.csv and"
    echo "         out/hinge_migration.csv keep their existing values. Both are"
    echo "         plate-model products and do not move when the paths do, so this"
    echo "         is safe here, but neither carries provenance from this run."
  fi
  # These three read the traced paths and were absent from the retrace, so a change
  # to the frozen configuration left them holding the previous run's numbers while
  # every other output moved. That is not a harmless staleness: bao_pdf feeds the
  # deflection distribution the audit checks, path_reliability feeds the stability
  # statement, and track_support decides how the section atlas is grouped - so the
  # atlas was being grouped by one configuration and drawn from another. They cost
  # minutes, and two of them do not even load the model.
  python3 bao_pdf.py --paths "out/conduit_paths_all_$PAPER_MODEL.json" \
          --tag "$PAPER_MODEL"      2>&1 | tee logs/bao_pdf.log         || rc=1
  # The depth census reads the paths, the ambient paths and the column products, so it
  # runs here again (it ran in step_columns too, which may precede a retrace of the
  # paths) and feeds Figure 6, Tables S4-S5 and the audit.
  python3 mid_mantle_census.py     2>&1 | tee logs/mid_mantle_census.log || rc=1
  python3 path_reliability.py --tag "$PAPER_MODEL" --control RevealLO_30km \
                                   2>&1 | tee logs/path_reliability.log || rc=1
  python3 track_support.py --file "$TOMO/RevealLO.nc" --tag "$PAPER_MODEL" \
                                   2>&1 | tee logs/track_support.log    || rc=1
  python3 plume_grouping.py        2>&1 | tee logs/plume_grouping.log   || rc=1
  python3 geochem_geometry.py      2>&1 | tee logs/geochem_geometry.log || rc=1
  python3 province_flip.py         2>&1 | tee logs/province_flip.log    || rc=1
  python3 deformation_drivers.py   2>&1 | tee logs/deformation.log      || rc=1
  python3 apm_crossmodel.py        2>&1 | tee logs/apm_crossmodel.log   || rc=1
  python3 lean_confirm.py --tags RevealLO,REVEAL,GLADM35,SPiRaL,SEMUCB-WM1,RevealLO_30km \
                                   2>&1 | tee logs/lean_confirm.log     || rc=1
  # The pre-registered ridge test (Text S3, Figure 4c) was in no runner: Text S3
  # quoted 24 near and 15 far paths at P = 0.657 and 0.963, traced under the
  # superseded configuration, beside a Figure 4c drawn from the current paths.
  # It now writes out/ridge_lean_<tag>.csv and the audit reads it.
  python3 ridge_lean_confirm.py --tags RevealLO,REVEAL,GLADM35 \
                                   2>&1 | tee logs/ridge_lean_confirm.log || rc=1
  # strictly after lean_confirm.py: it reads the hotspot side of its comparison from
  # that script's output, and refuses to run if the file predates the traced paths
  python3 ambient_null.py --tag "$PAPER_MODEL" \
                                   2>&1 | tee logs/ambient_null.log     || rc=1
  python3 crossmodel_table.py      2>&1 | tee logs/crossmodel_table.log || rc=1
  # Two more that were in no runner. occupancy_bound.py performs the section 3.7
  # exclusion test and writes occupancy_bound.csv, which fig_calibration.py now reads
  # to shade the excluded band rather than carrying it as a literal - so it must run
  # before the figures. supplement_table.py generates Table S1, all 49 rows of it,
  # and stood five days stale in the supplement because nothing re-ran it.
  python3 occupancy_bound.py --lateral "$LATERAL" --ncell "$NCELL" \
                                   2>&1 | tee logs/occupancy_bound.log  || rc=1
  python3 supplement_table.py      2>&1 | tee logs/supplement_table.log || rc=1
  # The worst cell each traced route has to cross, per site and per model, with what
  # the route ends on beside it. The cost integrates the anomaly along a path and is
  # indifferent to how the slow material is distributed, so a route through connected
  # rock and a route threading isolated blobs can price the same; this is the quantity
  # that separates them. --accepted-only skips the bottleneck field, which costs what
  # a cost field costs, so every model can afford this every run. The paper model then
  # gets the full comparison in the continuity step, which overwrites this file with
  # the achievable column filled in.
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every dmax <<< "$m"
    python3 bottleneck_report.py --accepted-only --file "$file" --tag "$tag" \
        --var "$var" --every "$every" --depth-max "$dmax" --ncell "$NCELL" \
        2>&1 | tee logs/bottleneck_"$tag".log || rc=1
  done
  return $rc
}

# ---------------------------------------------------------------- 10. figures
step_figures () {
  bash refigure.sh 2>&1 | tee logs/retrace_refigure.log
  return "${PIPESTATUS[0]}"
}

# A generated file that is not rebuilt by the step that owns its inputs will be read as
# current by something downstream, and the failure shows up as agreement rather than as
# an error. The sweep lists anything whose recorded inputs are now newer than it is, or
# that was written under a configuration other than the frozen one.
step_provenance () {
  python3 provenance.py --selftest 2>&1 | tee logs/provenance_selftest.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  python3 provenance.py --check --tag "$PAPER_MODEL" --pattern "*$PAPER_MODEL*" \
      2>&1 | tee logs/provenance.log
  return 0
}

# ------------------------------------------------------------------ 11. audit
# Expected to FAIL. Every number it flags is one the manuscript states and the
# retrace has changed. The manuscript moves to the computed value, never the
# reverse. It is run with `|| true` so the run finishes and the stamp is written.
step_audit () {
  # The inventory of scripts nothing calls, with whether their products predate the
  # current paths. Informational: it never fails the run, because most of these are
  # explorations that are meant to be orphans. It exists so that the list is in front
  # of someone every run rather than being rediscovered each time a stale number is
  # traced back to a script no runner has called since the configuration changed.
  python3 orphan_scripts.py 2>&1 | tee logs/orphan_scripts.log || true
  python3 audit_manuscript.py 2>&1 | tee logs/retrace_audit.log || true
  echo
  echo "The audit failures above are the work list. Change the manuscript to the"
  echo "computed value, never the computed value to the manuscript."
  return 0
}

# ------------------------------------------------------------------- 5. nulls
# Defined here, at the end of the file, but run in the order its number gives; the
# run_step block at the bottom is the authority on order.
# The two null populations the paper's headline comparisons are measured against.
# Neither script was in any runner. root_null.py traces 300 null roots for the province
# comparison and its own defaults are the SUPERSEDED move set, so the production values
# are passed explicitly; province_null_test.py reads its output and refuses to run
# against a null older than the paths. The third null - the 300 ambient paths behind the
# population test and the atlas grouping - is written by the ambient step above, which
# now passes --save-paths for exactly this reason.
step_nulls () {
  local m tag file var every zmax
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    [ "$tag" = "$PAPER_MODEL" ] || continue
    python3 root_null.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" \
        --lateral "$LATERAL" --ncell "$NCELL" \
        2>&1 | tee "logs/root_null_$tag.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  done
  python3 province_null_test.py --tag "$PAPER_MODEL" \
                                   2>&1 | tee logs/province_null.log
  return "${PIPESTATUS[0]}"
}

# ---------------------------------------------------------------- 5b. columns
# Two surface-anchored measurements that depend on the model alone and not on the
# path configuration: the connected slow column beneath each site (root_depth.py,
# section 3.1, Figure 6a, Table S4) and the tracked local minimum with its prominence
# (conduit_detect.py, section 3.2, Figure 6b, Table S5). Both ran once on 6-7 September,
# outside every runner and without provenance; mid_mantle_census.py reads them, so
# they run here, for every model, with the null seeds their scripts fix. The _xm suffix
# on the conduit profiles of the non-paper models is the one run_crossmodel_prominence.sh
# wrote and the census reads; the paper model's profile carries no suffix.
step_columns () {
  local m tag file var every zmax rc=0
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    [ "$tag" = "RevealLO_30km" ] && continue
    [ -f "$file" ] || { echo "!! $file missing"; return 1; }
    python3 root_depth.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" 2>&1 | tee "logs/root_depth_$tag.log" || rc=1
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    if [ "$tag" = "$PAPER_MODEL" ]; then suffix=""; else suffix="_xm"; fi
    python3 conduit_detect.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" --nulls 300 --suffix "$suffix" \
        2>&1 | tee "logs/conduit_detect_$tag.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    # Shell percentile along the traced paths in the five census bands, hotspots and
    # (where traced) ambient paths; the three-band version for the paper model is
    # track_support.py in step_derived.
    python3 band_percentile.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" 2>&1 | tee "logs/band_percentile_$tag.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
  done
  python3 mid_mantle_census.py 2>&1 | tee logs/mid_mantle_census.log || rc=1
  return "$rc"
}

# ------------------------------------------------------------- 8. calibration
# The injection sweeps behind section 3.7 and Figure 7. corridor_width_calibrate.py reads
# path_config and passes c.s and c.channel into cost_field, so its output moves with the
# configuration like everything else - and it was in no runner, so three of the four
# occupancy sweeps stood at the move set and channel of three different earlier days while
# the corridors they were compared against were retraced. Each sweep is 72 corridors, about
# an hour; run_calibration_all.sh skips any whose stored cfg_ columns already match the
# frozen configuration, so this step is cheap once it is current.
step_calibration () {
  bash run_width_calibration.sh full 2>&1 | tee logs/width_calibration_full.log
  [ "${PIPESTATUS[0]}" -eq 0 ] || return 1
  bash run_calibration_all.sh 2>&1 | tee logs/calibration_all.log
  return "${PIPESTATUS[0]}"
}

# ------------------------------------------------------- 9b. continuity frontier
# The cheapest route whose WORST cell is no worse than each of a ladder of levels,
# which is what the paths give up by not carrying a continuity term. frontier.py was
# in no runner, and the conclusion drawn from its one run - that no reweighting of
# cost against continuity helps - was computed on the anom channel and outlived it by
# five days and a whole retrace. It is here so that cannot happen again.
#
# One level per invocation, deliberately: successive solves in one process do not
# return the freed cost field to the allocator and the fourth allocation dies. The
# output is written after each level and resumed on the next run, so an interruption
# costs one level rather than the ladder. The paper model only; the ladder is about
# an hour and nothing cross-model rests on it.
step_continuity () {
  local m file var every rc=0
  # The second deep root every hotspot has in reach, and whether anything rests on
  # which of the two the search returns. One run does all 49; the second pass is a
  # no-op unless the first was killed partway, in which case it resumes. It refuses a
  # file written under a different configuration.
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every dmax <<< "$m"
    [ "$tag" = "$PAPER_MODEL" ] || continue
    for b in 1 2; do
      python3 competing_routes.py --file "$file" --tag "$tag" --var "$var" \
          --every "$every" --depth-max "$dmax" --lateral "$LATERAL" --ncell "$NCELL" \
          --save-paths "competing_tracks_$tag.json" \
          2>&1 | tee -a logs/competing_routes_"$tag".log
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    done
    python3 root_robustness.py --tag "$tag" \
        2>&1 | tee logs/root_robustness_"$tag".log || rc=1
  done
  # How much worse the least-cost route's worst cell is than it needed to be: the
  # minimax field against what the traced path accepts. This is the full form of what
  # derived writes with --accepted-only, and it is the paper model alone because the
  # field is a dynamic program the size of a cost field.
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every dmax <<< "$m"
    [ "$tag" = "$PAPER_MODEL" ] || continue
    python3 bottleneck_report.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$dmax" --ncell "$NCELL" \
        2>&1 | tee logs/bottleneck_full_"$tag".log || rc=1
  done
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every dmax <<< "$m"
    [ "$tag" = "$PAPER_MODEL" ] || continue
    for L in 100 50 35 20 12; do
      python3 frontier.py --file "$file" --tag "$tag" --var "$var" --every "$every" \
          --depth-max "$dmax" --levels "$L" 2>&1 | tee -a logs/frontier_"$tag".log
      [ "${PIPESTATUS[0]}" -eq 0 ] || rc=1
    done
  done
  return "$rc"
}


# Every name in STEPS must have a function, checked here where the functions exist.
# A typo in the list would otherwise surface as a skipped step rather than an error.
for s in $STEPS; do
  if ! declare -F "step_$s" > /dev/null; then
    echo "retrace.sh: STEPS names $s but there is no step_$s function" >&2
    exit 2
  fi
done

run_step preflight step_preflight

# Not a stamped step. The backup guards the only irreplaceable thing in the
# directory, and a stray stamp must never be able to skip it. It never overwrites
# what is already archived, so running it twice is harmless.
echo
echo "=============== backup  $(date '+%Y-%m-%d %H:%M') ==============="
step_backup || { echo "!! backup did not produce an archive - stopping" >&2; exit 1; }
for s in $STEPS; do
  [ "$s" = preflight ] && continue
  run_step "$s" "step_$s"
done

echo
echo "retrace complete at lateral $LATERAL, ncell $NCELL"
echo "the 1.0 run is in out/lateral_1.0/, logs in logs/"

#!/usr/bin/env bash
#
# Three-dimensional morphology beneath long-track hotspots.
#
# An extension of the hotspot root classification, not a replacement for it: the
# classification decides which hotspots the least-cost search roots, and this
# asks what the tomography admits beneath the ones it does not. Everything here
# reads the same detection tables and reuses the same cost field, so a corridor
# is the corridor of the published search rather than of a second search that
# resembles it.
#
# Run from the plume_pipeline directory, AFTER run_all.sh has produced
# detection_<tag>.csv and classification_<tag>.csv for the models used.
#
#   ./run_morphology.sh classes     the track classification only, seconds
#   ./run_morphology.sh pilot       the six-hotspot feasibility test, ~1 hour
#   ./run_morphology.sh full        every retained configuration, overnight
#   ./run_morphology.sh synth       the complex synthetic morphologies
#   ./run_morphology.sh crossmodel  repeat the pilot in the other models
#   ./run_morphology.sh smoothed    write laterally smoothed copies of RevealLO
#   ./run_morphology.sh scales      repeat the pilot on each smoothed copy
#   ./run_morphology.sh floor       the resolution floor, by injection recovery
#   ./run_morphology.sh figures     rebuild the figures from existing tables
#
# JOBS sets how many configurations run at once. Each worker holds the model
# twice over plus two cost fields, about 2.5 GB for RevealLO at native sampling,
# so six needs roughly 16 GB.
#
#   JOBS=6 ./run_morphology.sh pilot
#
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out figures

ROOT="$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction"
SSD="${SSD:-/Volumes/ZeitwissenDataSSD}"

# The tomographic models are large and may have been moved to an external disk.
# Rather than hard-coding one location, the first candidate that actually holds
# RevealLO.nc is used, so moving the folder does not mean editing scripts, and
# an unplugged SSD produces the ordinary "not found, skipping" path rather than
# a run that silently uses a stale copy. TOMO in the environment overrides all
# of it.
find_tomo () {
  local c
  for c in "${TOMO:-}" \
           "$ROOT/REVEAL_mantle_tomography" \
           "$SSD/archive/Muller_mantle_tomography_subduction/REVEAL_mantle_tomography" \
           "$SSD/REVEAL_mantle_tomography"; do
    [ -n "$c" ] && [ -f "$c/RevealLO.nc" ] && { echo "$c"; return; }
  done
  echo "$ROOT/REVEAL_mantle_tomography"
}
TOMO="$(find_tomo)"
echo "tomographic models: $TOMO"

# Where the smoothed copies are written and read. They belong with the model they
# were made from, so if the tomography folder is on the external disk they follow
# it there; otherwise the external disk is used when it is mounted and the
# internal one only when it is not. Set SMOOTH_DIR to override.
find_smooth_dir () {
  local c
  case "$TOMO" in /Volumes/*) echo "$TOMO"; return;; esac
  for c in "$SSD/archive/Muller_mantle_tomography_subduction/REVEAL_mantle_tomography" \
           "$SSD/REVEAL_mantle_tomography"; do
    [ -d "$c" ] && { echo "$c"; return; }
  done
  if [ -d "$SSD" ]; then
    echo "$SSD/archive/Muller_mantle_tomography_subduction/REVEAL_mantle_tomography"
  else
    echo "$TOMO"
  fi
}
SMOOTH_DIR="${SMOOTH_DIR:-$(find_smooth_dir)}"
echo "smoothed copies:    $SMOOTH_DIR"

# A smoothed copy written to one disk and then looked for on another is not an
# error a run should discover halfway through, so a missing external disk is
# reported here, once, while it can still be acted on.
for _d in "$TOMO" "$SMOOTH_DIR"; do
  case "$_d" in
    /Volumes/*) [ -d "$_d" ] || echo "!! $_d is not there; is the external disk mounted?" >&2;;
  esac
done
AGES="$ROOT/Papers/Chase_Wessel_2022_data/PHT2021_raw_data/PHT2021_pacific_ages.txt"

PAPER_MODEL=RevealLO
PAPER_FILE="$TOMO/RevealLO.nc"
PAPER_VAR=voigt
PAPER_EVERY=1
PAPER_ZMAX=2880

OTHERS=(
  "REVEAL|$ROOT/REVEAL_vs_full.nc|voigt|1|2880"
  "GLADM35|$TOMO/GLADM35.nc|voigt|2|2890"
  "SPiRaL|$TOMO/SPiRaL.nc|voigt|1|2891"
  "SEMUCB-WM1|$TOMO/SEMUCB-WM1.nc|vs|1|2891"
)

JOBS="${JOBS:-1}"
STAGE="${1:-pilot}"

classes () {
  python3 track_classes.py --ages "$AGES" 2>&1 | tee out/track_classes.log
  echo
  echo "The classification is now fixed. Review out/track_classes.csv BEFORE"
  echo "looking at any corridor: changing it afterwards makes every comparison"
  echo "below a comparison against a variable chosen to produce the answer."
}

pilot () {
  python3 morph_run.py --file "$PAPER_FILE" --tag "$PAPER_MODEL" \
      --var "$PAPER_VAR" --every "$PAPER_EVERY" --depth-max "$PAPER_ZMAX" \
      --targets morph_targets.txt --corridor --configs 12 --nulls 40 \
      --jobs "$JOBS" --suffix _pilot 2>&1 | tee out/morph_pilot.log
  python3 morph_metrics.py --tag "$PAPER_MODEL" --suffix _pilot \
      2>&1 | tee -a out/morph_pilot.log
  python3 morph_stats.py --tag "$PAPER_MODEL" --suffix _pilot \
      2>&1 | tee out/morph_stats_pilot.log
  python3 fig_paradox.py
  python3 fig_morphology.py --tag "$PAPER_MODEL" --suffix _pilot
  echo
  echo "GATE. Read out/morph_stats_pilot.log before going on. The full run is"
  echo "worth its cost only if the pilot shows a difference between the"
  echo "corridors of the long-track hotspots the search roots and those it does"
  echo "not, or shows those corridors leaving the matched null distribution."
  echo "If every corridor sits inside the grey band, the honest result is that"
  echo "the tomography does not resolve a difference, and that belongs in a"
  echo "methodological paper rather than a reinterpretation."
}

full () {
  python3 morph_run.py --file "$PAPER_FILE" --tag "$PAPER_MODEL" \
      --var "$PAPER_VAR" --every "$PAPER_EVERY" --depth-max "$PAPER_ZMAX" \
      --targets morph_targets.txt --corridor --nulls 60 \
      --jobs "$JOBS" 2>&1 | tee out/morph_full.log
  python3 morph_metrics.py --tag "$PAPER_MODEL" 2>&1 | tee -a out/morph_full.log
  python3 morph_stats.py --tag "$PAPER_MODEL" 2>&1 | tee out/morph_stats.log
  python3 fig_morphology.py --tag "$PAPER_MODEL"
}

synth () {
  python3 synth_morph.py --file "$PAPER_FILE" --tag "$PAPER_MODEL" \
      --var "$PAPER_VAR" --every "$PAPER_EVERY" --depth-max "$PAPER_ZMAX" \
      --configs 8 --corridor --jobs "$JOBS" 2>&1 | tee out/synth_morph.log
  python3 fig_synth_morph.py --tag "$PAPER_MODEL"
}

crossmodel () {
  for m in "${OTHERS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    [ -f "$file" ] || { echo "!! $tag: $file not found, skipping"; continue; }
    [ -f "out/detection_$tag.csv" ] || { echo "!! $tag: no detection table"; continue; }
    echo "== morphology $tag"
    python3 morph_run.py --file "$file" --tag "$tag" --var "$var" \
        --every "$every" --depth-max "$zmax" --targets morph_targets.txt \
        --corridor --configs 12 --nulls 40 --jobs "$JOBS" --suffix _pilot \
        2>&1 | tee "out/morph_pilot_$tag.log"
    python3 morph_metrics.py --tag "$tag" --suffix _pilot >/dev/null
  done
  python3 morph_compare.py --tags "$PAPER_MODEL" REVEAL GLADM35 SPiRaL \
      SEMUCB-WM1 --suffix _pilot 2>&1 | tee out/morph_crossmodel.log
}

SCALES="${SCALES:-200 400 800 1600}"

smoothed () {
  mkdir -p "$SMOOTH_DIR"
  python3 make_smoothed.py --file "$PAPER_FILE" --var "$PAPER_VAR" \
      --sigma $SCALES --tag "$PAPER_MODEL" --depth-max "$PAPER_ZMAX" \
      --out-dir "$SMOOTH_DIR" 2>&1 | tee out/make_smoothed.log
}

scales () {
  for sig in $SCALES; do
    f="$SMOOTH_DIR/${PAPER_MODEL}_s${sig}.nc"
    tag="${PAPER_MODEL}_s${sig}"
    if [ ! -f "$f" ]; then
      echo "!! $tag: $f not found; run the smoothed stage first"
      continue
    fi
    # The configuration set is COPIED from the native model rather than
    # recalibrated. The question the ladder asks is what changes when spatial
    # detail is removed, and recalibrating would let the set of retained
    # configurations change at the same time, so a difference between two rungs
    # would no longer be attributable to resolution alone.
    [ -f "out/detection_$tag.csv" ] || \
        cp "out/detection_${PAPER_MODEL}.csv" "out/detection_$tag.csv"
    [ -f "out/classification_$tag.csv" ] || \
        cp "out/classification_${PAPER_MODEL}.csv" "out/classification_$tag.csv"
    echo "== corridors at ${sig} km smoothing"
    # Twelve nulls, not forty, and eight configurations, not twelve. The scale
    # ladder compares a hotspot against ITSELF at native sampling, so the large
    # matched-null set that the cross-sectional comparison needs is redundant
    # here; a dozen are kept only so that "the corridors widened by X per cent"
    # can be set against how much random locations widened, which is the control
    # that matters. Each rung is then about a fifth of the pilot rather than the
    # whole of it, and four rungs cost less than one pilot did.
    python3 morph_run.py --file "$f" --tag "$tag" --var vs \
        --depth-max "$PAPER_ZMAX" --targets morph_targets.txt --corridor \
        --configs 8 --nulls 12 --jobs "$JOBS" --suffix _pilot \
        2>&1 | tee "out/morph_pilot_$tag.log"
    python3 morph_metrics.py --tag "$tag" --suffix _pilot >/dev/null
  done
  python3 morph_scales.py --base "$PAPER_MODEL" 2>&1 | tee out/morph_scales.log
}

floor () {
  python3 synth_morph.py --file "$PAPER_FILE" --tag "$PAPER_MODEL" \
      --var "$PAPER_VAR" --every "$PAPER_EVERY" --depth-max "$PAPER_ZMAX" \
      --configs 8 --sites 2 --amps -0.5 -0.75 -1.0 -1.5 \
      --radii 100 150 200 300 450 600 --shapes vertical uniform \
      --jobs "$JOBS" --suffix _floor 2>&1 | tee out/synth_floor.log
}

figures () {
  python3 fig_paradox.py
  for s in _pilot ''; do
    [ -f "out/morph_profile_${PAPER_MODEL}${s}.csv" ] || continue
    python3 fig_morphology.py --tag "$PAPER_MODEL" --suffix "$s"
  done
  [ -f "out/synth_morph_${PAPER_MODEL}.csv" ] && \
      python3 fig_synth_morph.py --tag "$PAPER_MODEL"
  [ -f "out/morph_displacement_by_model.csv" ] && python3 fig_displacement.py
  [ -f "out/morph_scales.csv" ] && python3 morph_scales.py --base "$PAPER_MODEL" \
      >/dev/null
}

case "$STAGE" in
  classes)    classes ;;
  pilot)      classes; pilot ;;
  full)       full ;;
  synth)      synth ;;
  crossmodel) crossmodel ;;
  smoothed)   smoothed ;;
  scales)     scales ;;
  floor)      floor ;;
  figures)    figures ;;
  *) echo "unknown stage: $STAGE" >&2; exit 1 ;;
esac
echo
echo "done. tables in out/, figures in figures/"

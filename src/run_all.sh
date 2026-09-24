#!/usr/bin/env bash
#
# Hotspot root classification by least-cost path search.
#
# Reproduces every number and figure in Müller, Fichtner & Schiller, "Which
# Hotspots Are Rooted in the Deep Mantle?". Run from the plume_pipeline
# directory. Each model is independent, so the per-model loop can be split
# across machines; the aggregation steps need all of them.
#
#   ./run_all.sh              run everything, skipping models already done
#   FORCE=1 ./run_all.sh      re-run models even if their output exists
#   ./run_all.sh RevealLO      run one model, then the aggregation steps
#
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p out figures

# Model files are not redistributed with this repository. All of those used in
# the study were provided by the Seismology and Wave Physics group at ETH
# Zürich; RevealLO is not yet released. See the "Model availability" section of
# the README. Any model whose file is missing is skipped, so this script runs
# correctly on whatever subset you have.
ROOT="$HOME/Documents/Papers/in_prep/Muller_mantle_tomography_subduction"
TOMO="$ROOT/REVEAL_mantle_tomography"

# Two comparisons read compilations published by other authors, which are not
# redistributed here. Each step is skipped when its file is absent.
#   Hardardottir & Jackson (2025), the ocean island basalt geochemical database
#   Chase & Wessel (2021), the raw Pacific radiometric age table on Zenodo
OIBID="$ROOT/Papers/Harðardottir_Jackson2025-Supp_dataset1.xlsx"
PACIFIC_AGES="$ROOT/Papers/Chase_Wessel_2022_data/PHT2021_raw_data/PHT2021_pacific_ages.txt"

# tag | file | velocity variable | depth decimation | deepest shell to keep (km)
#
# The velocity variable is 'voigt' to build the Voigt average from vsv and vsh,
# or the name of a variable already in the file; SEMUCB-WM1 is distributed as a
# Voigt average under 'vs'. Names are matched case-insensitively.
#
# Decimation thins the depth axis. It is safe here because the search prices a
# path by physical length and amplitude rather than by counting connected cells,
# so the answer does not depend on shell spacing - unlike a percolation or
# connected-component analysis, where it moves the threshold by a factor of
# thirty. RevealLO_30km exists to demonstrate exactly that.
#
# The depth cut matters more than it looks. RevealLO is distributed over the
# whole Earth radius, so its shells at 2890 and 2900 km straddle the core-mantle
# boundary at 2891 km and carry velocities where shear collapses; they read as
# anomalies of -97 per cent and would make a path to the deep mantle nearly
# free. Every other model here stops in the mantle and keeps its deepest shell.
MODELS=(
  "RevealLO|$TOMO/RevealLO.nc|voigt|1|2880"
  "RevealLO_30km|$TOMO/RevealLO.nc|voigt|3|2880"
  "REVEAL|$ROOT/REVEAL_vs_full.nc|voigt|1|2880"
  "GLADM35|$TOMO/GLADM35.nc|voigt|2|2890"
  "SPiRaL|$TOMO/SPiRaL.nc|voigt|1|2891"
  "SEMUCB-WM1|$TOMO/SEMUCB-WM1.nc|vs|1|2891"
)

# The model whose recovered conduits define the nine great circles every
# cross-section panel is cut on. Only this one gets --fix-azimuths; every other
# model is then drawn on the same planes, so the panels compare directly.
PAPER_MODEL=RevealLO

# Injections to run at once. Calibration is 48 independent injections per model
# and numpy is single-threaded, so this is the difference between one core and
# all of them. Each worker holds roughly two and a half copies of the model, so
# eight needs about 21 GB for RevealLO at native sampling; drop it if memory is
# tight, raise it for the smaller models.
#
#   JOBS=8 ./run_all.sh
JOBS="${JOBS:-1}"

# The move set the search may express, chosen by slant_calibrate.py on the
# criterion that a hotspot path should find slower material than an ambient path
# traced at the same setting. One lateral cell per slanted move caps the steepest
# expressible tilt near 64 degrees, and conduits leaning harder than that cannot
# be followed at any price; two cells raise the cap to about 76.
#
# These defaults reproduce the published run. To retrace at the calibrated move
# set, run with LATERAL=0.60 NCELL=2. Everything downstream of the cost field -
# the classification, the paths, the corridors, the depth ladder and the null
# snapshot - reads them, so a run cannot end up half at one setting and half at
# the other.
LATERAL="${LATERAL:-1.0}"
NCELL="${NCELL:-1}"
MOVESET=(--lateral "$LATERAL" --ncell "$NCELL")
echo "move set: lateral=$LATERAL ncell=$NCELL"

ONLY="${1:-}"

for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue
  if [ ! -f "$file" ]; then
    echo "!! $tag: $file not found, skipping" >&2
    continue
  fi
  if [ -f "out/classification_$tag.csv" ] && [ -z "${FORCE:-}" ]; then
    echo "== $tag: already done, skipping (FORCE=1 to re-run)"
    continue
  fi
  echo "== $tag  ($(basename "$file"), every=$every, depth<=${zmax} km)"
  python3 classify.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" --jobs "$JOBS" "${MOVESET[@]}" \
      2>&1 | tee "out/cls_$tag.log"
done

# Everything below needs the whole set.
if [ -n "$ONLY" ]; then
  echo "single model done; re-run without an argument for the aggregation steps"
  exit 0
fi

IFS='|' read -r tag file var every zmax <<< "${MODELS[0]}"
python3 null_snapshot.py --file "$file" --tag "$tag" --var "$var" \
    --every "$every" --depth-max "$zmax" "${MOVESET[@]}"

python3 compare.py            | tee out/compare.log
python3 table.py              | tee out/table.log
python3 longevity_analysis.py | tee out/longevity.log

# Depth-ladder connectivity and the geometry of each recovered connection, for
# the model the sections are cut on. It reads detection_<tag>.csv to pick the
# same median calibrated configuration the rest of the study uses, so it has to
# follow classify.py rather than be run against an older calibration.
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  [ "$tag" = "$PAPER_MODEL" ] || continue
  [ -f "out/detection_$tag.csv" ] || continue
  echo "== tracked depth $tag"
  python3 track_depth.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" "${MOVESET[@]}" \
      2>&1 | tee "out/track_$tag.log"
done

# Lowermost-mantle province membership, and the anomaly beneath the sites the
# discussion compares. Both read the paper model directly.
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  [ -f "$file" ] || continue
  # The site profiles and the shell root-mean-square are compared across models,
  # so every model gets them; province membership is only used for the paper's.
  echo "== site profiles $tag"
  python3 site_profile.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" 2>&1 | tee "out/site_profile_$tag.log"
  [ "$tag" = "$PAPER_MODEL" ] || continue
  python3 province.py --file "$file" --tag "$tag" --var "$var" \
      --every "$every" --depth-max "$zmax" 2>&1 | tee "out/province_$tag.log"
done

if [ -f "$OIBID" ]; then
  python3 geochem_link.py --oibid "$OIBID" --tag "$PAPER_MODEL" \
      2>&1 | tee "out/geochem_link_$PAPER_MODEL.log"
else
  echo "skipping geochem_link.py: OIBID does not point at a file"
fi
if [ -f "$PACIFIC_AGES" ]; then
  python3 ages_workbook.py --ages "$PACIFIC_AGES"
else
  echo "skipping ages_workbook.py: PACIFIC_AGES does not point at a file"
fi

# Every statistic the manuscript quotes, in one file. The manuscript build holds
# its numbers as literals, so without this a re-run leaves figures in the text
# that no longer follow from the data, and nothing errors.
python3 manuscript_stats.py --paper-model "$PAPER_MODEL" \
    | tee out/manuscript_stats.log

# Cross sections. There is no longer a nine-panel figure: the paper carries the
# paper model's full set of 49, three by three over six pages and ranked by the
# consensus across models, and each other model contributes its highest-ranked
# nine to the supplement. The paper model is traced first and derives the section
# planes for all 49; the rest reuse them, so a hotspot is cut on the same great
# circle in every model and the panels can be laid side by side.
for pass in fix rest; do
  for m in "${MODELS[@]}"; do
    IFS='|' read -r tag file var every zmax <<< "$m"
    [ -n "$ONLY" ] && [ "$ONLY" != "$tag" ] && continue
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
        --every "$every" --depth-max "$zmax" --all "${MOVESET[@]}" $fix
  done
done
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag file var every zmax <<< "$m"
  [ -f "out/conduit_paths_all_$tag.json" ] || continue
  if [ "$tag" = "$PAPER_MODEL" ]; then
    python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --all
    # Figure 3 of the paper is the highest-ranked nine on the paper model. It was
    # not built here, so it survived a rebuild as whatever the last hand-run left.
    python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --top 9
  else
    python3 sections_figure.py --file "$file" --tag "$tag" --var "$var" --top 9
  fi
done

# The standalone figures run LAST and cannot abort the run. They used to sit
# ahead of the path tracing, so a label collision - which is a figure defect and
# nothing to do with the computation - threw away the hours of tracing that had
# not happened yet. A failure here is reported loudly and sets the exit status,
# but only after every computed product exists on disk.
figfail=""
for f in sensitivity_figure.py map_figure.py longevity_figure.py; do
  echo "== figure $f"
  python3 "$f" || figfail="$figfail $f"
done

echo
if [ -n "$figfail" ]; then
  echo "!! these figures failed their own checks:$figfail"
  echo "   every computed product is on disk; fix the figures and re-run them alone."
  echo "done with figure failures. tables in out/, figures in figures/"
  exit 1
fi
echo "done. tables in out/, figures in figures/"

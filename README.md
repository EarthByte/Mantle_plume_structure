# Mantle_plume_structure

Code, figures and derived data for

> Müller, R. D., Fichtner, A., & Schiller, C. J. (2026). Full-waveform tomography reveals deep mantle plume conduits with non-unique source regions. *Geochemistry, Geophysics, Geosystems*.

Whether global tomography resolves mantle plume conduits was disputed for two decades, and the debate largely subsided: ray-based whole-mantle models changed little over the following decade, and the vertical smoothing that had made early finite-frequency models appear plume-rich was itself called into question. Nonlinear full-waveform inversion, fit to far larger seismic datasets, now resolves whole-mantle structure differently, making it timely to revisit these questions. The full-waveform models REVEAL and RevealLO now show slow columns beneath most hotspots, opening opportunities for detailed plume geometry analysis. We trace least-cost descending paths beneath 49 hotspots and from 300 ambient locations (sites at least 1500 km from any hotspot, used throughout as a control for what the search itself can produce), calibrate them with injected synthetic conduits, and analyse their properties by depth against the ambient population. A connected slow column reaches the base of the mantle beneath 22–27% of hotspots against 4–7% of random sites, a contrast other tomography models render weakly or not at all. However, many plume columns are shown to be intermittent. Hotspot conduits are more prominent beneath the lower mantle viscosity increase inferred across 800–1200 km rather than above it. In RevealLO, our preferred model, hotspot paths are more inclined than ambient paths at 1000–1500 km (19° against 14°, *P* = 0.003) and nowhere else. Deflections show no horizon: the 800–1200 km range of the viscosity increase holds at most 1.5 times the deflections expected from its thickness, ambient paths deflect at the same depths, and corridor width grows smoothly with depth as the resolving length. The lower mantle viscosity increase is thus expressed as plume column prominence and inclination beneath it. Traced endpoints lie within the low-velocity provinces 2.6 times more often than matched ambient endpoints, at no preferred position, and 33 of 49 hotspots have a second column within 3% of cost ending a median 808 km away. Tomography with the current generation of models therefore does not always uniquely determine an individual plume's source, even where plume conduits are imaged.

A conduit is represented as the **cheapest descending path** through a tomographic shear-velocity model. A step of physical length `L` through material whose anomaly is `a` per cent costs

```
w = L * exp(s * a)
```

so slow material is cheap to traverse and lateral wandering is penalised. Paths must descend monotonically - that is what separates a conduit from a tour of the low-velocity network - but are otherwise free to move sideways, so the lateral offset of a conduit is **recovered from the data rather than assumed**. The search runs in reverse, as a multi-source sweep upward from the deep mantle by min-plus dynamic programming, so one sweep yields the cost from every cell in the model and a hotspot or a null location is a lookup.

Two things make a result out of that. What the search can *see* is fixed by requiring it to recover synthetic conduits of known amplitude, which puts a stated sensitivity floor on every non-detection. What *counts* as a detection is fixed by comparing each hotspot with random locations put through the identical procedure. Neither is a free choice.

This repository holds the analysis code and the finished figures. The tomographic models are not redistributed (see *Model availability*), so the derived products the figures are drawn from (`out/`) are not carried either: `src/run_all.sh` rebuilds them from the models.

## Layout

```
src/            analysis and figure scripts (Python 3), and the shell drivers below
  run_all.sh          the whole workflow: trace, calibrate, classify, tabulate
  retrace.sh          the same, step by step at the calibrated move set, with resume (FROM=<step>)
  refigure.sh         redraw every figure from what an earlier run_all.sh already wrote
  path_cost.py        the least-cost path search
  tomo_io.py          model loading, depth cropping, file screening
  contrast.py         the local-contrast channel
  inject.py           synthetic conduits, for calibration
  classify.py         calibrate on a model, then classify on it
  paths.py            recover conduit geometry and section planes
  null_snapshot.py    the null distribution behind the classification
  compare.py          merge the models, agreement, correlations
  table.py            the ranked table
  track_depth.py      the depth ladder, and the geometry of each connection
  site_profile.py     strongest anomaly beneath a hotspot, against depth
  province.py         membership of the lowermost-mantle slow provinces
  longevity_analysis.py / longevity.py   hotspot-track longevity
  sections_figure.py  the annular cross-section atlas (Figures 9-12, S3a-c)
  tasman_split.py / tasman_slab_footprint.py / tasman_slab_compare.py
                      the Tasmantid-Lord Howe test of Mather et al. (2026), Text S9
  model_update.py     RevealLO against REVEAL, shell by shell
  fig*.py             the remaining main-text and supplementary figures
  manuscript_stats.py every statistic the paper quotes, in one file
  figcheck.py / figstyle.py   figure-legibility and label-collision checks
data/           small inputs carried with the repository (see Inputs)
figures/        the twelve main-text and supplementary figures S1-S8, PDF and PNG;
                figures/tasman_split/ holds the unnumbered regional figures behind Text S9
```

`out/` (derived products of every step) and `results/` are not tracked. Running `src/run_all.sh` regenerates both from scratch.

## Model availability

**None of the tomographic model files used here are redistributed in this repository.**

Every model file used in this study was provided by the Seismology and Wave Physics group at ETH Zürich. **RevealLO is described by Schiller et al. (2026) and is being archived by its authors; see the paper's Open Research section for its DOI.** The other four are published models, obtainable from their authors at the references below; the copies used here are ETH-prepared and will not be byte-identical to the authors' own distributions, so a run from an independently obtained copy may differ in detail from the numbers in the paper.

| model | as used here | reference |
|---|---|---|
| RevealLO | 0.5°, 10 km depth, cropped at 2880 km | Schiller et al. (2026), ETH Zürich, Seismology and Wave Physics |
| REVEAL | 0.5° | Thrastarson et al. (2024), *BSSA* 114, 1392-1406 |
| GLAD-M35 | 1°, every 2nd shell | Cui et al. (2024), *GJI* 239, 478-502 |
| SPiRaL | as distributed | Simmons et al. (2021), *GJI* 227, 1366-1391 |
| SEMUCB-WM1 | as distributed | French & Romanowicz (2014), *GJI* 199, 1303-1327 |

`src/tomo_io.load_anomaly` absorbs most differences in file layout on its own (variable and coordinate names matched case-insensitively, depth in metres or kilometres, longitude on -180..180 or 0..360, dimensions in any order). Two things it cannot infer, and that must be set per model in the `MODELS` array at the top of `src/run_all.sh`: `--var`, whether the file carries `vsv`/`vsh` to Voigt-average or a single pre-averaged variable, and `--depth-max`, which has to sit above the core-mantle boundary at 2891 km. The driver skips any model whose file it cannot find, so `run_all.sh` runs correctly on whatever subset is available.

## Inputs

Carried in `data/`:

- `hotspots_courtillot2003.csv`: the 49-hotspot list of Courtillot et al. (2003), *EPSL* 205, 295-308, redistributed here unmodified.
- `oib_hotspot_names.csv`, `hotspot_basin.csv`: hotspot naming and basin-membership lookups used by the classification and figure scripts.
- `bbs2007_figure14_ranking.csv`: the hotspot-longevity ranking of Boschi, Becker & Steinberger (2007), used by `src/longevity_analysis.py`.
- `ridges_presentday_zahirovic2022.csv` and `S2_Shapefiles_Extinct_and_active_ridge_segments/`: present-day and extinct/active ridge geometries following Zahirovic et al. (2022) and MacLeod et al. (2017), used by the ridge-proximity tests.

Obtained from their sources when the analysis is rerun from scratch: the five tomographic models above, and the ocean island basalt geochemical database of Hardardottir & Jackson (2025) and the raw Pacific radiometric age table of Chase & Wessel (2021) (https://doi.org/10.5281/zenodo.5576466, CC-BY-4.0), for the two steps that compare against them - each is skipped when its file is absent.

## Reproducing

```
python3 -m pip install -r requirements.txt
```

`pdfplumber`, needed only by `figcheck.py`, wants a Pillow newer than several other packages accept, so it gets its own environment rather than installing in front of whatever else is there:

```bash
python3 -m venv .venv-figcheck
.venv-figcheck/bin/pip install -q pdfplumber
```

Edit the `MODELS` array at the top of `src/run_all.sh` (and of `src/retrace.sh`) to point at your copies of the tomographic models, then

```bash
cd src
./run_all.sh                 # everything, skipping models already done
FORCE=1 ./run_all.sh         # re-run models even where output exists
./run_all.sh SEMUCB-WM1      # a single model
```

Models are independent, so the per-model loop splits across machines; the aggregation steps need whatever set you want compared. `refigure.sh` redraws every figure from what `run_all.sh` already wrote, without recomputing anything, and finishes by checking them against the width they are placed at in the paper:

```bash
cd src && ./refigure.sh        # or ./refigure.sh GLADM35 for one model
./checkfigs.sh                 # the same check on its own
```

**Cost.** The expensive step is calibration: one cost field per configuration per injected conduit, 72 configurations and 32 conduits, so roughly 1200 sweeps per model. A sweep is a few seconds at 1 degree and about four times that at 0.5 degrees. Budget one to two hours for a half-degree model at ~100 depth shells, four to six for RevealLO at its native 10 km sampling.

**Depth decimation is safe here.** `--every N` thins the depth axis, and the answer does not depend on shell spacing, because the search prices a path by physical length and amplitude rather than by counting connected cells - unlike a percolation or connected-component analysis, where the threshold moves by a factor of thirty between REVEAL's 97 shells and RevealLO's 10 km spacing.

## Rebuilding the figures

```
python3 src/fig1_map.py          # Figure 1: hotspots and traced paths
python3 src/fig_workflow.py      # Figure 2: workflow
python3 src/fig_corridor.py      # Figure 3: near-optimal corridor width
python3 src/fig_lean.py          # Figure 4: directional tests of path lean
python3 src/fig_bands.py         # Figure 5: where conduits differ from ambient mantle, by depth
python3 src/fig_province.py      # Figure 6: traced endpoints and the lowermost-mantle slow provinces
python3 src/fig_deflection.py    # Figure 7: distribution of path-deflection depths
python3 src/fig_calibration.py   # Figure 8: synthetic calibration of corridor width
python3 src/sections_figure.py --all              # Figures 9-12: the 49-hotspot cross-section atlas
python3 src/fig_si_sweeps.py     # Figure S1: calibration sweeps
python3 src/fig_si_profiles.py   # Figure S2: corridor profiles
python3 src/sections_figure.py --top 9   # Figures S3a-c: best-constrained sections, other models
python3 src/fig_tasman_slab.py   # Figure S4: the slab of Mather et al. (2026) over the tomography
```

Figures S5-S8 describe how RevealLO was built (data coverage, sensitivity, the model update and the model itself). They were supplied by the model's authors and are included here as they appear in the Supporting Information; nothing in `src/` draws them.

`sections_figure.py` needs `--file/--tag/--var` for the model being drawn; `refigure.sh` shows the full invocations, including the paper model's `--fix-azimuths` pass that derives the section planes every other model is then drawn on. `figures/figure_map.txt` is the authority for which file is which figure number.

## Three ways a model file will break this, all of them silent

**Whole-radius depth axes.** RevealLO is distributed from the surface to the centre of the Earth, so shells straddling the core-mantle boundary at 2891 km carry velocities where shear collapses and read as anomalies near -97%, which would make a path to the deep mantle almost free. `--depth-max` crops above the boundary.

**Files assembled from several depth ranges.** Joins between depth ranges can average across velocities belonging to different depths and produce spurious extreme anomalies. `tomo_io.check_shells` refuses to run on any model in which a shell's 99.9th-percentile absolute anomaly exceeds four times the median over all shells below 300 km.

**Naming and layout conventions.** Variables appear as `vsv`/`VSV`, depth in metres or kilometres, longitude on -180..180 or 0..360, dimensions in either order, and a repeated wrap meridian at +-180 that would let a path cross the seam for nothing. `tomo_io.load_anomaly` normalises all of these; a depth axis is taken to be in metres if its maximum exceeds 2e4, since a depth in kilometres cannot exceed the Earth's radius.

## Conventions

Figures carry no titles and no text below 8 pt at the printed width; `checkfigs.sh` enforces this on the PDFs. `sections_figure.py` and the other GMT-drawn figures quote every size through a `pt()` helper as the size it will print at, scaled to the drawing canvas, so changing a panel's pitch cannot quietly shrink the type.

## Licences

The code in `src/` is released under the MIT licence (`LICENSE`). The figures in `figures/` are released under CC BY 4.0. Third-party data in `data/` retain the licences of their sources, named above. The tomographic models are the property of their authors and are not redistributed here.

## Citation

Müller, R. D., Fichtner, A., & Schiller, C. J. (2026). Full-waveform tomography reveals deep mantle plume conduits with non-unique source regions. *Geochemistry, Geophysics, Geosystems*.

`CITATION.cff` carries the same in machine-readable form. Please also cite the tomographic models you use, listed above.

## References for the inputs

Boschi, L., Becker, T. W., & Steinberger, B. (2007). Mantle plumes: Dynamic models and seismic images. *Geochemistry, Geophysics, Geosystems*, 8, Q10006. https://doi.org/10.1029/2007GC001733
Chase, C. G., & Wessel, P. (2021). PHT2021: Pacific hotspot track ages. Zenodo. https://doi.org/10.5281/zenodo.5576466
Mather, B. R., Seton, M., et al. (2026). Parallel volcanic chains generated by plume-slab interaction. *Gondwana Research*, 160, 99-108. https://doi.org/10.1016/j.gr.2026.08.002
Müller, R. D., et al. (2019). A global plate model including lithospheric deformation along major rifts and orogens since the Triassic. *Tectonics*, 38, 1884-1907. https://doi.org/10.1029/2018TC005462
Courtillot, V., Davaille, A., Besse, J., & Stock, J. (2003). Three distinct types of hotspots in the Earth's mantle. *Earth and Planetary Science Letters*, 205, 295-308. https://doi.org/10.1016/S0012-821X(02)01048-8
Cui, C., et al. (2024). GLAD-M35. *Geophysical Journal International*, 239, 478-502. https://doi.org/10.1093/gji/ggae270
French, S. W., & Romanowicz, B. A. (2014). SEMUCB-WM1. *Geophysical Journal International*, 199, 1303-1327. https://doi.org/10.1093/gji/ggu334
MacLeod, C. J., et al. (2017). Extinct and active spreading ridge segments. Supplementary dataset.
Schiller, C. J., et al. (2026). RevealLO. Full reference in the paper.
Simmons, N. A., et al. (2021). SPiRaL. *Geophysical Journal International*, 227, 1366-1391.
Thrastarson, S., et al. (2024). REVEAL. *Bulletin of the Seismological Society of America*, 114, 1392-1406.
Zahirovic, S., et al. (2022). Plate model reconstruction.

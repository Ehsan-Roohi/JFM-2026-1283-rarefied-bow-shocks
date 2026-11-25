# Rarefaction-induced inflation and similarity breakdown of hypersonic bow shocks over a circular cylinder

**Ehsan Roohi and Ahmad Shoja-Sani** · JFM-2026-1283 · manuscript and supplementary material revised 25 November 2026

[Numbered figures](results/paper/figures.csv) · [Tables and values](results/paper/table_cells.csv) · [Movies](results/paper/movies.csv) · [Data and code for every result](coverage.csv) · [Numerical source index](results/result_source_index.csv) · [Versioned data downloads](https://github.com/Ehsan-Roohi/JFM-2026-1283-rarefied-bow-shocks/releases/tag/v1.0.0)

The study compares rarefied argon and nitrogen bow shocks over a circular cylinder with planar shock profiles and matched flat-plate calculations. The archive contains the 27 main and 11 supplementary figures, all 13 tables, all 12 supplementary films, the numerical fields and distributions used for the comparisons, and analysis and rendering code. Figure numbering follows the 25 November revision. Each figure component has its own filename and checksum.

## Find or reproduce a result

Use [coverage.csv](coverage.csv) to find the data group and code for a figure, table or movie. The [numerical source index](results/result_source_index.csv) resolves individual scientific inputs for the printed figures and tables. Small data files, model inputs and analysis code are in the repository. Large numerical arrays and MP4s are available as [release assets](https://github.com/Ehsan-Roohi/JFM-2026-1283-rarefied-bow-shocks/releases/tag/v1.0.0). A public download works without a GitHub account.

Clone or download this repository, then run these examples from its top directory. The restoration utility uses Python 3.11 or later. Particle decompression uses the standard library in Python 3.14 or the `zstandard` package on earlier versions.

```text
python tools/restore.py --list
python tools/restore.py ds2v figure_inputs pod
python tools/restore.py matched distributions
python tools/restore.py formation velocity
python tools/restore.py movies
```

Download only the groups required for the result. For example, the following command restores both 900-frame Ar particle arrays at Kn = 0.1 and the formation fields used for their profile strips.

```text
python tools/restore.py formation film_Ar_cylinder_Kn0p1_mol film_Ar_plate_Kn0p1_mol
```

Each encoded asset and extracted data file is checked against SHA-256. For particle files, the utility also verifies each decompressed byte range and the complete reassembled NPZ. The `--cache` and `--dest` options select the download and extraction locations. The complete numerical and film asset set is approximately 14.14 GB compressed. All eight reconstructed particle files occupy 16,949,323,216 bytes. [Release manifest](release_assets.json) · [Particle array manifest](particle_arrays.json)

## Experimental and numerical coverage

| Dataset | Contents |
| --- | --- |
| DS2V Knudsen sweep | 18 original gas/case input decks and 18 archived field exports. Nine Ar and eight N₂ fields passed the historical read and admission step. The original N₂ field at Kn = 0.075 was unreadable. The wide-domain SPARTA sensitivity has separate files and provenance. |
| DS2V Mach sweep | 20 Ar and 10 N₂ exported fields and postprocessed rays, plus a selection table. The original case-specific Mach input decks were unavailable in the source archives. Imposed quantities, measured upstream diagnostics and support masks retain their individual definitions. |
| Proper orthogonal decomposition | 67 Kn-sweep matrix files and 43 Mach-sweep matrix/ray files. Historical-only N₂ results and the nine-case hybrid sensitivity have distinct filenames. The banks include weighted finite-support masks, means, modes, coefficients and the training-only leave-one-out calculations. |
| Cylinder/plate comparison | All 20 Mach-10 case input decks, case parameters, physical surfaces, averaged fields, and 220 archived window dumps used by the profile reductions. The dumps in this group have the centreline bands or full coverage retained by the authors' figure preparation. |
| SPARTA controls | The two rays for each of seven mesh/time-step cases, their block diagnostics, planar configurations, matched statistical windows, and separate transport, internal-energy and chemical-species sensitivity results. The Mach-15 vibrational temperature maps are the no-chemistry calculation. |
| Molecular station data | Four independent gas/geometry runs, 2,000 native sampler files and ten compact analysis blocks per run, signed velocity distributions and fixed-station metadata. The cylinder uses the inward-radial velocity at 170°. The planar and flat-plate distributions use the streamwise velocity. The plate histograms come from their own steady run. |
| Movies and field diagnostics | Eight original particle-trajectory files, each with 900 frames. Sixteen standard and sixteen high-particle field arrays, 100 production windows per case, plus sixteen late-time velocity fields. The twelve final MP4 encodes can be downloaded together or streamed from the movie index. |

Full unbinned cell and particle dump series from every SPARTA timestep and replay checkpoints are outside this deposit. The published arrays preserve the sampling windows, cuts, coordinates and masks used in the reported comparisons. The case input decks specify the initial and physical models. Historical continuation decks need their matching initial state for a solver restart. The unmodified DS2V application and SPARTA runtime must be obtained independently. [SPARTA](https://sparta.github.io/)

## Analysis

Install the scientific Python packages listed in [requirements.txt](requirements.txt) to run the analysis. The movie renderers also need FFmpeg with libx264. The [movie and Figure S11 guide](code/publication_movies/README.md) gives commands for generating both the original particle sequences and their final typesetting. The frozen case-archive programs in `code/campaign` retain their original campaign layout. The portable profile exporter and the `code/fields`, `code/physics`, `code/pod` and `code/molecular` figure programs use the deposited datasets directly or accept their paths on the command line.

```text
python tools/restore.py figure_inputs matched distributions
python code/portable_profiles.py --out reproduced
python code/physics/flatplate_runs/fig_truncation_paper.py data/physics/flatplate_runs reproduced/figure15.pdf
python code/molecular/fig_vdf_planar_paper.py data/molecular/analysis_compact/planar_velocity_distributions.csv reproduced/figure17.pdf
python code/molecular/fig_three_shocks.py data/molecular data/molecular/strip_final reproduced
```

The figure scripts contain the original scalar normalizations, finite-support handling and window-selection rules. Rendering libraries and local fonts can affect the PDF or MP4 byte stream. The files in [results/paper](results/paper) and the direct-download movie ZIP preserve the published renderings.

## Reference sources

The experimental density profile is the authors' digitization of 28 plotted nitrogen measurements from Alsmeyer, as reproduced by Xu and Tang. The two-pixel reading envelope represents digitization uncertainty. The SWD comparison uses coordinates extracted for the authors' own DS2V case with the method of Akhlaghi, Daliri and Soltani. Third-party publications and solver software retain their respective rights. [Alsmeyer (1976)](https://doi.org/10.1017/S0022112076001912) · [Xu and Tang (2004)](https://doi.org/10.1063/1.1783372) · [Akhlaghi et al. (2017)](https://doi.org/10.2514/1.J055819)

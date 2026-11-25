# Reproducing the matched-body drag results

The surface tallies cover the Mach-10 cylinder and plate in argon and nitrogen at Kn = 0.01 and 0.1. Each case contributes ten committed 0.3 ms blocks. Five consecutive wall-tally windows make up each block.

From the root of the v1.1.0 checkout, restore the drag dataset and run the same Python reducer used for the manuscript.

```text
python tools/restore.py drag
python code/drag/analyze_surface.py --run-root data/drag/evidence --parent-root data/drag/evidence/parent --out reproduced/drag
```

The reducer verifies the geometry, inflow, sampling cadence, checkpoint lineage and force decomposition. It produces the signed per-window and per-block forces, the eight case statistics and the matched body comparisons. The force is per unit cylinder span. The drag coefficient uses the full projected height of 0.3048 m and the freestream dynamic pressure of the corresponding case. The reported standard deviations describe variation among its ten time blocks.

| File under `data/drag/results` | Contents |
| --- | --- |
| `drag_subwindows.csv` | Timestep limits, native wall file, SHA-256, force components and drag coefficients for every selected wall-tally interval. |
| `drag_blocks.csv` | Window-averaged forces and coefficients for each committed block. |
| `drag_summary.csv` | Case means, block standard deviations, pressure and shear contributions, normalization inputs and full-body drag per unit span. |
| `drag_shape_comparison.csv` | Paired cylinder and plate comparisons for each gas and Knudsen number. |

The independent momentum-balance calculation reconstructs the streamwise pressure and shear contributions from segment normals and tangents. The included `check_momentum.py` runs it against the recreated case summary.

```text
python code/drag/check_momentum.py --data-root data/drag --summary reproduced/drag/drag_summary.csv
python code/drag/validate_surface.py --run-root data/drag/evidence --parent-root data/drag/evidence/parent --results reproduced/drag
```

Table 6 rounds each full-precision coefficient independently to three decimals. The CSVs retain the unrounded pressure, shear and total values.

### Numerical provenance

The `committed_windows.csv` and `committed_blocks.csv` files identify the physical timestep, chosen attempt and original restart-write marker. Each native wall dump is archived byte-for-byte. The `parent` tree contains the original model, geometry and input decks. The continuation decks retain the original commands with repository-relative filenames for the admitted restart and collision table.

The public `log.case` excerpts retain the original final numerical statistics and closure marker. The public `log.drag` excerpts retain the original numerical admission, block-commit markers and completion record. The startup and pin records use case and attempt names for scheduler identifiers. Runtime evidence SHA-256 values and distributed continuation-deck checksums appear in `admissions.csv` and `provenance/native_sha256.json`. The numeric reduction uses the complete original surface dumps and requires no restart files.

The production driver `run_surface.py` reproduces the continuation-input construction and checkpoint-selection procedure. Running the original continuation requires the SPARTA executable, MPI and its admitted parent restart. The parent restart SHA-256 values identify these states.

"""Recalculate force contributions from geometry and momentum and check the case CSV."""
import argparse
import csv
import math
from pathlib import Path

from independent_momentum_balance import summarize

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/drag"))
    parser.add_argument("--summary", type=Path, default=Path("data/drag/results/drag_summary.csv"))
    args = parser.parse_args()
    rebuilt = {}
    for kn in ("Kn0p01", "Kn0p1"):
        rebuilt.update(summarize(args.data_root, kn))
    fields = {"Cd_mean":"mean_Cd", "Cd_pressure_mean":"normal_Cd", "Cd_shear_mean":"shear_Cd",
              "Cd_block_sd":"block_sd", "Cd_first5_mean":"first5_Cd", "Cd_last5_mean":"last5_Cd",
              "Fx_full_N_per_m_mean":"drag_N_per_m"}
    with args.summary.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8 or {row["case"] for row in rows} != set(rebuilt):
        raise ValueError("The matched-body case summary must include all eight cases.")
    for row in rows:
        case = row["case"]
        for column, independent in fields.items():
            if not math.isclose(float(row[column]), rebuilt[case][independent], rel_tol=1e-11, abs_tol=1e-12):
                raise ValueError(f"{case} differs in {column}")
        print(f"{case}  Cd {rebuilt[case]['mean_Cd']:.8g}  "
              f"pressure {rebuilt[case]['normal_Cd']:.8g}  shear {rebuilt[case]['shear_Cd']:.8g}")

if __name__ == "__main__":
    main()

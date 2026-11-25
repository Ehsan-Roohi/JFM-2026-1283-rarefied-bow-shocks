"""First inward density crossings on the archived 170-degree Kn-series cylinder rays.

Standard-library Python. Run alongside density_rays_170.csv to emit CSV on stdout.
Use --latex for a compact two-gas table of positions and width normalized by D.
"""
import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import sys

DIAMETER_MM = 304.8
FRACTIONS = (10, 50, 90)


def load_rays(path):
    rays = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["gas"], float(row["Kn"]))
            q = float(row["rho_over_rho_inf"]) if row["rho_over_rho_inf"] else None
            rays[key].append((1000 * float(row["s_m"]), q))
    return rays


def upstream_segment(samples):
    ordered = sorted(samples)
    while ordered and ordered[-1][1] is None:
        ordered.pop()
    last_gap = max((k for k, p in enumerate(ordered) if p[1] is None), default=-1)
    return ordered[last_gap+1:]


def roots(points, threshold):
    inward, outward = [], []
    for (s0, q0), (s1, q1) in zip(points, points[1:]):
        if (q0 >= threshold) == (q1 >= threshold):
            continue
        s = s0 + (threshold - q0) * (s1 - s0) / (q1 - q0)
        target = inward if q1 < threshold else outward
        target.append(s)
    return inward, outward


def crossing_row(gas, kn, samples):
    segment = upstream_segment(samples)
    gamma = 5/3 if gas == "Ar" else 7/5
    jump_ratio = (gamma+1)*100 / ((gamma-1)*100+2)
    result = dict(gas=gas, Kn=kn, finite_samples=len(segment))
    positions = {}
    for percent in FRACTIONS:
        threshold = 1 + percent/100 * (jump_ratio-1)
        inward, outward = roots(segment, threshold)
        valid = bool(inward and segment and segment[-1][1] < threshold)
        s_mm = inward[-1] if valid else None
        p = f"s{percent}"
        result[p+"_mm"] = s_mm
        result[p+"_over_D"] = s_mm/DIAMETER_MM if valid else None
        result[p+"_inward_count"] = len(inward)
        result[p+"_outward_count"] = len(outward)
        result[p+"_inward_roots_mm"] = "|".join(f"{s:.6f}" for s in reversed(inward))
        result[p+"_outward_roots_mm"] = "|".join(f"{s:.6f}" for s in reversed(outward))
        positions[percent] = s_mm
    complete = all(positions[p] is not None for p in FRACTIONS)
    result["ordered"] = complete and positions[10] > positions[50] > positions[90]
    width = positions[10] - positions[90] if result["ordered"] else None
    result["W_mm"] = width
    result["W_over_D"] = width/DIAMETER_MM if width is not None else None
    return result


def table(rows):
    by_case = {(r["gas"], r["Kn"]): r for r in rows}
    print(r"\begin{tabular}{c cccc cccc}")
    print(r"\hline")
    print(r"& \multicolumn{4}{c}{Ar} & \multicolumn{4}{c}{N$_2$}\\")
    print(r"$Kn_\infty$ & $s_{10}/D$ & $s_{50}/D$ & $s_{90}/D$ & $W_c/D$"
          r" & $s_{10}/D$ & $s_{50}/D$ & $s_{90}/D$ & $W_c/D$\\")
    print(r"\hline")
    for kn in sorted({r["Kn"] for r in rows}):
        cells = [f"{kn:g}"]
        for gas in ("Ar", "N2"):
            r = by_case.get((gas, kn))
            for key in ("s10_over_D", "s50_over_D", "s90_over_D", "W_over_D"):
                cells.append(f"{r[key]:.3f}" if r and r[key] is not None else r"---")
        print(" & ".join(cells) + r"\\")
    print(r"\hline")
    print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rays", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent/"density_rays_170.csv")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()
    rows = [crossing_row(gas, kn, points)
            for (gas, kn), points in sorted(load_rays(args.rays).items())]
    if args.latex:
        table(rows)
    else:
        writer = csv.DictWriter(sys.stdout, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Measure unsmoothed ideal-jump widths of the two archived Mach-10 planar mean profiles.

Run directly for the two planar results. Supply --cylinder to compare the nondimensional
upstream and downstream widths at every archived cylinder Kn with the rescaled planar mean.
"""
import argparse
import csv
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def plane_metrics(samples, gamma):
    ratio = (gamma + 1)*100/((gamma - 1)*100 + 2)
    results = {}
    for percent in (10, 50, 90):
        level = 1 + percent/100*(ratio - 1)
        roots = []
        count_back = 0
        for (xa, qa), (xb, qb) in zip(samples, samples[1:]):
            if qa < level <= qb:
                roots.append(xa + (level - qa)*(xb - xa)/(qb - qa))
            elif qb < level <= qa:
                count_back += 1
        if len(roots) != 1 or count_back or samples[0][1] >= level:
            raise ValueError(f"Ambiguous or unbracketed planar {percent}% location")
        results[f"x{percent}_mm"] = 1000*roots[0]
    results["upstream_10_50_mm"] = results["x50_mm"] - results["x10_mm"]
    results["downstream_50_90_mm"] = results["x90_mm"] - results["x50_mm"]
    results["Wp_mm"] = results["x90_mm"] - results["x10_mm"]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cylinder", action="store_true")
    parser.add_argument("--rays", type=Path, default=ROOT/"planar_density.csv")
    parser.add_argument("--cylinder-widths", type=Path,
                        default=ROOT.parent/"density_crossings_170.csv")
    args = parser.parse_args()
    all_samples = {}
    with args.rays.open(newline="", encoding="utf-8") as data:
        for row in csv.DictReader(data):
            all_samples.setdefault(row["gas"], []).append((float(row["x_m"]), float(row["rho_over_rho_inf"])))
    planes = {}
    for gas, samples in all_samples.items():
        planes[gas] = dict(gas=gas, nominal_Kn=0.01, **plane_metrics(samples, 5/3 if gas=="Ar" else 7/5))
    rows = list(planes.values())
    if args.cylinder:
        rows = []
        with args.cylinder_widths.open(newline="", encoding="utf-8") as data:
            for row in csv.DictReader(data):
                gas, kn = row["gas"], float(row["Kn"])
                plane, scale = planes[gas], kn/0.01
                up = float(row["s10_mm"]) - float(row["s50_mm"])
                down = float(row["s50_mm"]) - float(row["s90_mm"])
                length = float(row["W_mm"])
                up_plane = plane["upstream_10_50_mm"]*scale
                down_plane = plane["downstream_50_90_mm"]*scale
                wp = plane["Wp_mm"]*scale
                rows.append(dict(gas=gas, Kn=kn, Wc_mm=length, scaled_Wp_mm=wp, Wc_over_Wp=length/wp,
                                 cylinder_10_50_mm=up, scaled_planar_10_50_mm=up_plane, upstream_ratio=up/up_plane,
                                 cylinder_50_90_mm=down, scaled_planar_50_90_mm=down_plane, downstream_ratio=down/down_plane))
    writer = csv.DictWriter(sys.stdout, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()

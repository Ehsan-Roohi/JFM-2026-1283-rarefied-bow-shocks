"""Export the plotted arrays for figures 1, 9, 10, 24 and 25, with the final dimensionless coordinates."""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DIAMETER = 0.3048
MASS = {"Ar": 6.63e-26, "N2": 4.65e-26}
SPEED = {"Ar": 2634.1, "N2": 2883.3}
TEMPERATURE = 200.0


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(directory, name, columns):
    np.savetxt(directory / (name + ".csv"), np.column_stack(list(columns.values())),
               delimiter=",", header=",".join(columns), comments="", fmt="%.17g")


def validation(data, out):
    campaign = read_json(data / "evidence/figure_inputs/f427fe1b2ff9_CAMPAIGN.json")
    report = read_json(data / "evidence/figure_inputs/74b4d1e9c094_REPORT.json")
    base = data / "validation/experiment"
    experiment = np.genfromtxt(base / "Alsmeyer_N2_M153_digitized_v21.csv", delimiter=",", names=True)
    x = experiment["x_over_lambda"]
    rho = experiment["rho_star"]
    save(out, "figure_1_Alsmeyer_digitized", {"x_minus_half_density_over_lambda_mu": x - np.interp(0.5, rho, x), "rho_star": rho})
    for config in campaign["configs"]:
        name = config["name"]
        rho1 = config["nrho"] * campaign["models"]["baseline"]["mass"]
        jump = rho1 * (config["rankine_hugoniot"]["compression"] - 1)
        with np.load(base / (name + ".npz"), allow_pickle=False) as z:
            save(out, "figure_1_" + name, {
                "x_minus_half_density_over_lambda_mu": (z["x"] - report["cases"][name]["center_m"]) / config["transport"]["lambda_viscosity_m"],
                "rho_star": (z["mean"][:, 0] - rho1) / jump,
            })


def gradient_center(x, density, half_height):
    sigma = 0.0015 / np.median(np.diff(x))
    radius = max(1, int(np.ceil(4 * sigma)))
    k = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (k / sigma) ** 2)
    smoothed = np.convolve(np.pad(density, radius, mode="edge"), weights / weights.sum(), mode="valid")
    gradient = np.gradient(smoothed, x)
    allowed = np.flatnonzero(np.abs(x - half_height) < 0.025)
    return x[allowed[np.argmax(gradient[allowed])]]


def planar_and_cylinder(data, out, gas, number):
    name = gas + "_planar_long"
    report = read_json(data / "evidence/PLANAR_REPORT.json")["cases"][name]
    with np.load(data / "validation/planar" / (name + ".npz"), allow_pickle=False) as z:
        x, field = z["x"], z["mean"]
        center = gradient_center(x, field[:, 0], report["center_m"])
        columns = {"distance_downstream_over_D": (x - center) / DIAMETER,
                   "rho_over_rho_inf": field[:, 0] / (report["config"]["nrho"] * MASS[gas]),
                   "ux_over_U_inf": field[:, 1] / SPEED[gas], "Ttr_over_T_inf": field[:, 2] / TEMPERATURE}
        if gas == "N2":
            columns["Trot_over_T_inf"] = field[:, 3] / TEMPERATURE
        save(out, f"figure_{number}_{gas}_planar", columns)
    label = "C4_" + gas + "_L4"
    case = data / "controls" / label
    report = read_json(case / "REPORT.json")["cases"][label]
    marker = report["profiles"]["170"]["mean_profile_marker"]["standoff_m"]
    with np.load(case / (label + "_ray170.npz"), allow_pickle=False) as z:
        downstream = (marker - z["s"]) / DIAMETER
        order = np.argsort(downstream)
        field = z["mean"][order]
        columns = {"distance_downstream_over_D": downstream[order], "rho_over_rho_inf": field[:, 0],
                   "ux_over_U_inf": field[:, 2] / SPEED[gas], "Ttr_over_T_inf": field[:, 1] / TEMPERATURE}
        if gas == "N2":
            columns["Trot_over_T_inf"] = field[:, 3] / TEMPERATURE
        save(out, f"figure_{number}_{gas}_cylinder", columns)


def held_out(data, out, gas, number):
    with np.load(data / "evidence/HELDOUT_PREDICTION_PROFILES.npz", allow_pickle=False) as z:
        for method in ("true", "prediction"):
            columns = {"wall_distance_over_D": z["s_m"] / DIAMETER, "rho_over_rho_inf": z[f"{gas}_rho_{method}"],
                       "ux_over_U_inf": z[f"{gas}_u_{method}"], "Ttr_over_T_inf": z[f"{gas}_ttr_{method}"]}
            if gas == "N2":
                columns["Trot_over_T_inf"] = z[f"{gas}_trot_{method}"]
            save(out, f"figure_{number}_{gas}_{method}", columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--figures", nargs="+", type=int, choices=(1, 9, 10, 24, 25), default=(1, 9, 10, 24, 25))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    for number in args.figures:
        if number == 1:
            validation(args.data, args.out)
        elif number in (9, 10):
            planar_and_cylinder(args.data, args.out, "Ar" if number == 9 else "N2", number)
        else:
            held_out(args.data, args.out, "Ar" if number == 24 else "N2", number)


if __name__ == "__main__":
    main()

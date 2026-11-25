"""Test sampling sensitivity on the archived 170-degree density rays. No solver calls."""
import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def column(rows, key):
    return np.array([float(row[key]) for row in rows])


def levels(case):
    gamma = float(case["gamma"])
    mach = float(case["Mach"])
    jump = (gamma + 1) * mach**2 / ((gamma - 1) * mach**2 + 2)
    return 1 + np.array([0.10, 0.50, 0.90]) * (jump - 1)


def crossings(x, rho, thresholds):
    """First upward crossing walking from large s toward the wall."""
    roots = []
    for threshold in thresholds:
        brackets = np.flatnonzero((rho[:-1] >= threshold) & (rho[1:] < threshold))
        if brackets.size == 0:
            raise ValueError(f"Missing bracket for density {threshold}")
        i = brackets[-1]
        t = (threshold - rho[i]) / (rho[i + 1] - rho[i])
        roots.append(x[i] + t * (x[i + 1] - x[i]))
    s10, s50, s90 = roots
    return np.array([s10, s50, s90, s10 - s90])


def bin_operator(x, rho, pitch, phases, thresholds, reference=None):
    """Exact mean of the piecewise-linear ray in every full bin; no extrapolation."""
    dx = np.diff(x)
    if not np.all(dx > 0) or np.any(~np.isfinite(rho)):
        raise ValueError("Expected a finite, increasing profile.")
    area = np.r_[0.0, np.cumsum(0.5 * (rho[:-1] + rho[1:]) * dx)]
    slope = np.diff(rho) / dx

    def integral(t):
        idx = np.clip(np.searchsorted(x, t, side="right") - 1, 0, dx.size - 1)
        d = t - x[idx]
        return area[idx] + rho[idx] * d + 0.5 * slope[idx] * d**2

    boundaries = np.arange(np.floor(x[0] / pitch) - 1, np.ceil(x[-1] / pitch) + 2)
    filtered = []
    relative_rms = []
    if reference is not None:
        window = (x >= 25.0 - 1.e-8) & (x <= 150.0 + 1.e-8)
        target_x, target_rho = x[window], reference[window]
        jump = (thresholds[2] - thresholds[0]) / 0.8
    for phase in phases:
        centres = pitch * (boundaries + phase)
        left, right = centres - pitch / 2, centres + pitch / 2
        keep = (left >= x[0]) & (right <= x[-1])
        centres = centres[keep]
        mean = (integral(right[keep]) - integral(left[keep])) / pitch
        filtered.append(crossings(centres, mean, thresholds))
        if reference is not None:
            if target_x[0] < centres[0] or target_x[-1] > centres[-1]:
                raise ValueError("The fixed profile comparison window extends beyond the bin centres")
            reconstructed = np.interp(target_x, centres, mean)
            relative_rms.append(100 * np.sqrt(np.mean((reconstructed - target_rho)**2)) / jump)
    return np.array(filtered), np.array(relative_rms)


def keep_support(x, rho):
    valid = np.isfinite(rho) & (x >= 15.0 - 1.e-8) & (x <= 160.0 + 1.e-8)
    idx = np.flatnonzero(valid)
    continuous = idx.size >= 2 and np.all(np.diff(idx) == 1)
    regular_pitch = np.allclose(np.diff(x[idx]), np.median(np.diff(x)), rtol=1.e-6, atol=1.e-9)
    if not continuous or not regular_pitch:
        raise ValueError("Unexpected gap in the 15-160 mm support")
    return x[idx], rho[idx]


def load_profiles(data, gas, kn, all_rows):
    if kn == 0.01:
        rows = records(data / f"paired_solver_profile_{gas}_Kn001.csv")
        x = column(rows, "s_m") * 1000
        x, sparta = keep_support(x, column(rows, "SPARTA_rho"))
        paired_x, ds2v = keep_support(column(rows, "s_m") * 1000, column(rows, "DS2V_rho"))
        if not np.array_equal(paired_x, x):
            raise ValueError("Paired ray support mismatch")
        return x, sparta, ds2v
    rows = [r for r in all_rows if r["gas"] == gas and float(r["Kn"]) == kn]
    x = column(rows, "s_m") * 1000
    rho = np.array([float(r["rho_over_rho_inf"] or "nan") for r in rows])
    x, ds2v = keep_support(x, rho)
    return x, ds2v, ds2v


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(data, output, nphase):
    data = data.resolve()
    if nphase < 2 or nphase % 2 != 0:
        raise ValueError("Use a positive even phase count of at least 2")
    phases = np.arange(nphase, dtype=float) / nphase
    cases = {(r["gas"], float(r["Kn"])): r for r in records(data / "density_ray_cases.csv")}
    planar = {r["gas"]: float(r["Wp_mm"]) for r in records(data / "planar/planar_widths.csv")}
    all_rows = records(data / "density_rays_170.csv")
    summary = []
    for gas in ["Ar", "N2"]:
        for kn in [0.01, 0.025]:
            case = cases[gas, kn]
            x, ray, ds2v = load_profiles(data, gas, kn, all_rows)
            thresholds = levels(case)
            raw = crossings(x, ray, thresholds)
            paired = crossings(x, ds2v, thresholds)
            plane_width = planar[gas] * kn / 0.01
            hx = 1000 * float(case["x_spacing_proxy_m"])
            ray_pitch = hx / abs(np.cos(np.deg2rad(float(case["angle_deg"]))))
            window = (x >= 25.0 - 1.e-8) & (x <= 150.0 + 1.e-8)
            jump = (thresholds[2] - thresholds[0]) / 0.8
            raw_rms = 100 * np.sqrt(np.mean((ray[window] - ds2v[window])**2)) / jump
            for mode, h in [("x_pitch_proxy", hx), ("projected_x_pitch_proxy", ray_pitch)]:
                vals, rms = bin_operator(x, ray, h, phases, thresholds, ds2v)
                shift = vals[:, 1] - paired[1]
                delta = 100 * (vals[:, 3] / raw[3] - 1)
                for what, agg, rms_stat in [
                    ("raw_input", raw, raw_rms), ("raw_DS2V", paired, 0.0),
                    ("phase_0_wall_centre", vals[0], rms[0]),
                    ("phase_half_wall_edge", vals[nphase // 2], rms[nphase // 2]),
                    ("phase_mean", vals.mean(axis=0), rms.mean()),
                    ("phase_min", vals.min(axis=0), rms.min()),
                    ("phase_max", vals.max(axis=0), rms.max())
                ]:
                    s10, s50, s90, width = agg
                    row = dict(gas=gas, Kn=kn, input="SPARTA" if kn == 0.01 else "DS2V",
                               pitch_case=mode, pitch_mm=h, statistic=what, nphase=nphase,
                               s10_mm=s10, s50_mm=s50, s90_mm=s90, W_mm=width,
                               W_over_W_plane=width / plane_width, delta_W_from_input_pct=100 * (width / raw[3] - 1),
                               delta_s50_from_DS2V_mm=s50 - paired[1],
                               rho_RMS_vs_DS2V_pct_RH_jump_25_150mm=rms_stat)
                    summary.append({k: f"{v:.10g}" if isinstance(v, (float, np.floating)) else v
                                    for k, v in row.items()})
                print(f"{gas} Kn={kn:g} {mode} h={h:.4f} mm: W input {raw[3]:.4f}, DS2V {paired[3]:.4f},"
                      f" phase mean {vals[:,3].mean():.4f}, min..max {vals[:,3].min():.4f}..{vals[:,3].max():.4f},"
                      f" extra broadening {delta.mean():.3f}% ({delta.min():.3f}..{delta.max():.3f});"
                      f" s50 input {raw[1]:.4f}, DS2V {paired[1]:.4f}, mean filtered {vals[:,1].mean():.4f},"
                      f" shift vs DS2V {shift.min():.4f}..{shift.max():.4f} mm;"
                      f" RMS {raw_rms:.3f}% -> {rms.min():.3f}..{rms.max():.3f}% of jump")
    output.mkdir(exist_ok=True, parents=True)
    write_csv(output / "sampling_operator_summary.csv", summary)
    input_names = ["density_ray_cases.csv", "density_rays_170.csv",
                   "paired_solver_profile_Ar_Kn001.csv", "paired_solver_profile_N2_Kn001.csv",
                   "planar/planar_widths.csv"]
    fingerprints = [{"file": name, "sha256": hashlib.sha256((data/name).read_bytes()).hexdigest()}
                    for name in input_names]
    write_csv(output / "sampling_operator_inputs.csv", fingerprints)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="packaged reproducibility directory")
    p.add_argument("--out", type=Path, required=True, help="directory for the two CSV files")
    p.add_argument("--phases", type=int, default=2048)
    args = p.parse_args()
    run(args.data, args.out, args.phases)
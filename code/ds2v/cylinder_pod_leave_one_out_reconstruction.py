#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leave-one-out POD reconstruction test for the JFM rarefied-cylinder paper.

Purpose
-------
This script tests whether the POD compactness claim is useful for reconstruction
rather than only a spectrum plot.

For each variable q in the selected sweep:
  1. Leave one snapshot out.
  2. Build the POD basis from the remaining snapshots only.
  3. Project the left-out snapshot onto the first r modes.
  4. Compute relative L2 reconstruction error on the common valid support.
  5. Repeat for all snapshots and r = 1...max_modes.

This is the best low-cost check for a modal-analysis reviewer:
density should reconstruct with fewer modes than Mach and thermal/internal
variables if the selective-compactness interpretation is real.

The script reuses field extraction and registration routines from:
    cylinder_modal_sensitivity_analysis_v2_markers.py
Pass it through --base_script or put it in the same folder.

Outputs
-------
LeaveOneOut_POD_reconstruction_errors.csv
LeaveOneOut_POD_reconstruction_summary.csv
LeaveOneOut_POD_reconstruction_table.tex
LeaveOneOut_POD_reconstruction_errors.pdf
LeaveOneOut_POD_reconstruction_errors.png
LeaveOneOut_POD_reconstruction_summary_text.tex

Recommended manuscript use
--------------------------
Put the figure and/or table in Appendix, not the main text, unless you want to
make reduced-reconstruction a central result.

Example
-------
python cylinder_pod_leave_one_out_reconstruction.py \
  --base_script cylinder_modal_sensitivity_analysis_v2_markers.py \
  --inputs "N2-Bird Prop-Kn-Part1.rar" "N2-Bird Prop-Kn-Part2.rar" \
  --out LOO_POD_N2_Kn \
  --extract_root _loo_pod_extracted \
  --radius 0.1524 \
  --leftmost_x 0.0 \
  --center_y 0.0 \
  --default_kn_if_missing 0.01 \
  --default_mach_if_missing 10.0 \
  --kn_sweep_mach 10.0 \
  --kn_values 0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1.0 \
  --gas N2 \
  --variables rho mach p ttr trot \
  --max_modes 5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Dynamic import of existing parser/registration script
# ---------------------------------------------------------------------

def import_base_module(path: str):
    candidates = [Path(path)]
    if not Path(path).exists():
        candidates += [
            Path.cwd() / path,
            Path(__file__).resolve().parent / path,
            Path.cwd() / "cylinder_modal_sensitivity_analysis_v2_markers.py",
            Path(__file__).resolve().parent / "cylinder_modal_sensitivity_analysis_v2_markers.py",
        ]

    base_path = None
    for p in candidates:
        if p.exists():
            base_path = p
            break

    if base_path is None:
        raise FileNotFoundError(
            "Could not find base script. Put "
            "cylinder_modal_sensitivity_analysis_v2_markers.py in the same folder "
            "or pass --base_script /path/to/cylinder_modal_sensitivity_analysis_v2_markers.py"
        )

    spec = importlib.util.spec_from_file_location("cylinder_base", str(base_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import base script from {base_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cylinder_base"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------
# POD / reconstruction utilities
# ---------------------------------------------------------------------

LABELS = {
    "rho": r"$\rho$",
    "mach": r"$M$",
    "p": r"$p$",
    "ttr": r"$T_{tr}$",
    "trot": r"$T_{rot}$",
    "u": r"$U$",
    "v": r"$V$",
}


def common_valid_mask(cube: np.ndarray, valid_fraction: float) -> np.ndarray:
    frac = np.isfinite(cube).mean(axis=0)
    valid = frac >= valid_fraction
    if valid.sum() < 20:
        valid = frac >= 0.50
    return valid


def flatten_on_mask(cube: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows = []
    for k in range(cube.shape[0]):
        vals = cube[k][valid].astype(float)
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        rows.append(vals)
    return np.asarray(rows, dtype=float)


def relative_l2(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    num = np.linalg.norm(x_true - x_pred)
    den = np.linalg.norm(x_true)
    if den <= 1e-14:
        den = np.sqrt(len(x_true))
    return float(num / den)


def train_pod_basis(X_train: np.ndarray, max_modes: int):
    """
    X_train: snapshots x points
    Returns mean, modes, energies.
    """
    mean = X_train.mean(axis=0)
    Xc = X_train - mean[None, :]
    if np.linalg.norm(Xc) <= 1e-14:
        return mean, np.zeros((0, X_train.shape[1])), np.array([])

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    energies = S**2
    if energies.sum() > 0:
        energies = energies / energies.sum()
    else:
        energies = np.zeros_like(S)

    nm = min(max_modes, Vt.shape[0])
    modes = Vt[:nm].copy()

    # Deterministic sign convention
    for i in range(modes.shape[0]):
        j = np.argmax(np.abs(modes[i]))
        if modes[i, j] < 0:
            modes[i] *= -1

    return mean, modes, energies


def loo_reconstruction_for_variable(
    cube: np.ndarray,
    meta: pd.DataFrame,
    variable: str,
    valid_fraction: float = 0.70,
    max_modes: int = 5,
) -> pd.DataFrame:
    """
    Leave-one-out POD reconstruction for one variable.
    """
    cube = np.asarray(cube, dtype=float)
    ns = cube.shape[0]
    if ns < 3:
        return pd.DataFrame()

    valid = common_valid_mask(cube, valid_fraction)
    X = flatten_on_mask(cube, valid)

    rows = []
    max_train_modes = min(max_modes, ns - 2)

    for i in range(ns):
        train_idx = [j for j in range(ns) if j != i]
        X_train = X[train_idx]
        x_test = X[i]

        mean, modes, train_energy = train_pod_basis(X_train, max_train_modes)
        x_centered = x_test - mean

        for r in range(1, max_train_modes + 1):
            if modes.shape[0] < r:
                err = np.nan
                cum_energy = np.nan
            else:
                Phi = modes[:r]
                coeff = x_centered @ Phi.T
                x_rec = mean + coeff @ Phi
                err = relative_l2(x_test, x_rec)
                cum_energy = float(np.sum(train_energy[:r])) if len(train_energy) >= r else np.nan

            row = {
                "variable": variable,
                "left_out_index": i,
                "n_train": len(train_idx),
                "n_points": int(valid.sum()),
                "modes": r,
                "relative_L2_error": err,
                "training_Cr": cum_energy,
            }
            # Attach metadata for the left-out snapshot
            for col in meta.columns:
                val = meta.iloc[i][col]
                if isinstance(val, (np.floating, float, int, np.integer, str)):
                    row[col] = val
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_errors(errdf: pd.DataFrame) -> pd.DataFrame:
    if errdf.empty:
        return pd.DataFrame()

    rows = []
    for (var, r), sub in errdf.groupby(["variable", "modes"]):
        vals = sub["relative_L2_error"].to_numpy(float)
        rows.append({
            "variable": var,
            "modes": int(r),
            "mean_error": float(np.nanmean(vals)),
            "median_error": float(np.nanmedian(vals)),
            "p10_error": float(np.nanpercentile(vals, 10)),
            "p90_error": float(np.nanpercentile(vals, 90)),
            "max_error": float(np.nanmax(vals)),
            "min_error": float(np.nanmin(vals)),
            "n_left_out": int(np.isfinite(vals).sum()),
            "mean_training_Cr": float(np.nanmean(sub["training_Cr"].to_numpy(float))),
        })
    return pd.DataFrame(rows).sort_values(["variable", "modes"])


def plot_loo_errors(summary: pd.DataFrame, outdir: Path, title: str):
    if summary.empty:
        return

    variables = list(dict.fromkeys(summary["variable"].tolist()))
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=280)

    for var in variables:
        sub = summary[summary["variable"] == var].sort_values("modes")
        x = sub["modes"].to_numpy(int)
        y = sub["median_error"].to_numpy(float)
        ylo = sub["p10_error"].to_numpy(float)
        yhi = sub["p90_error"].to_numpy(float)

        ax.plot(x, y, marker="o", lw=1.6, label=LABELS.get(var, var))
        ax.fill_between(x, ylo, yhi, alpha=0.16, lw=0)

    ax.set_xlabel("Number of retained POD modes")
    ax.set_ylabel("Leave-one-out relative reconstruction error")
    ax.set_yscale("log")
    ax.set_xticks(sorted(summary["modes"].unique()))
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "LeaveOneOut_POD_reconstruction_errors.pdf", bbox_inches="tight")
    fig.savefig(outdir / "LeaveOneOut_POD_reconstruction_errors.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def fmt(x, nd=3):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{nd}g}"


def write_latex_table(summary: pd.DataFrame, outpath: Path, variables_to_show: Optional[List[str]] = None):
    if summary.empty:
        outpath.write_text("% No leave-one-out reconstruction data available.\n", encoding="utf-8")
        return

    if variables_to_show is None:
        variables_to_show = ["rho", "mach", "p", "ttr", "trot"]

    # Use modes 1, 2, 3 if available
    mode_list = [m for m in [1, 2, 3] if m in set(summary["modes"].astype(int))]
    if not mode_list:
        mode_list = sorted(summary["modes"].unique())[:3]

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Leave-one-out POD reconstruction errors for the nitrogen Knudsen-number sweep. For each left-out snapshot, the POD basis is formed from the remaining snapshots only, and the excluded snapshot is reconstructed with the first \(r\) modes. Values are median relative \(L_2\) errors over the left-out cases, with the 10--90 percentile range in parentheses.}",
        r"\label{tab:loo_pod_reconstruction}",
        r"\begin{tabular}{l" + "c" * len(mode_list) + r"}",
        r"\toprule",
        "Variable & " + " & ".join([rf"\(r={m}\)" for m in mode_list]) + r" \\",
        r"\midrule",
    ]

    for var in variables_to_show:
        sub = summary[summary["variable"] == var]
        if sub.empty:
            continue
        vals = []
        for m in mode_list:
            row = sub[sub["modes"] == m]
            if row.empty:
                vals.append("--")
            else:
                rr = row.iloc[0]
                vals.append(
                    rf"{fmt(rr['median_error'])} "
                    rf"({fmt(rr['p10_error'])}--{fmt(rr['p90_error'])})"
                )
        lines.append(f"{LABELS.get(var,var)} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    outpath.write_text("\n".join(lines), encoding="utf-8")


def write_summary_text(outpath: Path):
    text = r"""
% Suggested Appendix/Main-text wording:
The leave-one-out reconstruction test in table~\ref{tab:loo_pod_reconstruction}
uses each snapshot in turn as an excluded case and constructs the POD basis from
the remaining snapshots only. The errors therefore test whether the observed
compactness is useful for reconstructing unseen parameter states within the
sampled sweep, rather than only describing the singular-value spectrum of the
full data matrix. The expected outcome is selective compactness: density should
be reconstructed accurately with fewer modes, whereas Mach number and thermal
variables should require additional modes.
"""
    outpath.write_text(text.strip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_script", default="cylinder_modal_sensitivity_analysis_v2_markers.py",
                    help="Path to cylinder_modal_sensitivity_analysis_v2_markers.py.")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="RAR/ZIP archives, extracted folders, or folders containing DS2V/Tecplot files.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract_root", default="_loo_pod_extracted")
    ap.add_argument("--radius", type=float, default=0.1524)
    ap.add_argument("--leftmost_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)
    ap.add_argument("--default_kn_if_missing", type=float, default=0.01)
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0)
    ap.add_argument("--kn_sweep_mach", type=float, default=10.0)
    ap.add_argument("--kn_values", nargs="+", type=float,
                    default=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--gas", default="N2", choices=["N2", "Ar"])
    ap.add_argument("--variables", nargs="+", default=["rho", "mach", "p", "ttr", "trot"])
    ap.add_argument("--theta_min", type=float, default=100.0)
    ap.add_argument("--theta_max", type=float, default=180.0)
    ap.add_argument("--n_theta", type=int, default=65)
    ap.add_argument("--xi_min", type=float, default=-1.0)
    ap.add_argument("--xi_max", type=float, default=4.0)
    ap.add_argument("--n_xi", type=int, default=260)
    ap.add_argument("--ray_ns", type=int, default=900)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)
    ap.add_argument("--valid_fraction", type=float, default=0.70)
    ap.add_argument("--marker_method", default="maxgrad",
                    choices=["maxgrad", "halfjump", "gcentroid"])
    ap.add_argument("--max_modes", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    base = import_base_module(args.base_script)

    roots = base.prepare_inputs(args.inputs, base.ensure_dir(args.extract_root))
    cases = base.find_flow_files(
        roots,
        default_kn=args.default_kn_if_missing,
        default_mach=args.default_mach_if_missing,
    )
    inv = base.build_inventory(cases)
    if inv.empty:
        raise RuntimeError("No valid DSMC/Tecplot cases found.")

    inv.to_csv(outdir / "LeaveOneOut_case_inventory.csv", index=False)

    selected = base.select_kn_sweep(inv, args.gas, args.kn_sweep_mach, args.kn_values)
    if selected.empty:
        raise RuntimeError(f"No {args.gas} Kn-sweep cases selected.")

    selected = selected.sort_values("parameter").reset_index(drop=True)
    selected.to_csv(outdir / "LeaveOneOut_selected_cases.csv", index=False)

    cx = args.leftmost_x + args.radius
    cy = args.center_y
    R = args.radius

    xi_grid, theta_grid, snapshots, meta = base.build_attached_field_family(
        selected, cx, cy, R,
        variables=args.variables,
        theta_min_deg=args.theta_min,
        theta_max_deg=args.theta_max,
        n_theta=args.n_theta,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        n_xi=args.n_xi,
        ray_ns=args.ray_ns,
        smooth_sigma=args.smooth_sigma,
        marker_method=args.marker_method,
    )
    meta.to_csv(outdir / "LeaveOneOut_snapshot_meta.csv", index=False)

    all_errors = []
    for var in args.variables:
        if var not in snapshots:
            print(f"Skipping {var}: no snapshots.")
            continue
        cube = snapshots[var]
        if cube.shape[0] < 3:
            print(f"Skipping {var}: fewer than three snapshots.")
            continue
        err = loo_reconstruction_for_variable(
            cube, meta, var,
            valid_fraction=args.valid_fraction,
            max_modes=args.max_modes
        )
        if not err.empty:
            all_errors.append(err)

    if not all_errors:
        raise RuntimeError("No leave-one-out reconstruction errors were computed.")

    errdf = pd.concat(all_errors, ignore_index=True)
    summary = summarize_errors(errdf)

    errdf.to_csv(outdir / "LeaveOneOut_POD_reconstruction_errors.csv", index=False)
    summary.to_csv(outdir / "LeaveOneOut_POD_reconstruction_summary.csv", index=False)

    title = f"Leave-one-out POD reconstruction: {args.gas} Kn sweep"
    plot_loo_errors(summary, outdir, title)
    write_latex_table(summary, outdir / "LeaveOneOut_POD_reconstruction_table.tex", args.variables)
    write_summary_text(outdir / "LeaveOneOut_POD_reconstruction_summary_text.tex")

    print("\nDone. Outputs written to:", outdir)
    print("Key files:")
    print("  LeaveOneOut_POD_reconstruction_errors.csv")
    print("  LeaveOneOut_POD_reconstruction_summary.csv")
    print("  LeaveOneOut_POD_reconstruction_table.tex")
    print("  LeaveOneOut_POD_reconstruction_errors.pdf/.png")
    print("  LeaveOneOut_POD_reconstruction_summary_text.tex")


if __name__ == "__main__":
    main()

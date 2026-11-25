#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cylinder_step2c_shocksector_profile_collapse_v4_rho_registered.py

Robust Step-2c profile-collapse code for rarefied hypersonic cylinder DSMC data.

What this version fixes
-----------------------
The previous stagnation-line horizontal cut could mix the bow-shock compression
with the near-wall compression/thermal layer.  That is why density profiles
could overshoot and fail to collapse.  This version returns to the physically
cleaner construction:

  * sample body-normal rays in the upstream shock sector, away from the wall
    boundary layer;
  * identify the local maximum-gradient shock/layer centre from the density
    profile on each ray;
  * register every variable using the density centre and density thickness,
    xi_rho=(s-s_rho)/delta_rho;
  * normalize each variable with its own local upstream/downstream values around
    the density-centred shock layer;
  * plot both physical-coordinate profiles and density-attached profiles.

This gives a fair test of whether variables other than density collapse when
the shock/compression layer is aligned using the density field.  It also avoids
the unstable case-by-case variable-specific registration that can make the
low-Kn cases look artificially anomalous.

Outputs
-------
CSV:
  Step2c_case_inventory.csv
  Step2c_selected_cases.csv
  Step2c_profile_metrics.csv

Figures:
  Step2c_<gas>_<sweep>_rho_registered_profile_collapse.pdf/png
  Step2c_<gas>_<sweep>_<var>_rho_registered_profiles.pdf/png

Recommended first figure to inspect:
  Step2c_N2_Kn_rho_registered_profile_collapse.pdf

Typical use:
  --profile_angles 135 145 155 165

Author: ChatGPT for Ehsan Roohi
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import interp1d
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "figure.titlesize": 18,
    "savefig.dpi": 350,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

LABELS = {
    "rho": r"$\rho$",
    "mach": r"$M$",
    "p": r"$p$",
    "ttr": r"$T_{tr}$",
    "trot": r"$T_{rot}$",
    "tvib": r"$T_{vib}$",
}

PRETTY = {
    "rho": "rho",
    "mach": "Mach",
    "p": "p",
    "ttr": "Ttr",
    "trot": "Trot",
    "tvib": "Tvib",
}


# ============================================================
# Utilities and extraction
# ============================================================

def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except Exception:
        return False


def extract_one(src, dst):
    src = Path(src)
    dst = ensure_dir(dst)

    if src.suffix.lower() == ".rar":
        cmds = []
        if shutil.which("unrar"):
            cmds.append(["unrar", "x", "-o+", str(src), str(dst)])
        if shutil.which("7z"):
            cmds.append(["7z", "x", "-y", f"-o{dst}", str(src)])
        if shutil.which("unar"):
            cmds.append(["unar", "-f", "-o", str(dst), str(src)])
        if shutil.which("bsdtar"):
            cmds.append(["bsdtar", "-xf", str(src), "-C", str(dst)])
        for c in cmds:
            if run_cmd(c):
                return
        raise RuntimeError(f"Could not extract RAR archive: {src}")

    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
        return

    if run_cmd(["bsdtar", "-xf", str(src), "-C", str(dst)]):
        return

    raise RuntimeError(f"Could not extract archive: {src}")


def prepare_inputs(inputs, extract_root):
    roots = []
    extract_root = ensure_dir(extract_root)

    for inp in inputs:
        p = Path(inp)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {p}")

        if p.is_dir():
            roots.append(p)
        else:
            target = extract_root / p.stem.replace(" ", "_")
            target.mkdir(parents=True, exist_ok=True)
            extract_one(p, target)
            roots.append(target)

    return roots


def normalize_key(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ============================================================
# Tecplot reader
# ============================================================

def parse_zone_points(header: str) -> Optional[int]:
    for key in ["N", "Nodes"]:
        m = re.search(rf"\b{key}\s*=\s*(\d+)", header, re.I)
        if m:
            return int(m.group(1))

    im = re.search(r"\bI\s*=\s*(\d+)", header, re.I)
    jm = re.search(r"\bJ\s*=\s*(\d+)", header, re.I)
    km = re.search(r"\bK\s*=\s*(\d+)", header, re.I)

    if im:
        I = int(im.group(1))
        J = int(jm.group(1)) if jm else 1
        K = int(km.group(1)) if km else 1
        return I * J * K

    return None


def read_tecplot_any(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    text = raw.decode("latin1", errors="ignore")
    lines = text.splitlines()

    default_vars = [
        "X", "Y", "ND", "D", "U", "V", "W",
        "TTR", "TRT", "TVB", "TOV", "MA", "MC", "MCT",
        "MFP", "SOF", "FSP", "ANG", "P", "Angular momentum/(Mass.L)"
    ]

    var_idx = None
    zone_idx = None

    for i, line in enumerate(lines):
        s = line.strip().upper()
        if "VARIABLES" in s and var_idx is None:
            var_idx = i
        if "ZONE" in s and zone_idx is None:
            zone_idx = i
            break

    if var_idx is not None:
        var_text = []
        for k in range(var_idx, len(lines)):
            s = lines[k].strip()
            us = s.upper()
            if k > var_idx and (us.startswith("ZONE") or re.match(r"^[\s\+\-]?(?:\d|\.\d)", s)):
                break
            var_text.append(s)
        variables = re.findall(r'"([^"]+)"', " ".join(var_text))
        if not variables:
            variables = default_vars
    else:
        variables = default_vars

    nvar = len(variables)

    if zone_idx is not None:
        header_lines = []
        data_start = zone_idx + 1

        for k in range(zone_idx, len(lines)):
            s = lines[k].strip()
            if k > zone_idx and re.match(r"^[\s\+\-]?(?:\d|\.\d)", s):
                data_start = k
                break
            header_lines.append(s)
            if k > zone_idx + 80:
                data_start = k + 1
                break

        npts = parse_zone_points(" ".join(header_lines))
        data_text = "\n".join(lines[data_start:])
    else:
        npts = None
        data_lines = []
        for line in lines:
            s = line.strip()
            us = s.upper()
            if "VARIABLES" in us or us.startswith("TITLE") or us.startswith("ZONE"):
                continue
            if re.fullmatch(r'"[^"]+"\s*,?', s):
                continue
            data_lines.append(line)
        data_text = "\n".join(data_lines)

    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", data_text)
    if len(nums) < nvar:
        raise ValueError(f"{path}: no numeric data; len(nums)={len(nums)}, nvar={nvar}")

    if npts is None:
        npts = len(nums) // nvar

    need = npts * nvar
    if len(nums) < need:
        npts = len(nums) // nvar
        need = npts * nvar

    arr = np.asarray(nums[:need], dtype=float).reshape(npts, nvar)
    return pd.DataFrame(arr, columns=variables), variables


def variable_map(columns: List[str]) -> Dict[str, Optional[str]]:
    norm = {normalize_key(c): c for c in columns}

    def choose(cands, required=False):
        for c in cands:
            key = normalize_key(c)
            if key in norm:
                return norm[key]

        for key, orig in norm.items():
            for c in cands:
                cc = normalize_key(c)
                if key == cc or key.startswith(cc) or cc in key:
                    return orig

        if required:
            raise KeyError(f"Could not find {cands} in columns: {columns}")
        return None

    return {
        "x":    choose(["X", "x"], required=True),
        "y":    choose(["Y", "y"], required=True),
        "rho":  choose(["D", "Density", "Rho", "rho", "ND", "numberdensity"], required=True),
        "u":    choose(["U", "u"]),
        "v":    choose(["V", "v"]),
        "w":    choose(["W", "w"]),
        "ttr":  choose(["TTR", "Ttr", "Ttrans", "TranslationalTemperature", "T", "TOV"]),
        "trot": choose(["TRT", "Trot", "RotationalTemperature"]),
        "tvib": choose(["TVB", "Tvib", "VibrationalTemperature"]),
        "mach": choose(["MA", "Mach", "M"]),
        "p":    choose(["P", "Pressure"]),
        "mfp":  choose(["MFP", "MeanFreePath"]),
    }


# ============================================================
# Metadata inference
# ============================================================

def infer_gas_from_path(path: Path) -> str:
    s = str(path).lower()
    if re.search(r"\bn2\b|nitrogen", s):
        return "N2"
    if re.search(r"\bar\b|argon", s):
        return "Ar"
    return "Unknown"


def infer_kn_from_path(path: Path) -> Optional[float]:
    s = " / ".join(path.parts)
    m = re.search(r"Kn\s*=\s*([0-9]+(?:[.,][0-9]+)?)", s, re.I)
    if not m:
        m = re.search(r"Kn\s*[_-]?\s*([0-9]+(?:[.,][0-9]+)?)", s, re.I)
    return float(m.group(1).replace(",", ".")) if m else None


def infer_mach_from_path(path: Path) -> Optional[float]:
    def clean(raw):
        raw = str(raw).replace(",", ".")
        if "." in raw:
            val = float(raw)
            return val if 1 <= val <= 50 else np.nan
        if re.match(r"^[0-9]+$", raw):
            intval = int(raw)
            if 1 <= intval <= 50:
                return float(intval)
            if len(raw) >= 2:
                val = float(raw[0] + "." + raw[1:])
                return val if 1 <= val <= 50 else np.nan
        return np.nan

    fname = path.name

    for pat in [
        r"(?:TriangulateGrid|TriangulatedGrid)\s*M\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])M\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])Ma\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])Mach\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, fname, re.I)
        if m:
            val = clean(m.group(1))
            if np.isfinite(val):
                return val

    for comp in list(path.parts)[::-1]:
        if re.search(r"Mach\s*[0-9]+(?:\.[0-9]+)?\s*-\s*[0-9]+", comp, re.I):
            continue
        for pat in [
            r"(?:^|[^A-Za-z])M\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])Ma\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])Mach\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        ]:
            m = re.search(pat, comp, re.I)
            if m:
                val = clean(m.group(1))
                if np.isfinite(val):
                    return val

    return None


def source_tag(path: Path) -> str:
    s = str(path).lower()
    if "ma=15-n2" in s or "ma15" in s:
        return "mach_sweep_n2_fixedM15"
    if "ma=9-n2" in s or "ma9" in s:
        return "mach_sweep_n2_fixedM9"
    if "argon-data-mach" in s or "triangulategrid" in s or "triangulatedgrid" in s:
        return "mach_sweep_argon"
    if "results-n2" in s:
        return "mach_sweep_n2"
    if "prop-kn" in s or "kn=" in s:
        return "kn_sweep"
    return "unknown"


def find_flow_files(roots, default_kn=0.01, default_mach=10.0):
    pats = [
        "**/DS2FF.DAT", "**/DS2FF.dat", "**/*DS2FF*.DAT", "**/*DS2FF*.dat",
        "**/TriangulateGrid*.dat", "**/TriangulateGrid*.DAT",
        "**/TriangulatedGrid*.dat", "**/TriangulatedGrid*.DAT",
    ]

    cases = []
    for root in roots:
        for pat in pats:
            for f in root.glob(pat):
                if "DS2SU" in f.name.upper():
                    continue

                gas = infer_gas_from_path(f)
                if gas == "Unknown":
                    gas = infer_gas_from_path(root)

                kn = infer_kn_from_path(f)
                if kn is None:
                    kn = infer_kn_from_path(root)
                if kn is None:
                    kn = default_kn

                mach = infer_mach_from_path(f)
                if mach is None or not np.isfinite(mach):
                    mach = infer_mach_from_path(root)
                if mach is None or not np.isfinite(mach):
                    mach = default_mach

                cases.append({
                    "path": f,
                    "gas": gas,
                    "kn": float(kn),
                    "mach": float(mach),
                    "source": source_tag(f),
                })

    seen, out = set(), []
    for c in cases:
        if c["path"] in seen:
            continue
        seen.add(c["path"])
        out.append(c)

    return out


def build_inventory(cases):
    rows = []

    for c in cases:
        try:
            df, cols = read_tecplot_any(c["path"])
            vm = variable_map(cols)

            x = df[vm["x"]].to_numpy(float)
            y = df[vm["y"]].to_numpy(float)
            rho = df[vm["rho"]].to_numpy(float)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(rho)

            if good.sum() < 100:
                continue

            row = {
                "path": str(c["path"]),
                "gas": c["gas"],
                "kn": c["kn"],
                "mach": c["mach"],
                "source": c["source"],
                "n_points": int(good.sum()),
                "x_min": float(np.nanmin(x[good])),
                "x_max": float(np.nanmax(x[good])),
                "y_min": float(np.nanmin(y[good])),
                "y_max": float(np.nanmax(y[good])),
            }

            for key, val in vm.items():
                row[f"col_{key}"] = val if val is not None else ""

            rows.append(row)

        except Exception as e:
            print(f"[WARN] skipped unreadable field: {c['path']} :: {e}")

    inv = pd.DataFrame(rows)
    if inv.empty:
        return inv

    priority = {
        "mach_sweep_n2_fixedM15": 6,
        "mach_sweep_n2_fixedM9": 5,
        "mach_sweep_argon": 4,
        "mach_sweep_n2": 3,
        "kn_sweep": 2,
        "unknown": 1,
    }

    inv["priority"] = inv["source"].map(priority).fillna(0)
    inv = inv.sort_values(
        ["gas", "kn", "mach", "priority", "n_points"],
        ascending=[True, True, True, False, False],
    )
    inv = inv.drop_duplicates(subset=["gas", "kn", "mach"], keep="first").reset_index(drop=True)
    return inv


def read_case(row):
    df, cols = read_tecplot_any(Path(row["path"]))

    x = df[row["col_x"]].to_numpy(float)
    y = df[row["col_y"]].to_numpy(float)

    good = np.isfinite(x) & np.isfinite(y) & (y >= 0.0)

    data = {"x": x[good], "y": y[good]}
    for key in ["rho", "u", "v", "w", "ttr", "trot", "tvib", "mach", "p", "mfp"]:
        col = row.get(f"col_{key}", "")
        if isinstance(col, str) and len(col) and col in df.columns:
            arr = df[col].to_numpy(float)
            data[key] = arr[good]

    return data


# ============================================================
# Selection
# ============================================================

def select_kn_sweep(inv, gas, mach_ref, kn_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return pd.DataFrame()

    ginv["machdiff"] = np.abs(ginv["mach"] - mach_ref)
    rows = []
    available_kn = sorted(ginv["kn"].unique())
    for target in kn_values:
        k = min(available_kn, key=lambda a: abs(a-target))
        sub = ginv[np.isclose(ginv["kn"], k)].sort_values(
            ["machdiff", "priority", "n_points"],
            ascending=[True, False, False],
        )
        if not sub.empty:
            rows.append(sub.iloc[0])

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["gas", "kn", "mach"]).reset_index(drop=True)
    out["sweep"] = "Kn"
    out["parameter"] = out["kn"]
    return out


def select_mach_sweep(inv, gas, kn_ref, mach_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return pd.DataFrame()

    ginv["kndiff"] = np.abs(ginv["kn"] - kn_ref)
    rows = []
    available_m = sorted(ginv["mach"].unique())
    for target in mach_values:
        m = min(available_m, key=lambda a: abs(a-target))
        sub = ginv[np.isclose(ginv["mach"], m)].sort_values(
            ["kndiff", "priority", "n_points"],
            ascending=[True, False, False],
        )
        if not sub.empty:
            rows.append(sub.iloc[0])

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["gas", "kn", "mach"]).reset_index(drop=True)
    out["sweep"] = "Mach"
    out["parameter"] = out["mach"]
    return out


# ============================================================
# Ray interpolation and shock-local metrics
# ============================================================

def make_interpolator(data, var):
    if var not in data:
        return None
    x = data["x"]
    y = data["y"]
    q = data[var]
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(q)
    if good.sum() < 100:
        return None
    tri = mtri.Triangulation(x[good], y[good])
    return mtri.LinearTriInterpolator(tri, q[good])


def sample_body_normal_ray(data, interp, cx, cy, R, theta_deg,
                           ns=1200, smax_frac=8.0, wall_gap_frac=2e-3):
    """Sample a body-normal ray from the cylinder surface into the gas.

    theta_deg is the polar angle in degrees.  theta=180 is the stagnation line,
    theta=90 is the top of the cylinder.  For profile-collapse we recommend
    theta=135--165 to stay in the upstream shock sector but away from the wall
    boundary layer and interpolation boundary.
    """
    theta = np.deg2rad(theta_deg)
    ux, uy = np.cos(theta), np.sin(theta)

    x0 = cx + R * ux
    y0 = cy + R * uy

    xdata = data["x"]
    ydata = data["y"]
    xmin, xmax = float(np.nanmin(xdata)), float(np.nanmax(xdata))
    ymin, ymax = float(np.nanmin(ydata)), float(np.nanmax(ydata))

    cand = []
    if abs(ux) > 1e-12:
        cand += [(xmin - x0)/ux, (xmax - x0)/ux]
    if abs(uy) > 1e-12:
        cand += [(ymin - y0)/uy, (ymax - y0)/uy]

    cand = [c for c in cand if c > 0]
    if not cand:
        return None

    smax = min(min(cand), smax_frac * R)
    if smax <= 0:
        return None

    gap = wall_gap_frac * R
    s = np.linspace(gap, smax, ns)
    x = x0 + s * ux
    y = y0 + s * uy

    # Upper-half only.
    m = y >= cy - 1e-12
    if m.sum() < 80:
        return None
    s, x, y = s[m], x[m], y[m]

    q = interp(x, y)
    q = np.asarray(q.filled(np.nan) if hasattr(q, "filled") else q, dtype=float)

    good = np.isfinite(q)
    if good.sum() < 80:
        return None

    return s[good], q[good], x[good], y[good]


def smooth_profile(q, sigma=2.0):
    q = np.asarray(q, dtype=float)
    if SCIPY_OK and len(q) > 10 and sigma > 0:
        return gaussian_filter1d(q, sigma=sigma)
    return q


def interp_profile(xold, yold, xnew):
    xold = np.asarray(xold, dtype=float)
    yold = np.asarray(yold, dtype=float)
    good = np.isfinite(xold) & np.isfinite(yold)
    xold = xold[good]
    yold = yold[good]
    if len(xold) < 5:
        return np.full_like(xnew, np.nan, dtype=float)

    order = np.argsort(xold)
    xold = xold[order]
    yold = yold[order]
    xu, idx = np.unique(xold, return_index=True)
    yu = yold[idx]
    if len(xu) < 5:
        return np.full_like(xnew, np.nan, dtype=float)

    if SCIPY_OK:
        f = interp1d(xu, yu, bounds_error=False, fill_value=np.nan)
        return f(xnew)

    return np.interp(xnew, xu, yu, left=np.nan, right=np.nan)


def find_density_registration(s, rho, R, smooth_sigma=2.0,
                              wall_exclude_R=0.04,
                              far_exclude_frac=0.10,
                              local_window_frac=0.09):
    """Find density-based shock/layer centre and thickness on one ray."""
    s = np.asarray(s, dtype=float)
    rho = np.asarray(rho, dtype=float)

    good = np.isfinite(s) & np.isfinite(rho)
    s, rho = s[good], rho[good]
    if len(s) < 100:
        return None

    order = np.argsort(s)
    s, rho = s[order], rho[order]
    rhos = smooth_profile(rho, sigma=smooth_sigma)

    n = len(s)
    smax = float(np.nanmax(s))
    valid = (
        (s > wall_exclude_R * R) &
        (s < (1.0 - far_exclude_frac) * smax) &
        np.isfinite(rhos)
    )
    if valid.sum() < 30:
        valid = (s > 0.02*R) & (s < 0.95*smax) & np.isfinite(rhos)

    drds = np.gradient(rhos, s)
    g = np.abs(drds)

    if valid.sum() < 30 or not np.isfinite(g[valid]).any():
        return None

    idx_valid = np.where(valid)[0]
    imax = idx_valid[int(np.nanargmax(g[valid]))]
    max_grad = float(g[imax])
    if not np.isfinite(max_grad) or max_grad <= 0:
        return None

    half_w = max(12, int(local_window_frac * n))
    gap = max(3, int(0.012 * n))

    d0 = max(0, imax - half_w)
    d1 = max(d0 + 3, imax - gap)
    u0 = min(n - 3, imax + gap)
    u1 = min(n, imax + half_w)

    if d1 <= d0 + 2 or u1 <= u0 + 2:
        return None

    rho_down = float(np.nanmedian(rhos[d0:d1]))
    rho_up = float(np.nanmedian(rhos[u0:u1]))
    jump = rho_down - rho_up
    if not np.isfinite(jump) or abs(jump) < 1e-14:
        return None

    delta = abs(jump) / max_grad
    delta = float(np.clip(delta, 0.001*R, 20.0*R))
    s_peak = float(s[imax])

    return {
        "s_peak": s_peak,
        "delta": delta,
        "max_grad": max_grad,
        "imax": int(imax),
        "d0": int(d0),
        "d1": int(d1),
        "u0": int(u0),
        "u1": int(u1),
        "rho_up": rho_up,
        "rho_down": rho_down,
        "rho_jump": jump,
    }


def normalize_variable_about_density_layer(s, q, reg, smooth_sigma=2.0):
    """Normalize variable q using local windows defined by density registration.

    The registration coordinate is density-based, but q* uses the local q jump
    across the same density-defined compression layer.  This tests whether q
    follows the density-attached shock layer.
    """
    s = np.asarray(s, dtype=float)
    q = np.asarray(q, dtype=float)
    good = np.isfinite(s) & np.isfinite(q)
    s, q = s[good], q[good]
    if len(s) < 100:
        return None

    order = np.argsort(s)
    s, q = s[order], q[order]
    qs = smooth_profile(q, sigma=smooth_sigma)
    n = len(s)

    # Map density-window indices approximately to this variable profile.  The
    # profiles were sampled on the same ray and same ns, but NaNs may have been
    # removed, so use physical s windows instead of raw indices.
    s_peak = reg["s_peak"]
    delta = reg["delta"]

    # Use windows around density peak: downstream is smaller s, upstream larger s.
    w = max(0.35*delta, 0.03*(np.nanmax(s)-np.nanmin(s)))
    gap = max(0.08*delta, 0.005*(np.nanmax(s)-np.nanmin(s)))

    dmask = (s >= s_peak - w) & (s <= s_peak - gap)
    umask = (s >= s_peak + gap) & (s <= s_peak + w)

    # Fallback to broader windows if needed.
    if dmask.sum() < 4 or umask.sum() < 4:
        w = max(0.70*delta, 0.06*(np.nanmax(s)-np.nanmin(s)))
        gap = max(0.04*delta, 0.003*(np.nanmax(s)-np.nanmin(s)))
        dmask = (s >= s_peak - w) & (s <= s_peak - gap)
        umask = (s >= s_peak + gap) & (s <= s_peak + w)

    if dmask.sum() < 3 or umask.sum() < 3:
        return None

    q_down = float(np.nanmedian(qs[dmask]))
    q_up = float(np.nanmedian(qs[umask]))
    jump = q_down - q_up
    if not np.isfinite(jump) or abs(jump) < 1e-14:
        return None

    qstar = (qs - q_up) / jump
    xi_rho = (s - s_peak) / delta

    # For Mach, the jump can be negative depending on convention; keep
    # downstream=1/upstream=0 using the formula above.  Overshoot remains
    # physically informative.
    return {
        "s": s,
        "q": q,
        "qs": qs,
        "qstar": qstar,
        "xi_rho": xi_rho,
        "q_up": q_up,
        "q_down": q_down,
        "jump": jump,
        "s_rho": s_peak,
        "delta_rho": delta,
    }


def case_profiles_shocksector(row, cx, cy, R, variables, profile_angles,
                              ns=1200, smax_frac=8.0, smooth_sigma=2.0):
    """Extract median shock-sector profile for each variable.

    Registration is density-based:
      xi_rho = (s-s_rho)/delta_rho

    Every variable q is normalized with its local q jump across the same
    density-defined layer.  This directly tests whether thermal/Mach/pressure
    profiles collapse when aligned by the density shock/compression layer.
    """
    data = read_case(row)

    # Build interpolators.  Density is mandatory.
    interps = {}
    for var in set(["rho"] + list(variables)):
        I = make_interpolator(data, var)
        if I is not None:
            interps[var] = I
    if "rho" not in interps:
        return {}

    profiles_by_var = {v: {"profiles": [], "metrics_rows": []} for v in variables}

    for th in profile_angles:
        rho_prof = sample_body_normal_ray(
            data, interps["rho"], cx, cy, R, th,
            ns=ns,
            smax_frac=smax_frac,
        )
        if rho_prof is None:
            continue

        s_rho, q_rho, _, _ = rho_prof
        reg = find_density_registration(s_rho, q_rho, R, smooth_sigma=smooth_sigma)
        if reg is None:
            continue

        # Process all variables using this density registration.
        for var in variables:
            if var not in interps:
                continue

            prof = sample_body_normal_ray(
                data, interps[var], cx, cy, R, th,
                ns=ns,
                smax_frac=smax_frac,
            )
            if prof is None:
                continue

            s, q, _, _ = prof
            met = normalize_variable_about_density_layer(s, q, reg, smooth_sigma=smooth_sigma)
            if met is None:
                continue

            met["theta_deg"] = float(th)
            profiles_by_var[var]["profiles"].append(met)
            profiles_by_var[var]["metrics_rows"].append({
                "theta_deg": float(th),
                "q_up": met["q_up"],
                "q_down": met["q_down"],
                "jump": met["jump"],
                "s_rho": met["s_rho"],
                "delta_rho": met["delta_rho"],
                "rho_max_grad": reg["max_grad"],
            })

    # Remove empty vars and convert metrics to dataframes.
    out = {}
    for var, d in profiles_by_var.items():
        if d["profiles"]:
            out[var] = {
                "profiles": d["profiles"],
                "metrics": pd.DataFrame(d["metrics_rows"]),
            }
    return out


# ============================================================
# Family construction
# ============================================================

def build_family(selected, cx, cy, R, variables, profile_angles,
                 s_over_R_grid, xi_grid,
                 ns=1200, smax_frac=8.0, smooth_sigma=2.0):
    fields_phys = {v: [] for v in variables}
    fields_att = {v: [] for v in variables}
    meta = []
    rows = []

    s_grid = s_over_R_grid * R

    for _, row in selected.iterrows():
        print(f"[INFO] shock-sector profiles {row['sweep']} {row['gas']} Kn={row['kn']:g} M={row['mach']:g}")
        pvars = case_profiles_shocksector(
            row, cx, cy, R, variables, profile_angles,
            ns=ns,
            smax_frac=smax_frac,
            smooth_sigma=smooth_sigma,
        )

        if "rho" not in pvars:
            print(f"[WARN] no rho shock-sector profile: {row['gas']} Kn={row['kn']} M={row['mach']}")
            continue

        meta.append({
            "gas": row["gas"],
            "sweep": row["sweep"],
            "kn": float(row["kn"]),
            "mach": float(row["mach"]),
            "parameter": float(row["parameter"]),
        })

        for var in variables:
            if var not in pvars:
                fields_phys[var].append(np.full_like(s_over_R_grid, np.nan))
                fields_att[var].append(np.full_like(xi_grid, np.nan))
                continue

            phys_curves = []
            att_curves = []
            for met in pvars[var]["profiles"]:
                s = met["s"]
                qstar = met["qstar"]
                xi = met["xi_rho"]

                phys_curves.append(interp_profile(s, qstar, s_grid))
                att_curves.append(interp_profile(xi, qstar, xi_grid))

            if phys_curves:
                q_phys = np.nanmedian(np.asarray(phys_curves), axis=0)
                q_att = np.nanmedian(np.asarray(att_curves), axis=0)
            else:
                q_phys = np.full_like(s_over_R_grid, np.nan)
                q_att = np.full_like(xi_grid, np.nan)

            fields_phys[var].append(q_phys)
            fields_att[var].append(q_att)

            mdf = pvars[var]["metrics"]
            rows.append({
                "gas": row["gas"],
                "sweep": row["sweep"],
                "kn": float(row["kn"]),
                "mach": float(row["mach"]),
                "parameter": float(row["parameter"]),
                "var": var,
                "n_angles": int(len(mdf)),
                "q_up_median": float(np.nanmedian(mdf["q_up"])),
                "q_down_median": float(np.nanmedian(mdf["q_down"])),
                "jump_median": float(np.nanmedian(mdf["jump"])),
                "s_rho_median": float(np.nanmedian(mdf["s_rho"])),
                "delta_rho_median": float(np.nanmedian(mdf["delta_rho"])),
                "rho_max_grad_median": float(np.nanmedian(mdf["rho_max_grad"])),
            })

    fields_phys = {v: np.asarray(fields_phys[v], dtype=float) for v in variables}
    fields_att = {v: np.asarray(fields_att[v], dtype=float) for v in variables}
    return fields_phys, fields_att, pd.DataFrame(meta), pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def colors_for_meta(meta_df, sweep):
    param = meta_df["kn"].to_numpy(float) if sweep == "Kn" else meta_df["mach"].to_numpy(float)
    if len(param) == 0:
        return [], param

    if sweep == "Kn":
        vals = np.log10(param)
        cmap = plt.get_cmap("viridis")
    else:
        vals = param
        cmap = plt.get_cmap("plasma")

    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    colors = [cmap((v-vmin)/(vmax-vmin+1e-12)) for v in vals]
    return colors, param


def robust_ylim_for_panel(*arrays):
    vals = []
    for A in arrays:
        arr = np.asarray(A, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr):
            vals.append(arr)
    if not vals:
        return -0.1, 1.1

    v = np.concatenate(vals)
    lo, hi = np.nanpercentile(v, [1, 99])
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    pad = 0.08 * max(hi - lo, 1e-6)
    lo -= pad
    hi += pad

    # Prevent a few outliers from making the plot useless.
    lo = max(lo, -0.8)
    hi = min(hi, 2.2)

    if hi - lo < 0.2:
        lo, hi = -0.1, 1.1

    return float(lo), float(hi)


def plot_family(gas, sweep, s_over_R_grid, xi_grid, fields_phys, fields_att,
                meta_df, variables, outdir):
    if meta_df.empty:
        return

    vars_present = [v for v in variables if fields_phys.get(v, np.empty((0,))).shape[0] == len(meta_df)]

    # Choose variables for combined plot.
    if gas == "N2":
        order = [v for v in ["rho", "ttr", "trot"] if v in vars_present]
    else:
        order = [v for v in ["rho", "ttr", "mach"] if v in vars_present]

    if not order:
        return

    colors, param = colors_for_meta(meta_df, sweep)
    param_label = r"$Kn_\infty$" if sweep == "Kn" else r"$M_\infty$"

    fig, axes = plt.subplots(2, len(order), figsize=(4.3*len(order), 7.2), sharey=False)
    if len(order) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for j, var in enumerate(order):
        Aphys = fields_phys[var]
        Aatt = fields_att[var]
        lo, hi = robust_ylim_for_panel(Aphys, Aatt)

        ax = axes[0, j]
        for k in range(Aphys.shape[0]):
            lab = None
            if j == 0:
                lab = fr"{param_label}={param[k]:g}"
            ax.plot(s_over_R_grid, Aphys[k], color=colors[k], lw=1.8, alpha=0.92, label=lab)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.axhline(1, color="0.4", lw=0.8, ls="--")
        ax.set_title(f"({chr(97+j)}) {LABELS.get(var,var)}: physical")
        ax.set_xlabel(r"$s/R$ from wall")
        ax.set_ylabel(r"$q^*=(q-q_{up})/(q_{down}-q_{up})$")
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)

        ax = axes[1, j]
        for k in range(Aatt.shape[0]):
            ax.plot(xi_grid, Aatt[k], color=colors[k], lw=1.8, alpha=0.92)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.axhline(1, color="0.4", lw=0.8, ls="--")
        ax.axvline(0, color="0.25", lw=0.9, ls=":")
        ax.set_title(f"({chr(97+len(order)+j)}) {LABELS.get(var,var)}: attached")
        ax.set_xlabel(r"$\xi_\rho=(s-s_\rho)/\delta_\rho$")
        ax.set_ylabel(r"$q^*$")
        ax.set_xlim(float(np.nanmin(xi_grid)), float(np.nanmax(xi_grid)))
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(f"{gas}: {sweep}-sweep density-registered shock-sector profile collapse")
    fig.tight_layout()
    fname = f"Step2c_{gas}_{sweep}_rho_registered_profile_collapse"
    fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Individual variable figures
    for var in vars_present:
        Aphys = fields_phys[var]
        Aatt = fields_att[var]
        if Aphys.shape[0] != len(meta_df):
            continue

        lo, hi = robust_ylim_for_panel(Aphys, Aatt)
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))

        ax = axes[0]
        for k in range(Aphys.shape[0]):
            ax.plot(s_over_R_grid, Aphys[k], color=colors[k], lw=1.8, alpha=0.92,
                    label=fr"{param_label}={param[k]:g}")
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.axhline(1, color="0.4", lw=0.8, ls="--")
        ax.set_title(f"{LABELS.get(var,var)} physical coordinate")
        ax.set_xlabel(r"$s/R$ from wall")
        ax.set_ylabel(r"$q^*$")
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, frameon=True)

        ax = axes[1]
        for k in range(Aatt.shape[0]):
            ax.plot(xi_grid, Aatt[k], color=colors[k], lw=1.8, alpha=0.92)
        ax.axhline(0, color="0.4", lw=0.8, ls="--")
        ax.axhline(1, color="0.4", lw=0.8, ls="--")
        ax.axvline(0, color="0.25", lw=0.9, ls=":")
        ax.set_title(f"{LABELS.get(var,var)} attached coordinate")
        ax.set_xlabel(r"$\xi_\rho=(s-s_\rho)/\delta_\rho$")
        ax.set_ylabel(r"$q^*$")
        ax.set_xlim(float(np.nanmin(xi_grid)), float(np.nanmax(xi_grid)))
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.25)

        fig.suptitle(f"{gas}: {sweep}-sweep {PRETTY.get(var,var)} density-registered profile collapse")
        fig.tight_layout()
        fname = f"Step2c_{gas}_{sweep}_{var}_rho_registered_profiles"
        fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
        fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Density-registered shock-sector physical and attached profile collapse for DSMC cylinder data.")

    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract_root", default="_step2c_shocksector_extracted")

    ap.add_argument("--radius", type=float, default=0.1524)
    ap.add_argument("--leftmost_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)

    ap.add_argument("--default_kn_if_missing", type=float, default=0.01)
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0)

    ap.add_argument("--kn_sweep_mach", type=float, default=10.0)
    ap.add_argument("--mach_sweep_kn", type=float, default=0.01)
    ap.add_argument("--kn_values", nargs="+", type=float, default=[0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 0.75, 1.0])
    ap.add_argument("--mach_values", nargs="+", type=float, default=[5, 5.5, 6, 6.5, 7, 7.5, 8, 8.25, 8.5, 8.75, 9, 9.25, 9.5, 9.75, 10, 11, 12, 13, 14, 15])

    ap.add_argument("--variables", nargs="+", default=["rho", "mach", "p", "ttr", "trot"])

    ap.add_argument("--profile_angles", nargs="+", type=float, default=[135, 145, 155, 165],
                    help="Body-normal angles in degrees. Recommended: 135 145 155 165.")
    ap.add_argument("--smax_over_R", type=float, default=6.0)
    ap.add_argument("--xi_min", type=float, default=-3.0)
    ap.add_argument("--xi_max", type=float, default=6.0)
    ap.add_argument("--n_profile", type=int, default=600)
    ap.add_argument("--line_ns", type=int, default=1200)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)
    ap.add_argument("--smax_frac", type=float, default=8.0)

    args = ap.parse_args()

    outdir = ensure_dir(args.out)
    roots = prepare_inputs(args.inputs, ensure_dir(args.extract_root))
    cases = find_flow_files(roots, default_kn=args.default_kn_if_missing, default_mach=args.default_mach_if_missing)

    inv = build_inventory(cases)
    if inv.empty:
        raise RuntimeError("No valid cases found.")

    inv.to_csv(outdir / "Step2c_case_inventory.csv", index=False)

    cx = args.leftmost_x + args.radius
    cy = args.center_y
    R = args.radius

    selections = []
    for gas in ["Ar", "N2"]:
        kn_sel = select_kn_sweep(inv, gas, args.kn_sweep_mach, args.kn_values)
        mach_sel = select_mach_sweep(inv, gas, args.mach_sweep_kn, args.mach_values)
        if not kn_sel.empty:
            selections.append(kn_sel)
        if not mach_sel.empty:
            selections.append(mach_sel)

    if not selections:
        raise RuntimeError("No selected cases.")

    selected = pd.concat(selections, ignore_index=True).drop_duplicates(subset=["gas", "kn", "mach", "sweep"])
    selected.to_csv(outdir / "Step2c_selected_cases.csv", index=False)

    s_over_R_grid = np.linspace(0.0, args.smax_over_R, args.n_profile)
    xi_grid = np.linspace(args.xi_min, args.xi_max, args.n_profile)

    all_metrics = []

    for gas in ["Ar", "N2"]:
        for sweep in ["Kn", "Mach"]:
            sel = selected[(selected["gas"] == gas) & (selected["sweep"] == sweep)].copy()
            if sel.empty:
                continue
            sel = sel.sort_values("parameter")

            fields_phys, fields_att, meta_df, metrics_df = build_family(
                sel, cx, cy, R,
                variables=args.variables,
                profile_angles=args.profile_angles,
                s_over_R_grid=s_over_R_grid,
                xi_grid=xi_grid,
                ns=args.line_ns,
                smax_frac=args.smax_frac,
                smooth_sigma=args.smooth_sigma,
            )

            if not metrics_df.empty:
                all_metrics.append(metrics_df)

            plot_family(
                gas, sweep,
                s_over_R_grid, xi_grid,
                fields_phys, fields_att,
                meta_df, args.variables,
                outdir,
            )

    if all_metrics:
        pd.concat(all_metrics, ignore_index=True).to_csv(outdir / "Step2c_profile_metrics.csv", index=False)

    print("\n[OK] Step 2c density-registered shock-sector profile collapse complete.")
    print(f"[OK] Output directory: {outdir.resolve()}")
    print("[OK] Main files to inspect:")
    print("     Step2c_N2_Kn_rho_registered_profile_collapse.pdf")
    print("     Step2c_Ar_Kn_rho_registered_profile_collapse.pdf")
    print("     Step2c_N2_Kn_rho_rho_registered_profiles.pdf")
    print("     Step2c_N2_Kn_ttr_rho_registered_profiles.pdf")
    print("     Step2c_N2_Kn_trot_rho_registered_profiles.pdf")
    print("     Step2c_profile_metrics.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cylinder_modal_step2b_profiles_and_modes_v3_clean_thermal_modes.py

STEP 2b for the JFM cylinder paper.

Purpose
-------
This script produces two types of clean, paper-oriented figures from the DSMC
cylinder field files:

1) Profile-collapse figures:
   physical-coordinate profiles q(s) versus shock-attached profiles
   q((r-r_q)/delta_q).  These figures directly show that density collapses
   much better than thermal variables after shock/layer registration.

2) Improved POD mode figures:
   larger and cleaner POD mode maps with cropped xi-window, fewer panels, and
   less useless white/blank space.

The script is intentionally independent and conservative:
- It reads the same DS2V/Tecplot-style field files used by Step 1/Step 2.
- It does not plot contours.
- It samples profiles along body-normal rays.
- It saves both PNG and PDF.
- It writes CSV summaries for checking.

Recommended use for the paper:
- Main text: use N2 Kn-sweep profile-collapse figure.
- Main text or appendix: use improved N2 thermal POD mode figures.
- Supplement: use Ar profile-collapse and Mach-sweep figures.

Author: ChatGPT for Ehsan Roohi
"""

from __future__ import annotations

import argparse
import math
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
# Matplotlib style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "savefig.dpi": 300,
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
# Utilities
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
# Case selection
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
# Interpolation and ray sampling
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


def sample_ray_from_surface(data, interp, cx, cy, R, theta, ns=900, smax_frac=8.0):
    ux, uy = np.cos(theta), np.sin(theta)
    x0, y0 = cx + R * ux, cy + R * uy

    x = data["x"]
    y = data["y"]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))

    cand = []
    if abs(ux) > 1e-12:
        cand += [(xmin - x0) / ux, (xmax - x0) / ux]
    if abs(uy) > 1e-12:
        cand += [(ymin - y0) / uy, (ymax - y0) / uy]

    cand = [c for c in cand if c > 0]
    if not cand:
        return None

    smax = min(min(cand), smax_frac * R)
    if smax <= 0:
        return None

    s = np.linspace(0.0, smax, ns)
    xs = x0 + s * ux
    ys = y0 + s * uy

    m = ys >= cy - 1e-12
    if m.sum() < 50:
        return None

    s = s[m]
    xs = xs[m]
    ys = ys[m]

    q = interp(xs, ys)
    q = np.asarray(q.filled(np.nan) if hasattr(q, "filled") else q, dtype=float)

    good = np.isfinite(q)
    if good.sum() < 50:
        return None

    return s[good], q[good], xs[good], ys[good]


def smooth_profile(q, sigma=2.0):
    q = np.asarray(q, dtype=float)
    if SCIPY_OK and len(q) > 10 and sigma > 0:
        return gaussian_filter1d(q, sigma=sigma)
    return q


def metrics_from_profile(s, q, R, smooth_sigma=2.0):
    s = np.asarray(s, dtype=float)
    q = np.asarray(q, dtype=float)

    good = np.isfinite(s) & np.isfinite(q)
    s = s[good]
    q = q[good]
    if len(s) < 80:
        return None

    order = np.argsort(s)
    s = s[order]
    q = q[order]
    qs = smooth_profile(q, sigma=smooth_sigma)

    n = len(s)
    near0 = max(3, int(0.03*n))
    near1 = max(near0 + 8, int(0.23*n))
    far_n = max(12, int(0.15*n))

    q_down = float(np.nanmedian(qs[near0:near1]))
    q_up = float(np.nanmedian(qs[-far_n:]))
    dq = q_down - q_up
    if not np.isfinite(dq) or abs(dq) < 1e-12:
        return None

    dqds = np.gradient(qs, s)
    g = np.abs(dqds)
    valid = (s > 0.015*R) & (s < 0.94*np.nanmax(s)) & np.isfinite(g)
    if valid.sum() < 20:
        return None

    idx_valid = np.where(valid)[0]
    imax = idx_valid[int(np.nanargmax(g[valid]))]
    max_grad = float(g[imax])
    if not np.isfinite(max_grad) or max_grad <= 0:
        return None

    r_peak = float(s[imax])
    delta = abs(dq) / max_grad
    delta = float(np.clip(delta, 0.001*R, 20.0*R))

    # normalized profile: upstream ~0, downstream ~1
    qnorm = (qs - q_up) / dq
    return {
        "s": s,
        "q": q,
        "qs": qs,
        "qnorm": qnorm,
        "q_up": q_up,
        "q_down": q_down,
        "jump": dq,
        "r_peak": r_peak,
        "delta": delta,
        "max_grad": max_grad,
    }


def interpolate_profile(xold, yold, xnew):
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

    # remove duplicate x values
    xu, ind = np.unique(xold, return_index=True)
    yu = yold[ind]
    if len(xu) < 5:
        return np.full_like(xnew, np.nan, dtype=float)

    if SCIPY_OK:
        f = interp1d(xu, yu, bounds_error=False, fill_value=np.nan)
        return f(xnew)

    return np.interp(xnew, xu, yu, left=np.nan, right=np.nan)


# ============================================================
# Profile extraction for collapse and POD
# ============================================================

def get_case_profile(row, cx, cy, R, theta_deg, variables, ray_ns=900, smooth_sigma=2.0):
    """Extract stagnation/near-stagnation ray profiles from one case."""
    data = read_case(row)
    theta = np.deg2rad(theta_deg)

    profiles = {}
    for var in variables:
        interp = make_interpolator(data, var)
        if interp is None:
            continue
        prof = sample_ray_from_surface(data, interp, cx, cy, R, theta, ns=ray_ns)
        if prof is None:
            continue
        s, q, xs, ys = prof
        met = metrics_from_profile(s, q, R, smooth_sigma=smooth_sigma)
        if met is None:
            continue
        profiles[var] = met
    return profiles


def build_profile_family(selected, cx, cy, R, variables, theta_deg=180.0,
                         s_grid=None, xi_grid=None, ray_ns=900, smooth_sigma=2.0):
    if s_grid is None:
        s_grid = np.linspace(0.0, 1.2, 500)
    if xi_grid is None:
        xi_grid = np.linspace(-2.0, 5.0, 500)

    rows = []
    fields_phys = {v: [] for v in variables}
    fields_att = {v: [] for v in variables}
    meta = []

    for _, row in selected.iterrows():
        print(f"[INFO] profile {row['sweep']} {row['gas']} Kn={row['kn']:g} M={row['mach']:g}")
        profiles = get_case_profile(row, cx, cy, R, theta_deg, variables, ray_ns, smooth_sigma)
        if "rho" not in profiles:
            continue

        for var in variables:
            if var not in profiles:
                fields_phys[var].append(np.full_like(s_grid, np.nan))
                fields_att[var].append(np.full_like(xi_grid, np.nan))
                continue

            p = profiles[var]
            s = p["s"]
            qn = p["qnorm"]
            xi = (s - p["r_peak"]) / p["delta"]

            q_phys = interpolate_profile(s, qn, s_grid)
            q_att = interpolate_profile(xi, qn, xi_grid)

            fields_phys[var].append(q_phys)
            fields_att[var].append(q_att)

            rows.append({
                "gas": row["gas"],
                "sweep": row["sweep"],
                "kn": float(row["kn"]),
                "mach": float(row["mach"]),
                "parameter": float(row["parameter"]),
                "var": var,
                "q_up": p["q_up"],
                "q_down": p["q_down"],
                "jump": p["jump"],
                "r_peak": p["r_peak"],
                "delta": p["delta"],
                "max_grad": p["max_grad"],
            })

        meta.append({
            "gas": row["gas"],
            "sweep": row["sweep"],
            "kn": float(row["kn"]),
            "mach": float(row["mach"]),
            "parameter": float(row["parameter"]),
        })

    fields_phys = {v: np.asarray(fields_phys[v], dtype=float) for v in variables}
    fields_att = {v: np.asarray(fields_att[v], dtype=float) for v in variables}
    meta_df = pd.DataFrame(meta)
    metrics_df = pd.DataFrame(rows)

    return s_grid, xi_grid, fields_phys, fields_att, meta_df, metrics_df


# ============================================================
# POD from ray-angle fields
# ============================================================

def build_attached_field_family(selected, cx, cy, R, variables,
                                theta_min_deg=100.0, theta_max_deg=180.0, n_theta=65,
                                xi_min=-1.0, xi_max=4.0, n_xi=260,
                                ray_ns=900, smooth_sigma=2.0):
    """Build q(xi,theta) attached snapshots for POD maps.

    This function is designed for clean mode maps:
    - crop xi to a compact useful range
    - build a common valid mask and crop blank columns automatically
    """
    theta_grid = np.linspace(theta_min_deg, theta_max_deg, n_theta)
    xi_grid = np.linspace(xi_min, xi_max, n_xi)

    snapshots = {v: [] for v in variables}
    meta = []

    for _, row in selected.iterrows():
        print(f"[INFO] field {row['sweep']} {row['gas']} Kn={row['kn']:g} M={row['mach']:g}")
        data = read_case(row)
        interps = {v: make_interpolator(data, v) for v in variables}
        interps = {v: I for v, I in interps.items() if I is not None}
        if "rho" not in interps:
            continue

        case_fields = {v: [] for v in variables}

        for th_deg in theta_grid:
            th = np.deg2rad(th_deg)
            # First density profile helps but each variable uses own peak/thickness.
            for var in variables:
                if var not in interps:
                    case_fields[var].append(np.full_like(xi_grid, np.nan))
                    continue
                prof = sample_ray_from_surface(data, interps[var], cx, cy, R, th, ns=ray_ns)
                if prof is None:
                    case_fields[var].append(np.full_like(xi_grid, np.nan))
                    continue
                s, q, xs, ys = prof
                met = metrics_from_profile(s, q, R, smooth_sigma=smooth_sigma)
                if met is None:
                    case_fields[var].append(np.full_like(xi_grid, np.nan))
                    continue
                xi = (met["s"] - met["r_peak"]) / met["delta"]
                qn = met["qnorm"]
                case_fields[var].append(interpolate_profile(xi, qn, xi_grid))

        for var in variables:
            A = np.asarray(case_fields[var], dtype=float)  # theta x xi
            snapshots[var].append(A)

        meta.append({
            "gas": row["gas"],
            "sweep": row["sweep"],
            "kn": float(row["kn"]),
            "mach": float(row["mach"]),
            "parameter": float(row["parameter"]),
        })

    snapshots = {v: np.asarray(snapshots[v], dtype=float) for v in variables}
    return xi_grid, theta_grid, snapshots, pd.DataFrame(meta)


def pod_decompose(snapshot_cube, valid_fraction=0.70):
    """POD of snapshots with common valid mask.

    snapshot_cube shape: n_snap x n_theta x n_xi
    Returns dict or None.
    """
    Xcube = np.asarray(snapshot_cube, dtype=float)
    if Xcube.ndim != 3 or Xcube.shape[0] < 2:
        return None

    n_snap, n_th, n_xi = Xcube.shape
    valid = np.isfinite(Xcube).mean(axis=0) >= valid_fraction
    if valid.sum() < 20:
        # try looser
        valid = np.isfinite(Xcube).mean(axis=0) >= 0.50
    if valid.sum() < 20:
        return None

    # Fill remaining NaNs at valid points by snapshot mean.
    X = []
    for k in range(n_snap):
        a = Xcube[k].copy()
        vals = a[valid]
        med = np.nanmedian(vals) if np.isfinite(vals).any() else 0.0
        vals = np.where(np.isfinite(vals), vals, med)
        X.append(vals)
    X = np.asarray(X, dtype=float)

    mean = X.mean(axis=0)
    Xc = X - mean[None, :]

    # If very small variance, skip.
    norm = np.linalg.norm(Xc)
    if not np.isfinite(norm) or norm <= 1e-14:
        return None

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    energy = S**2
    energy = energy / energy.sum()

    def map_back(vec):
        A = np.full((n_th, n_xi), np.nan)
        A[valid] = vec
        return A

    mean_map = map_back(mean)
    modes = [map_back(Vt[i]) for i in range(min(4, len(S)))]

    # Auto-crop blank xi columns: keep columns with at least some valid data.
    col_valid = np.isfinite(mean_map).mean(axis=0) > 0.2
    if col_valid.any():
        j = np.where(col_valid)[0]
        crop = (max(0, j[0]-2), min(n_xi, j[-1]+3))
    else:
        crop = (0, n_xi)

    return {
        "mean_map": mean_map,
        "modes": modes,
        "singular_values": S,
        "energy": energy,
        "valid_mask": valid,
        "xi_crop": crop,
    }


# ============================================================
# Plotting
# ============================================================

def colors_for_meta(meta_df, sweep):
    if meta_df.empty:
        return []
    param = meta_df["kn"].to_numpy(float) if sweep == "Kn" else meta_df["mach"].to_numpy(float)
    if sweep == "Kn":
        vals = np.log10(param)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        cmap = plt.get_cmap("viridis")
        return [cmap((v-vmin)/(vmax-vmin+1e-12)) for v in vals], param
    else:
        vals = param
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        cmap = plt.get_cmap("plasma")
        return [cmap((v-vmin)/(vmax-vmin+1e-12)) for v in vals], param



def _set_profile_ylim(ax, *arrays):
    """Robust y limits for normalized profile plots.

    The older version forced [-0.1, 1.15].  That could hide some profiles
    when the robust upstream/downstream normalization produced overshoots,
    especially for density in the attached coordinate.
    """
    vals = []
    for A in arrays:
        if A is None:
            continue
        aa = np.asarray(A, dtype=float)
        vals.append(aa[np.isfinite(aa)].ravel())
    vals = [v for v in vals if len(v)]
    if not vals:
        ax.set_ylim(-0.1, 1.15)
        return

    v = np.concatenate(vals)
    lo = np.nanpercentile(v, 1)
    hi = np.nanpercentile(v, 99)
    lo = min(lo, -0.05)
    hi = max(hi, 1.05)

    # Avoid crazy scaling caused by a few extrapolated points.
    if hi - lo > 4.0:
        lo = max(np.nanpercentile(v, 5), -0.5)
        hi = min(np.nanpercentile(v, 95), 2.0)
        lo = min(lo, -0.05)
        hi = max(hi, 1.05)

    pad = 0.06 * max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0.0, color="0.65", lw=0.8, ls="--", zorder=0)
    ax.axhline(1.0, color="0.65", lw=0.8, ls="--", zorder=0)

def plot_profile_collapse(gas, sweep, s_grid, xi_grid, fields_phys, fields_att,
                          meta_df, variables, outdir):
    if meta_df.empty:
        return

    # Use up to 3 variables per figure for clarity.
    vars_present = [v for v in variables if v in fields_phys and fields_phys[v].shape[0] == len(meta_df)]
    if not vars_present:
        return

    # Preferred order for the main paper.
    ordered = [v for v in ["rho", "ttr", "trot", "mach", "p"] if v in vars_present]
    vars_to_plot = ordered[:3] if len(ordered) >= 3 else ordered

    colors, param = colors_for_meta(meta_df, sweep)

    fig, axes = plt.subplots(2, len(vars_to_plot), figsize=(4.2*len(vars_to_plot), 7.2), sharey=False)
    if len(vars_to_plot) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for j, var in enumerate(vars_to_plot):
        Aphys = fields_phys[var]
        Aatt = fields_att[var]

        ax = axes[0, j]
        for k in range(Aphys.shape[0]):
            lab = None
            if j == 0:
                lab = (fr"$Kn_\infty={param[k]:g}$" if sweep == "Kn" else fr"$M_\infty={param[k]:g}$")
            ax.plot(s_grid, Aphys[k], color=colors[k], lw=1.7, alpha=0.95, label=lab)
        ax.set_title(f"({chr(97+j)}) {LABELS.get(var,var)}: physical")
        ax.set_xlabel(r"$s$ from wall (m)")
        ax.set_ylabel("normalized profile (-)")
        _set_profile_ylim(ax, Aphys)
        ax.grid(True, alpha=0.25)

        ax = axes[1, j]
        for k in range(Aatt.shape[0]):
            ax.plot(xi_grid, Aatt[k], color=colors[k], lw=1.7, alpha=0.95)
        ax.set_title(f"({chr(97+len(vars_to_plot)+j)}) {LABELS.get(var,var)}: attached")
        ax.set_xlabel(r"$\xi_q=(r-r_q)/\delta_q$ (-)")
        ax.set_ylabel("normalized profile (-)")
        ax.set_xlim(np.nanmin(xi_grid), np.nanmax(xi_grid))
        _set_profile_ylim(ax, Aatt)
        ax.grid(True, alpha=0.25)

    # external legend only once
    if len(vars_to_plot) > 0:
        axes[0, 0].legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle(f"{gas}: {sweep}-sweep physical vs shock-attached profile collapse")
    fig.tight_layout()
    fname = f"Step2b_{gas}_{sweep}_profile_collapse"
    fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pod_modes_clean(gas, sweep, var, xi_grid, theta_grid, pod, outdir, n_modes=2):
    if pod is None:
        return

    j0, j1 = pod["xi_crop"]
    xi = xi_grid[j0:j1]
    th = theta_grid

    maps = [pod["mean_map"][:, j0:j1]]
    titles = ["mean"]
    for i in range(min(n_modes, len(pod["modes"]))):
        maps.append(pod["modes"][i][:, j0:j1])
        e = pod["energy"][i] if i < len(pod["energy"]) else np.nan
        titles.append(f"mode {i+1}, E={e:.3f}")

    ncols = len(maps)
    fig, axes = plt.subplots(1, ncols, figsize=(3.3*ncols, 4.2), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, (ax, A, title) in enumerate(zip(axes, maps, titles)):
        # robust scale
        if i == 0:
            vmin = np.nanpercentile(A, 2)
            vmax = np.nanpercentile(A, 98)
            cmap = "viridis"
        else:
            vmax = np.nanpercentile(np.abs(A[np.isfinite(A)]), 98) if np.isfinite(A).any() else 1.0
            vmin = -vmax
            cmap = "coolwarm"

        im = ax.imshow(
            A,
            origin="lower",
            aspect="auto",
            extent=[xi.min(), xi.max(), th.min(), th.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(r"$\xi_q$")
        if i == 0:
            ax.set_ylabel(r"$\theta$ (deg)")
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{gas} {sweep} attached POD: {PRETTY.get(var,var)}")
    fig.tight_layout()
    fname = f"Step2b_{gas}_{sweep}_POD_modes_{var}_clean"
    fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pod_modes_thermal_combined_clean(gas, sweep, xi_grid, theta_grid, pods, outdir, n_modes=2):
    """Combined two-row figure for N2 thermal POD modes.

    Rows: Ttr and Trot.
    Columns: mean, mode 1, mode 2.
    This is intended to replace the old crowded 5-column mode figure.
    """
    wanted = ["ttr", "trot"]
    if any(v not in pods or pods[v] is None for v in wanted):
        return

    ncols = 1 + n_modes
    fig, axes = plt.subplots(
        len(wanted), ncols,
        figsize=(3.4*ncols, 3.4*len(wanted)),
        sharey=True
    )

    if len(wanted) == 1:
        axes = np.asarray([axes])

    for r, var in enumerate(wanted):
        pod = pods[var]
        j0, j1 = pod["xi_crop"]

        # Extra trimming to remove nearly empty margins.
        j0 = max(0, j0 + 1)
        j1 = min(len(xi_grid), j1 - 1)
        xi = xi_grid[j0:j1]
        th = theta_grid

        maps = [pod["mean_map"][:, j0:j1]]
        titles = [f"{LABELS.get(var,var)} mean"]
        for i in range(min(n_modes, len(pod["modes"]))):
            maps.append(pod["modes"][i][:, j0:j1])
            e = pod["energy"][i] if i < len(pod["energy"]) else np.nan
            titles.append(f"mode {i+1}, $E_{i+1}$={e:.3f}")

        for c, (A, title) in enumerate(zip(maps, titles)):
            ax = axes[r, c]

            # Mask invalid/blank regions as white.
            Am = np.ma.masked_invalid(A)

            if c == 0:
                finite = A[np.isfinite(A)]
                if len(finite):
                    vmin, vmax = np.nanpercentile(finite, [2, 98])
                else:
                    vmin, vmax = 0.0, 1.0
                cmap = plt.get_cmap("viridis").copy()
            else:
                finite = np.abs(A[np.isfinite(A)])
                vmax = np.nanpercentile(finite, 98) if len(finite) else 1.0
                vmin = -vmax
                cmap = plt.get_cmap("coolwarm").copy()

            cmap.set_bad("white")

            im = ax.imshow(
                Am,
                origin="lower",
                aspect="auto",
                extent=[xi.min(), xi.max(), th.min(), th.max()],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            ax.set_title(title, fontsize=11)
            ax.set_xlabel(r"$\xi_q$")
            if c == 0:
                ax.set_ylabel(r"$\theta$ (deg)")
                # Row label outside plot
                ax.text(
                    -0.28, 0.5, PRETTY.get(var, var),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=13,
                    fontweight="bold"
                )

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"{gas} {sweep} density-attached POD modes: nitrogen thermal fields", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fname = f"Step2b_{gas}_{sweep}_POD_modes_thermal_combined_clean"
    fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_pod_spectrum_clean(gas, sweep, pods, outdir):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    any_plot = False
    for var, pod in pods.items():
        if pod is None:
            continue
        cum = np.cumsum(pod["energy"])
        ax.plot(np.arange(1, len(cum)+1), cum, marker="o", lw=1.8, label=LABELS.get(var, var))
        any_plot = True
    if not any_plot:
        plt.close(fig)
        return
    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(f"{gas} {sweep}: clean attached-coordinate POD")
    fig.tight_layout()
    fname = f"Step2b_{gas}_{sweep}_POD_spectrum_clean"
    fig.savefig(outdir / f"{fname}.png", dpi=350, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Step 2b: profile-collapse and clean POD modes for DSMC cylinder data.")

    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract_root", default="_step2b_extracted")

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

    # Profile collapse options.
    ap.add_argument("--profile_theta", type=float, default=180.0, help="Ray angle for profile collapse. 180 is stagnation line.")
    ap.add_argument("--smax", type=float, default=1.2, help="Maximum physical distance from wall for profile plots.")
    ap.add_argument("--xi_min", type=float, default=-2.0)
    ap.add_argument("--xi_max", type=float, default=5.0)
    ap.add_argument("--n_profile", type=int, default=500)

    # POD mode map options.
    ap.add_argument("--theta_min", type=float, default=100.0)
    ap.add_argument("--theta_max", type=float, default=180.0)
    ap.add_argument("--n_theta", type=int, default=65)
    ap.add_argument("--mode_xi_min", type=float, default=-1.0)
    ap.add_argument("--mode_xi_max", type=float, default=4.0)
    ap.add_argument("--n_xi", type=int, default=260)
    ap.add_argument("--n_modes_plot", type=int, default=2)

    ap.add_argument("--ray_ns", type=int, default=900)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)

    args = ap.parse_args()

    outdir = ensure_dir(args.out)
    roots = prepare_inputs(args.inputs, ensure_dir(args.extract_root))
    cases = find_flow_files(roots, default_kn=args.default_kn_if_missing, default_mach=args.default_mach_if_missing)

    inv = build_inventory(cases)
    if inv.empty:
        raise RuntimeError("No valid cases found.")

    inv.to_csv(outdir / "Step2b_case_inventory.csv", index=False)

    cx = args.leftmost_x + args.radius
    cy = args.center_y
    R = args.radius

    selected_all = []
    for gas in ["Ar", "N2"]:
        kn_sel = select_kn_sweep(inv, gas, args.kn_sweep_mach, args.kn_values)
        mach_sel = select_mach_sweep(inv, gas, args.mach_sweep_kn, args.mach_values)
        if not kn_sel.empty:
            selected_all.append(kn_sel)
        if not mach_sel.empty:
            selected_all.append(mach_sel)

    if not selected_all:
        raise RuntimeError("No selected cases for Kn or Mach sweeps.")

    selected = pd.concat(selected_all, ignore_index=True).drop_duplicates(subset=["gas", "kn", "mach", "sweep"])
    selected.to_csv(outdir / "Step2b_selected_cases.csv", index=False)

    all_profile_metrics = []

    s_grid = np.linspace(0.0, args.smax, args.n_profile)
    xi_grid_profile = np.linspace(args.xi_min, args.xi_max, args.n_profile)

    # Profile-collapse figures and improved POD maps.
    for gas in ["Ar", "N2"]:
        for sweep in ["Kn", "Mach"]:
            sel = selected[(selected["gas"] == gas) & (selected["sweep"] == sweep)].copy()
            if sel.empty:
                continue
            sel = sel.sort_values("parameter")

            # Profile-collapse
            s_grid, xi_grid_profile, fields_phys, fields_att, meta_df, metrics_df = build_profile_family(
                sel, cx, cy, R,
                variables=args.variables,
                theta_deg=args.profile_theta,
                s_grid=s_grid,
                xi_grid=xi_grid_profile,
                ray_ns=args.ray_ns,
                smooth_sigma=args.smooth_sigma,
            )

            if not metrics_df.empty:
                all_profile_metrics.append(metrics_df)

            plot_profile_collapse(
                gas, sweep,
                s_grid, xi_grid_profile,
                fields_phys, fields_att,
                meta_df,
                args.variables,
                outdir,
            )

            # POD mode maps
            xi_grid_modes, theta_grid, snapshots, mode_meta = build_attached_field_family(
                sel, cx, cy, R,
                variables=args.variables,
                theta_min_deg=args.theta_min,
                theta_max_deg=args.theta_max,
                n_theta=args.n_theta,
                xi_min=args.mode_xi_min,
                xi_max=args.mode_xi_max,
                n_xi=args.n_xi,
                ray_ns=args.ray_ns,
                smooth_sigma=args.smooth_sigma,
            )

            pods = {}
            for var in args.variables:
                if var not in snapshots:
                    pods[var] = None
                    continue
                pod = pod_decompose(snapshots[var], valid_fraction=0.70)
                pods[var] = pod

                if pod is not None:
                    # Save clean modes for all relevant variables.  Main interest:
                    # N2 ttr/trot, and optionally rho.
                    if var in ["rho", "ttr", "trot"]:
                        plot_pod_modes_clean(
                            gas, sweep, var,
                            xi_grid_modes, theta_grid,
                            pod, outdir,
                            n_modes=args.n_modes_plot,
                        )

            # Combined replacement figure for the old crowded N2 thermal mode plot.
            if gas == "N2" and sweep == "Kn":
                plot_pod_modes_thermal_combined_clean(
                    gas, sweep,
                    xi_grid_modes, theta_grid,
                    pods, outdir,
                    n_modes=2,
                )

            plot_pod_spectrum_clean(gas, sweep, pods, outdir)

            # Save POD energy summary
            pod_rows = []
            for var, pod in pods.items():
                if pod is None:
                    continue
                e = pod["energy"]
                pod_rows.append({
                    "gas": gas,
                    "sweep": sweep,
                    "var": var,
                    "E1": float(e[0]) if len(e) else np.nan,
                    "C2": float(np.sum(e[:2])) if len(e) >= 2 else np.nan,
                    "modes95": int(np.searchsorted(np.cumsum(e), 0.95) + 1),
                    "n_modes": int(len(e)),
                    "xi_crop_min": float(xi_grid_modes[pod["xi_crop"][0]]),
                    "xi_crop_max": float(xi_grid_modes[pod["xi_crop"][1]-1]),
                })
            pd.DataFrame(pod_rows).to_csv(outdir / f"Step2b_{gas}_{sweep}_POD_energy_summary.csv", index=False)

    if all_profile_metrics:
        pd.concat(all_profile_metrics, ignore_index=True).to_csv(outdir / "Step2b_profile_metrics.csv", index=False)

    print("\n[OK] Step 2b complete.")
    print(f"[OK] Output directory: {outdir.resolve()}")
    print("[OK] Important figures:")
    print("     Step2b_N2_Kn_profile_collapse.pdf")
    print("     Step2b_N2_Kn_POD_modes_ttr_clean.pdf")
    print("     Step2b_N2_Kn_POD_modes_trot_clean.pdf")
    print("     Step2b_Ar_Kn_profile_collapse.pdf")
    print("     Step2b_*_POD_spectrum_clean.pdf")


if __name__ == "__main__":
    main()

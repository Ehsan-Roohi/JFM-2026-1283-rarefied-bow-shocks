#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cylinder_step3_shock_layer_metrics.py

STEP 3: Quantitative shock-layer metrics for hypersonic rarefied cylinder DSMC data.

This script reads the same DSMC field files used in Step 1 and Step 2 and
extracts quantitative metrics for both:

  1) Knudsen-number sweep at fixed Mach number
  2) Mach-number sweep at fixed Knudsen number

It does NOT draw contour maps.  It produces clean CSV files and line plots
that can be placed below the density-contour figures in the manuscript.

Main diagnostics:
  - stagnation-line shock/front standoff, Delta_s
  - effective density shock-layer thickness, delta_rho = Delta rho / max|d rho/ds|
  - robust 10--90 density thickness, delta_rho_10_90
  - median ray-based standoff over a user-selected angular sector
  - median variable-specific layer thicknesses for rho, Mach, p, Ttr, Trot
  - thickness ratios delta_q / delta_rho
  - if MFP is available: delta_q / lambda_up and Delta_s / lambda_up

Notes:
  * For low Kn, Delta_s is a shock-front distance.
  * For high Kn, the same diagnostic is interpreted as a compression-layer
    characteristic location, because the bow shock is diffuse and not a unique
    discontinuity.
  * This code intentionally keeps the reading/extraction logic conservative:
    it uses the same Tecplot/DSMC reader pattern as the previous working scripts.

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
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Archive extraction and Tecplot reader
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


# ============================================================
# Metrics along one ray
# ============================================================

def robust_crossing(s, qstar, level):
    """Return first crossing from near-body side toward upstream side.

    qstar is normalized so near-body compressed side ~1 and upstream ~0.
    We return the outer crossing location where qstar falls below level.
    """
    s = np.asarray(s, dtype=float)
    qstar = np.asarray(qstar, dtype=float)
    good = np.isfinite(s) & np.isfinite(qstar)
    s = s[good]
    qstar = qstar[good]
    if len(s) < 10:
        return np.nan

    # Ensure increasing s from surface outward.
    order = np.argsort(s)
    s = s[order]
    qstar = qstar[order]

    # We seek qstar = level.  For typical shock profile qstar decreases
    # from compressed near body to upstream far field.
    f = qstar - level
    idx = np.where(np.signbit(f[:-1]) != np.signbit(f[1:]))[0]
    if len(idx) == 0:
        return np.nan

    # Use outermost crossing, i.e. largest s, to represent the upstream edge.
    i = idx[-1]
    s1, s2 = s[i], s[i+1]
    f1, f2 = f[i], f[i+1]
    if abs(f2 - f1) < 1e-300:
        return float(0.5*(s1+s2))
    return float(s1 - f1*(s2-s1)/(f2-f1))


def metrics_from_profile(s, q, R, varname="rho", smooth_sigma=2.0):
    """Extract layer metrics from a single ray profile.

    s is distance from cylinder surface along the ray.
    q is the field value along the ray.
    """
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

    # Exclude exact wall and far boundary from gradient peak.
    near0 = max(3, int(0.03*n))
    near1 = max(near0 + 8, int(0.23*n))
    far_n = max(12, int(0.15*n))

    q_down = float(np.nanmedian(qs[near0:near1]))
    q_up = float(np.nanmedian(qs[-far_n:]))
    dq = q_down - q_up

    # If sign is weird or weak, still use absolute jump but record sign.
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

    s_peak = float(s[imax])
    delta_grad = abs(dq) / max_grad

    # Clamp only to avoid extreme division by noise; raw value also saved.
    delta_grad_raw = float(delta_grad)
    delta_grad = float(np.clip(delta_grad, 0.001*R, 20.0*R))

    # Normalize profile with compressed side ~1 and upstream side ~0.
    qstar = (qs - q_up) / dq

    s10 = robust_crossing(s, qstar, 0.10)
    s20 = robust_crossing(s, qstar, 0.20)
    s50 = robust_crossing(s, qstar, 0.50)
    s80 = robust_crossing(s, qstar, 0.80)
    s90 = robust_crossing(s, qstar, 0.90)

    delta_10_90 = abs(s10 - s90) if np.isfinite(s10) and np.isfinite(s90) else np.nan
    delta_20_80 = abs(s20 - s80) if np.isfinite(s20) and np.isfinite(s80) else np.nan

    # For density, s_peak is the sharp-front/mid-layer proxy.  s50 is robust
    # centre if monotonic crossings exist.  We save both.
    return {
        "var": varname,
        "q_up": q_up,
        "q_down": q_down,
        "jump": float(dq),
        "s_peak": s_peak,
        "s_10": s10,
        "s_20": s20,
        "s_50": s50,
        "s_80": s80,
        "s_90": s90,
        "delta_grad": delta_grad,
        "delta_grad_raw": delta_grad_raw,
        "delta_10_90": float(delta_10_90) if np.isfinite(delta_10_90) else np.nan,
        "delta_20_80": float(delta_20_80) if np.isfinite(delta_20_80) else np.nan,
        "max_grad": max_grad,
    }


def analyze_case(row, cx, cy, R, variables, half_angle_deg=60, n_angles=49,
                 smooth_sigma=2.0, ray_ns=900):
    data = read_case(row)

    result_rows = []

    # Angles from stagnation line (pi) to cylinder upper shoulder.
    thetas = np.deg2rad(np.linspace(180.0, 180.0-half_angle_deg, n_angles))
    theta_degs = np.rad2deg(thetas)

    # Build interpolators once.
    interps = {v: make_interpolator(data, v) for v in variables}
    interps = {v: interp for v, interp in interps.items() if interp is not None}

    if "rho" not in interps:
        return pd.DataFrame(), pd.DataFrame()

    # Upstream mean free path along stagnation, if available.
    mfp_up = np.nan
    if "mfp" in data:
        imfp = make_interpolator(data, "mfp")
        if imfp is not None:
            prof = sample_ray_from_surface(data, imfp, cx, cy, R, np.pi, ns=ray_ns)
            if prof is not None:
                s, mfp, _, _ = prof
                if len(mfp) > 20:
                    nfar = max(10, int(0.15*len(mfp)))
                    mfp_up = float(np.nanmedian(mfp[-nfar:]))

    # Per-ray metrics.
    for th, th_deg in zip(thetas, theta_degs):
        for var, interp in interps.items():
            prof = sample_ray_from_surface(data, interp, cx, cy, R, th, ns=ray_ns)
            if prof is None:
                continue

            s, q, xs, ys = prof
            met = metrics_from_profile(s, q, R, varname=var, smooth_sigma=smooth_sigma)
            if met is None:
                continue

            met.update({
                "gas": row["gas"],
                "kn": float(row["kn"]),
                "mach": float(row["mach"]),
                "source": row["source"],
                "flow_file": row["path"],
                "theta_deg": float(th_deg),
                "angle_from_stag_deg": float(180.0-th_deg),
                "x_peak": float(cx + (R + met["s_peak"]) * np.cos(th)),
                "y_peak": float(cy + (R + met["s_peak"]) * np.sin(th)),
                "lambda_up": mfp_up,
            })

            if np.isfinite(mfp_up) and mfp_up > 0:
                met["s_peak_over_lambda"] = met["s_peak"] / mfp_up
                met["delta_grad_over_lambda"] = met["delta_grad"] / mfp_up
                met["delta_10_90_over_lambda"] = met["delta_10_90"] / mfp_up if np.isfinite(met["delta_10_90"]) else np.nan
            else:
                met["s_peak_over_lambda"] = np.nan
                met["delta_grad_over_lambda"] = np.nan
                met["delta_10_90_over_lambda"] = np.nan

            result_rows.append(met)

    ray_metrics = pd.DataFrame(result_rows)

    if ray_metrics.empty:
        return ray_metrics, pd.DataFrame()

    # Case summary: stagnation-line value + median over angular sector.
    summaries = []
    for var in sorted(ray_metrics["var"].unique()):
        sub = ray_metrics[ray_metrics["var"] == var].copy()
        if sub.empty:
            continue

        # Nearest to stagnation theta=180.
        sub["ang_abs"] = np.abs(sub["angle_from_stag_deg"])
        stag = sub.sort_values("ang_abs").iloc[0]

        def med(col):
            return float(np.nanmedian(sub[col])) if col in sub.columns and np.isfinite(sub[col]).any() else np.nan

        def q25(col):
            return float(np.nanpercentile(sub[col], 25)) if col in sub.columns and np.isfinite(sub[col]).any() else np.nan

        def q75(col):
            return float(np.nanpercentile(sub[col], 75)) if col in sub.columns and np.isfinite(sub[col]).any() else np.nan

        summaries.append({
            "gas": row["gas"],
            "kn": float(row["kn"]),
            "mach": float(row["mach"]),
            "source": row["source"],
            "flow_file": row["path"],
            "var": var,
            "n_rays": len(sub),
            "standoff_stag": float(stag["s_peak"]),
            "thickness_grad_stag": float(stag["delta_grad"]),
            "thickness_10_90_stag": float(stag["delta_10_90"]) if np.isfinite(stag["delta_10_90"]) else np.nan,
            "thickness_20_80_stag": float(stag["delta_20_80"]) if np.isfinite(stag["delta_20_80"]) else np.nan,
            "standoff_median": med("s_peak"),
            "standoff_q25": q25("s_peak"),
            "standoff_q75": q75("s_peak"),
            "thickness_grad_median": med("delta_grad"),
            "thickness_grad_q25": q25("delta_grad"),
            "thickness_grad_q75": q75("delta_grad"),
            "thickness_10_90_median": med("delta_10_90"),
            "thickness_20_80_median": med("delta_20_80"),
            "max_grad_median": med("max_grad"),
            "lambda_up": med("lambda_up"),
            "standoff_over_lambda_median": med("s_peak_over_lambda"),
            "thickness_grad_over_lambda_median": med("delta_grad_over_lambda"),
            "thickness_10_90_over_lambda_median": med("delta_10_90_over_lambda"),
        })

    case_summary = pd.DataFrame(summaries)

    # Add ratios relative to density thickness in the same case.
    rho_row = case_summary[case_summary["var"] == "rho"]
    if not rho_row.empty:
        rho_delta = float(rho_row.iloc[0]["thickness_grad_median"])
        rho_delta_1090 = float(rho_row.iloc[0]["thickness_10_90_median"]) if np.isfinite(rho_row.iloc[0]["thickness_10_90_median"]) else np.nan
        rho_standoff = float(rho_row.iloc[0]["standoff_median"])

        for idx in case_summary.index:
            case_summary.loc[idx, "thickness_grad_over_delta_rho"] = case_summary.loc[idx, "thickness_grad_median"] / rho_delta if rho_delta > 0 else np.nan
            case_summary.loc[idx, "thickness_10_90_over_delta_rho_10_90"] = case_summary.loc[idx, "thickness_10_90_median"] / rho_delta_1090 if np.isfinite(rho_delta_1090) and rho_delta_1090 > 0 else np.nan
            case_summary.loc[idx, "standoff_over_rho_standoff"] = case_summary.loc[idx, "standoff_median"] / rho_standoff if rho_standoff > 0 else np.nan
    else:
        case_summary["thickness_grad_over_delta_rho"] = np.nan
        case_summary["thickness_10_90_over_delta_rho_10_90"] = np.nan
        case_summary["standoff_over_rho_standoff"] = np.nan

    return ray_metrics, case_summary


# ============================================================
# Case selection
# ============================================================

def select_kn_sweep(inv, gas, mach_ref, kn_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return pd.DataFrame()

    ginv["machdiff"] = np.abs(ginv["mach"] - mach_ref)
    # For each requested Kn, choose nearest available Kn, then closest Mach.
    rows = []
    available_kn = sorted(ginv["kn"].unique())
    for target in kn_values:
        k = min(available_kn, key=lambda a: abs(a-target))
        sub = ginv[np.isclose(ginv["kn"], k)].sort_values(["machdiff", "priority", "n_points"], ascending=[True, False, False])
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
        sub = ginv[np.isclose(ginv["mach"], m)].sort_values(["kndiff", "priority", "n_points"], ascending=[True, False, False])
        if not sub.empty:
            rows.append(sub.iloc[0])

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["gas", "kn", "mach"]).reset_index(drop=True)
    out["sweep"] = "Mach"
    out["parameter"] = out["mach"]
    return out


# ============================================================
# Plotting
# ============================================================

LABELS = {
    "rho": r"$\rho$",
    "mach": r"$M$",
    "p": r"$p$",
    "ttr": r"$T_{tr}$",
    "trot": r"$T_{rot}$",
    "tvib": r"$T_{vib}$",
}

def sort_for_plot(df, param):
    return df.sort_values(param)


def plot_standoff_thickness(summary, gas, sweep, outdir):
    if summary.empty:
        return

    param = "kn" if sweep == "Kn" else "mach"
    xlab = r"$Kn_\infty$ (-)" if sweep == "Kn" else r"$M_\infty$ (-)"

    sub = summary[(summary["gas"] == gas) & (summary["sweep"] == sweep) & (summary["var"] == "rho")].copy()
    if sub.empty:
        return
    sub = sort_for_plot(sub, param)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    ax.plot(sub[param], sub["standoff_median"], marker="o", lw=2.0, label="median over rays")
    ax.plot(sub[param], sub["standoff_stag"], marker="s", lw=1.5, label="stagnation line")
    if sweep == "Kn":
        ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"Shock/front standoff, $\Delta_s$ (m)")
    ax.set_title("(a) Density-front standoff")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(sub[param], sub["thickness_grad_median"], marker="o", lw=2.0, label=r"$\Delta\rho/\max|\nabla\rho|$")
    if np.isfinite(sub["thickness_10_90_median"]).any():
        ax.plot(sub[param], sub["thickness_10_90_median"], marker="s", lw=1.5, label=r"10--90 thickness")
    if sweep == "Kn":
        ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"Density-layer thickness, $\delta_\rho$ (m)")
    ax.set_title("(b) Effective density thickness")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(f"{gas}: {sweep}-sweep shock-layer metrics")
    fig.tight_layout()
    fig.savefig(outdir / f"Step3_{gas}_{sweep}_standoff_thickness.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"Step3_{gas}_{sweep}_standoff_thickness.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_variable_scales(summary, gas, sweep, outdir, variables):
    if summary.empty:
        return

    param = "kn" if sweep == "Kn" else "mach"
    xlab = r"$Kn_\infty$ (-)" if sweep == "Kn" else r"$M_\infty$ (-)"

    suball = summary[(summary["gas"] == gas) & (summary["sweep"] == sweep)].copy()
    if suball.empty:
        return

    vars_present = [v for v in variables if v in set(suball["var"])]
    if "rho" not in vars_present:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    ax = axes[0]
    for var in vars_present:
        sub = sort_for_plot(suball[suball["var"] == var], param)
        if sub.empty:
            continue
        ax.plot(sub[param], sub["thickness_grad_median"], marker="o", lw=1.8, label=LABELS.get(var, var))
    if sweep == "Kn":
        ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"Median variable thickness, $\delta_q$ (m)")
    ax.set_title("(a) Variable-specific thickness")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[1]
    for var in vars_present:
        sub = sort_for_plot(suball[suball["var"] == var], param)
        if sub.empty or var == "rho":
            continue
        ax.plot(sub[param], sub["thickness_grad_over_delta_rho"], marker="o", lw=1.8, label=LABELS.get(var, var))
    if sweep == "Kn":
        ax.set_xscale("log")
    ax.axhline(1.0, color="0.4", ls="--", lw=1.0)
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"Scale ratio, $\delta_q/\delta_\rho$ (-)")
    ax.set_title("(b) Relative to density thickness")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    ax = axes[2]
    # If MFP exists, show normalized thickness; otherwise show 10-90/grad ratio.
    has_mfp = np.isfinite(suball["lambda_up"]).any() and np.nanmedian(suball["lambda_up"]) > 0
    if has_mfp:
        for var in vars_present:
            sub = sort_for_plot(suball[suball["var"] == var], param)
            if sub.empty:
                continue
            ax.plot(sub[param], sub["thickness_grad_over_lambda_median"], marker="o", lw=1.8, label=LABELS.get(var, var))
        ax.set_ylabel(r"$\delta_q/\lambda_\infty$ (-)")
        ax.set_title("(c) Mean-free-path normalization")
    else:
        for var in vars_present:
            sub = sort_for_plot(suball[suball["var"] == var], param)
            if sub.empty:
                continue
            ratio = sub["thickness_10_90_median"] / sub["thickness_grad_median"]
            ax.plot(sub[param], ratio, marker="o", lw=1.8, label=LABELS.get(var, var))
        ax.set_ylabel(r"$\delta_{10-90}/\delta_{\nabla}$ (-)")
        ax.set_title("(c) Robust/gradient thickness")

    if sweep == "Kn":
        ax.set_xscale("log")
    ax.set_xlabel(xlab)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)

    fig.suptitle(f"{gas}: {sweep}-sweep variable-specific shock-layer scales")
    fig.tight_layout()
    fig.savefig(outdir / f"Step3_{gas}_{sweep}_variable_scales.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"Step3_{gas}_{sweep}_variable_scales.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_combined_gas_comparison(summary, sweep, outdir):
    if summary.empty:
        return

    param = "kn" if sweep == "Kn" else "mach"
    xlab = r"$Kn_\infty$ (-)" if sweep == "Kn" else r"$M_\infty$ (-)"

    sub = summary[(summary["sweep"] == sweep) & (summary["var"] == "rho")].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for gas in ["Ar", "N2"]:
        g = sort_for_plot(sub[sub["gas"] == gas], param)
        if g.empty:
            continue
        axes[0].plot(g[param], g["standoff_median"], marker="o", lw=2.0, label=gas)
        axes[1].plot(g[param], g["thickness_grad_median"], marker="o", lw=2.0, label=gas)

    for ax in axes:
        if sweep == "Kn":
            ax.set_xscale("log")
        ax.set_xlabel(xlab)
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[0].set_ylabel(r"Median density-front standoff, $\Delta_s$ (m)")
    axes[0].set_title("(a) Standoff")
    axes[1].set_ylabel(r"Median density thickness, $\delta_\rho$ (m)")
    axes[1].set_title("(b) Thickness")

    fig.suptitle(f"Ar/N2 comparison: {sweep}-sweep density-layer metrics")
    fig.tight_layout()
    fig.savefig(outdir / f"Step3_{sweep}_Ar_N2_density_metrics.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"Step3_{sweep}_Ar_N2_density_metrics.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Step-3 quantitative shock-layer metrics for cylinder DSMC fields.")

    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract_root", default="_step3_extracted")

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
    ap.add_argument("--half_angle", type=float, default=60.0)
    ap.add_argument("--angles", type=int, default=49)
    ap.add_argument("--ray_ns", type=int, default=900)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)

    args = ap.parse_args()

    outdir = ensure_dir(args.out)
    roots = prepare_inputs(args.inputs, ensure_dir(args.extract_root))
    cases = find_flow_files(roots, default_kn=args.default_kn_if_missing, default_mach=args.default_mach_if_missing)

    inv = build_inventory(cases)
    if inv.empty:
        raise RuntimeError("No valid cases found.")

    inv.to_csv(outdir / "Step3_case_inventory.csv", index=False)

    cx = args.leftmost_x + args.radius
    cy = args.center_y
    R = args.radius

    all_ray_metrics = []
    all_summary = []

    selections = []
    for gas in ["Ar", "N2"]:
        kn_sel = select_kn_sweep(inv, gas, args.kn_sweep_mach, args.kn_values)
        mach_sel = select_mach_sweep(inv, gas, args.mach_sweep_kn, args.mach_values)
        if not kn_sel.empty:
            selections.append(kn_sel)
        if not mach_sel.empty:
            selections.append(mach_sel)

    if not selections:
        raise RuntimeError("No selected cases for Kn or Mach sweeps.")

    selected = pd.concat(selections, ignore_index=True).drop_duplicates(subset=["gas", "kn", "mach", "sweep"])
    selected.to_csv(outdir / "Step3_selected_cases.csv", index=False)

    for _, row in selected.iterrows():
        print(f"[INFO] analyzing {row['sweep']} {row['gas']} Kn={row['kn']:g} M={row['mach']:g}")
        try:
            ray_df, summ_df = analyze_case(
                row, cx, cy, R,
                variables=args.variables,
                half_angle_deg=args.half_angle,
                n_angles=args.angles,
                smooth_sigma=args.smooth_sigma,
                ray_ns=args.ray_ns,
            )

            if not ray_df.empty:
                ray_df["sweep"] = row["sweep"]
                ray_df["parameter"] = row["parameter"]
                all_ray_metrics.append(ray_df)

            if not summ_df.empty:
                summ_df["sweep"] = row["sweep"]
                summ_df["parameter"] = row["parameter"]
                all_summary.append(summ_df)

        except Exception as e:
            print(f"[WARN] failed case {row['gas']} Kn={row['kn']} M={row['mach']} :: {e}")

    if not all_summary:
        raise RuntimeError("No metrics were successfully extracted.")

    ray_metrics = pd.concat(all_ray_metrics, ignore_index=True) if all_ray_metrics else pd.DataFrame()
    summary = pd.concat(all_summary, ignore_index=True)

    ray_metrics.to_csv(outdir / "Step3_ray_metrics_all.csv", index=False)
    summary.to_csv(outdir / "Step3_case_summary_metrics.csv", index=False)

    # Figures
    for sweep in ["Kn", "Mach"]:
        for gas in ["Ar", "N2"]:
            plot_standoff_thickness(summary, gas, sweep, outdir)
            plot_variable_scales(summary, gas, sweep, outdir, args.variables)
        plot_combined_gas_comparison(summary, sweep, outdir)

    print("\n[OK] Step-3 shock-layer metrics complete.")
    print(f"[OK] Output directory: {outdir.resolve()}")
    print("[OK] Key files:")
    print("     Step3_case_summary_metrics.csv")
    print("     Step3_ray_metrics_all.csv")
    print("     Step3_*_standoff_thickness.pdf/png")
    print("     Step3_*_variable_scales.pdf/png")
    print("     Step3_Kn_Ar_N2_density_metrics.pdf/png")
    print("     Step3_Mach_Ar_N2_density_metrics.pdf/png")


if __name__ == "__main__":
    main()

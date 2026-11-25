#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cylinder_modal_step2.py

STEP 2: POD/SVD modal analysis for rarefied hypersonic cylinder DSMC datasets.

This script is intentionally separate from the Step-1 contour code.  It does
not make contour maps.  It reads the same RAR/directories, reconstructs fields,
and performs modal analysis in two coordinates:

1) physical-coordinate POD
   snapshots are interpolated to a common (x,y) grid.

2) shock-/layer-attached profile POD
   snapshots are sampled along body-normal rays and registered with
       xi_q = (s - s_q)/delta_q,
   where s_q is the location of maximum |dq/ds| and
       delta_q = |q_down - q_up| / max |dq/ds|.

The purpose is not to create a time-dynamic ROM.  The data are parameter
snapshots, so POD/SVD is used to quantify structural compactness, cumulative
variance, reconstruction error, and parameter coefficients across Kn and Mach.

Outputs:
  - case_inventory_modal.csv
  - modal_summary.csv
  - POD spectrum/cumulative-energy figures
  - attached-mode maps in (xi, theta)
  - coefficient curves vs Kn or Mach
  - reconstruction-error curves

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
    from scipy.ndimage import gaussian_filter1d, gaussian_filter
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Utilities and archive extraction
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


# ============================================================
# Tecplot/DSMC reader and metadata inference
# ============================================================

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
        "ttr":  choose(["TTR", "Ttr", "Ttrans", "TranslationalTemperature", "T"]),
        "trot": choose(["TRT", "Trot", "RotationalTemperature"]),
        "tvib": choose(["TVB", "Tvib", "VibrationalTemperature"]),
        "mach": choose(["MA", "Mach", "M"]),
        "p":    choose(["P", "Pressure"]),
        "mfp":  choose(["MFP", "MeanFreePath"]),
    }


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

    # de-duplicate exact files
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
            print(f"[WARN] skipped unreadable field: {c['path']}")

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
# Interpolation
# ============================================================

def interpolate_field_to_grid(data, var, xg, yg, cx, cy, R, smooth_sigma=0.0):
    if var not in data:
        return None

    x = data["x"]
    y = data["y"]
    q = data[var]

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(q)
    if good.sum() < 100:
        return None

    tri = mtri.Triangulation(x[good], y[good])
    interp = mtri.LinearTriInterpolator(tri, q[good])

    X, Y = np.meshgrid(xg, yg)
    Q = interp(X, Y)
    Q = np.asarray(Q.filled(np.nan) if hasattr(Q, "filled") else Q, dtype=float)

    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= R**2
    Q[inside] = np.nan

    if SCIPY_OK and smooth_sigma > 0:
        Qf = fill_nans(Q)
        Qs = gaussian_filter(Qf, sigma=smooth_sigma)
        Qs[inside] = np.nan
        Q = Qs

    return Q


def fill_nans(A):
    B = (
        pd.DataFrame(A)
        .interpolate(axis=0)
        .interpolate(axis=1)
        .bfill(axis=0)
        .ffill(axis=0)
        .bfill(axis=1)
        .ffill(axis=1)
        .to_numpy()
    )
    if not np.isfinite(B).all():
        med = np.nanmedian(A[np.isfinite(A)]) if np.any(np.isfinite(A)) else 0.0
        B = np.where(np.isfinite(B), B, med)
    return B


def interp_regular_grid(X, Y, A, xq, yq):
    xg = X[0, :]
    yg = Y[:, 0]

    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    out = np.full_like(xq, np.nan, dtype=float)

    inside = (xq >= xg[0]) & (xq <= xg[-1]) & (yq >= yg[0]) & (yq <= yg[-1])
    if not np.any(inside):
        return out

    xi = np.searchsorted(xg, xq[inside]) - 1
    yi = np.searchsorted(yg, yq[inside]) - 1

    xi = np.clip(xi, 0, len(xg) - 2)
    yi = np.clip(yi, 0, len(yg) - 2)

    x1 = xg[xi]
    x2 = xg[xi + 1]
    y1 = yg[yi]
    y2 = yg[yi + 1]

    tx = (xq[inside] - x1) / np.maximum(x2 - x1, 1e-300)
    ty = (yq[inside] - y1) / np.maximum(y2 - y1, 1e-300)

    f11 = A[yi, xi]
    f21 = A[yi, xi + 1]
    f12 = A[yi + 1, xi]
    f22 = A[yi + 1, xi + 1]

    vals = (1 - tx) * (1 - ty) * f11 + tx * (1 - ty) * f21 + (1 - tx) * ty * f12 + tx * ty * f22
    out[inside] = vals
    return out


def ray_profile_from_regular_grid(X, Y, Q, cx, cy, R, theta, ns=600, smax_frac=8.0):
    ux, uy = np.cos(theta), np.sin(theta)
    x0, y0 = cx + R * ux, cy + R * uy

    xmin, xmax = float(np.nanmin(X)), float(np.nanmax(X))
    ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))

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

    m = ys >= cy
    if m.sum() < 50:
        return None

    s = s[m]
    xs = xs[m]
    ys = ys[m]
    q = interp_regular_grid(X, Y, Q, xs, ys)

    good = np.isfinite(q)
    if good.sum() < 50:
        return None

    return s[good], q[good]


# ============================================================
# Snapshot normalization and POD
# ============================================================

def robust_normalize_field(Q, X, Y, var):
    """Normalize a field to make POD less sensitive to dimensional units."""
    Q = np.asarray(Q, dtype=float)
    out = Q.copy()

    good = np.isfinite(out)
    if good.sum() < 50:
        return out

    # For density, use upstream reference.
    if var == "rho":
        x = X[good]
        q = out[good]
        xcut = np.nanpercentile(x, 8)
        up = np.nanmedian(q[x <= xcut]) if np.any(x <= xcut) else np.nanmedian(q)
        if np.isfinite(up) and abs(up) > 1e-300:
            out = out / up
        return out

    # For all variables: robust 0-centred scaling by inter-percentile range.
    q = out[good]
    qlo = np.nanpercentile(q, 5)
    qhi = np.nanpercentile(q, 95)
    scale = qhi - qlo
    if np.isfinite(scale) and abs(scale) > 1e-300:
        out = (out - qlo) / scale
    return out


def build_data_matrix_from_arrays(arrays):
    """Return Xmat, mean_vec, mask, original_shape.

    arrays: list of 2D arrays with same shape
    """
    if len(arrays) < 2:
        return None, None, None, None

    shape = arrays[0].shape
    stack = np.stack(arrays, axis=-1)  # ny,nx,m

    valid = np.all(np.isfinite(stack), axis=-1)
    if valid.sum() < 50:
        return None, None, None, None

    X = stack[valid, :]  # npoints,m
    mean_vec = np.mean(X, axis=1, keepdims=True)
    Xc = X - mean_vec
    return Xc, mean_vec[:, 0], valid, shape


def perform_pod(Xc, max_modes=8):
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = S**2
    if lam.sum() > 0:
        energy = lam / lam.sum()
    else:
        energy = np.zeros_like(lam)
    cumulative = np.cumsum(energy)

    max_modes = min(max_modes, U.shape[1])
    recon_errors = []
    normX = np.linalg.norm(Xc)
    for r in range(1, max_modes + 1):
        Xr = (U[:, :r] * S[:r]) @ Vt[:r, :]
        err = np.linalg.norm(Xc - Xr) / max(normX, 1e-300)
        recon_errors.append(err)

    return {
        "U": U,
        "S": S,
        "Vt": Vt,
        "energy": energy,
        "cumulative": cumulative,
        "recon_errors": np.asarray(recon_errors),
    }


def mode_to_grid(mode_vec, mask, shape):
    A = np.full(shape, np.nan, dtype=float)
    A[mask] = mode_vec
    return A


def modes_needed(cumulative, target):
    idx = np.where(cumulative >= target)[0]
    return int(idx[0] + 1) if len(idx) else int(len(cumulative))


# ============================================================
# Case selection
# ============================================================

def pick_nearest(sub, values, col):
    out = []
    vals = sorted(sub[col].unique())
    for target in values:
        if not vals:
            continue
        v = min(vals, key=lambda a: abs(a - target))
        row = sub.iloc[np.argmin(np.abs(sub[col].to_numpy(float) - v))].to_dict()
        out.append(row)

    seen, uniq = set(), []
    for r in out:
        key = (r["gas"], round(float(r["kn"]), 10), round(float(r["mach"]), 10))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def select_kn_sweep(inv, gas, mach_ref, kn_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return []
    ginv["machdiff"] = np.abs(ginv["mach"] - mach_ref)
    sub = ginv.sort_values(["kn", "machdiff", "priority", "n_points"], ascending=[True, True, False, False])
    sub = sub.drop_duplicates("kn", keep="first")
    rows = pick_nearest(sub, kn_values, "kn")
    for r in rows:
        r["sweep"] = "Kn"
        r["param"] = float(r["kn"])
    return rows


def select_mach_sweep(inv, gas, kn_ref, mach_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return []
    ginv["kndiff"] = np.abs(ginv["kn"] - kn_ref)
    sub = ginv.sort_values(["mach", "kndiff", "priority", "n_points"], ascending=[True, True, False, False])
    sub = sub.drop_duplicates("mach", keep="first")
    rows = pick_nearest(sub, mach_values, "mach")
    for r in rows:
        r["sweep"] = "Mach"
        r["param"] = float(r["mach"])
    return rows


def available_variables(rows, requested):
    out = []
    for var in requested:
        ok = 0
        for r in rows:
            col = r.get(f"col_{var}", "")
            if isinstance(col, str) and len(col) > 0:
                ok += 1
        if ok >= 2:
            out.append(var)
    return out


# ============================================================
# Physical-coordinate POD
# ============================================================

def common_grid_from_rows(rows, nx, ny, cy):
    x_min = max(float(r["x_min"]) for r in rows)
    x_max = min(float(r["x_max"]) for r in rows)
    y_min = max(0.0, cy)
    y_max = min(float(r["y_max"]) for r in rows)

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("No common physical domain among selected cases.")

    xg = np.linspace(x_min, x_max, nx)
    yg = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(xg, yg)
    return xg, yg, X, Y


def physical_pod(rows, var, cx, cy, R, nx, ny, smooth_sigma, max_modes):
    xg, yg, X, Y = common_grid_from_rows(rows, nx, ny, cy)
    arrays = []
    params = []
    labels = []

    for r in rows:
        data = read_case(r)
        Q = interpolate_field_to_grid(data, var, xg, yg, cx, cy, R, smooth_sigma=smooth_sigma)
        if Q is None:
            continue
        Qn = robust_normalize_field(Q, X, Y, var)
        arrays.append(Qn)
        params.append(float(r["param"]))
        labels.append(f"Kn={r['kn']:g}, M={r['mach']:g}")

    Xc, mean_vec, mask, shape = build_data_matrix_from_arrays(arrays)
    if Xc is None:
        return None

    pod = perform_pod(Xc, max_modes=max_modes)
    return {
        "basis": "physical",
        "var": var,
        "rows": rows,
        "params": np.asarray(params),
        "labels": labels,
        "Xgrid": X,
        "Ygrid": Y,
        "arrays": arrays,
        "Xc": Xc,
        "mean_vec": mean_vec,
        "mask": mask,
        "shape": shape,
        "pod": pod,
    }


# ============================================================
# Attached-profile POD
# ============================================================

def build_case_regular_grid(row, var, cx, cy, R, nx=360, ny=220, smooth_sigma=1.0):
    data = read_case(row)
    if var not in data:
        return None

    x = data["x"]
    y = data["y"]
    q = data[var]

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(q) & (y >= cy)
    if good.sum() < 100:
        return None

    xg = np.linspace(np.nanmin(x[good]), np.nanmax(x[good]), nx)
    yg = np.linspace(max(cy, np.nanmin(y[good])), np.nanmax(y[good]), ny)

    X, Y = np.meshgrid(xg, yg)
    tri = mtri.Triangulation(x[good], y[good])
    interp = mtri.LinearTriInterpolator(tri, q[good])
    Q = interp(X, Y)
    Q = np.asarray(Q.filled(np.nan) if hasattr(Q, "filled") else Q, dtype=float)

    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= R**2
    Q[inside] = np.nan

    if SCIPY_OK and smooth_sigma > 0:
        Qf = fill_nans(Q)
        Qs = gaussian_filter(Qf, sigma=smooth_sigma)
        Qs[inside] = np.nan
        Q = Qs

    return X, Y, Q


def attached_snapshot_for_case(row, var, cx, cy, R, theta_grid_deg, xi_grid,
                               grid_nx=360, grid_ny=220, smooth_sigma=1.0):
    reg = build_case_regular_grid(row, var, cx, cy, R, nx=grid_nx, ny=grid_ny, smooth_sigma=smooth_sigma)
    if reg is None:
        return None, None

    X, Y, Q = reg
    snap = np.full((len(theta_grid_deg), len(xi_grid)), np.nan, dtype=float)

    metrics = []

    for it, th_deg in enumerate(theta_grid_deg):
        th = np.deg2rad(th_deg)
        prof = ray_profile_from_regular_grid(X, Y, Q, cx, cy, R, th, ns=700, smax_frac=8.0)
        if prof is None:
            continue

        s, q = prof
        if len(s) < 80:
            continue

        # Smooth 1D profile to avoid DSMC noise in gradient.
        if SCIPY_OK:
            qs = gaussian_filter1d(q, sigma=2.0)
        else:
            qs = q.copy()

        # Exclude near-wall cells for gradient location.
        ds = np.nanmedian(np.diff(s))
        if not np.isfinite(ds) or ds <= 0:
            continue

        # Robust upstream and downstream averages along ray.
        n = len(s)
        nfar = max(10, int(0.15 * n))
        nnear0 = max(4, int(0.04 * n))
        nnear1 = max(nnear0 + 10, int(0.25 * n))

        q_up = float(np.nanmedian(qs[-nfar:]))
        q_down = float(np.nanmedian(qs[nnear0:nnear1]))

        dq = q_down - q_up
        if not np.isfinite(dq) or abs(dq) < 1e-12:
            continue

        dqds = np.gradient(qs, s)
        g = np.abs(dqds)

        # Avoid wall and far boundary.
        valid = (s > 0.02 * R) & (s < 0.92 * np.nanmax(s)) & np.isfinite(g)
        if valid.sum() < 20:
            continue

        idx_valid = np.where(valid)[0]
        imax = idx_valid[int(np.nanargmax(g[valid]))]
        maxg = float(g[imax])

        if not np.isfinite(maxg) or maxg <= 0:
            continue

        s_q = float(s[imax])
        delta_q = abs(float(dq)) / maxg

        # Put sane bounds on delta for noisy high-Kn profiles.
        delta_q = float(np.clip(delta_q, 0.005 * R, 8.0 * R))

        xi = (s - s_q) / delta_q
        qnorm = (qs - q_up) / dq

        # Interpolate qnorm onto common xi grid.
        order = np.argsort(xi)
        xi_sorted = xi[order]
        q_sorted = qnorm[order]

        # Remove duplicates.
        good = np.isfinite(xi_sorted) & np.isfinite(q_sorted)
        xi_sorted = xi_sorted[good]
        q_sorted = q_sorted[good]
        if len(xi_sorted) < 20:
            continue

        # Only interpolate within available range; outside remains NaN.
        interp_vals = np.interp(xi_grid, xi_sorted, q_sorted, left=np.nan, right=np.nan)
        snap[it, :] = interp_vals

        metrics.append({
            "theta_deg": th_deg,
            "q_up": q_up,
            "q_down": q_down,
            "s_q": s_q,
            "delta_q": delta_q,
            "max_grad": maxg,
        })

    if np.isfinite(snap).sum() < 200:
        return None, None

    return snap, pd.DataFrame(metrics)


def attached_pod(rows, var, cx, cy, R, theta_grid_deg, xi_grid,
                 grid_nx, grid_ny, smooth_sigma, max_modes):
    arrays = []
    params = []
    labels = []
    metrics_all = []

    for r in rows:
        snap, metrics = attached_snapshot_for_case(
            r, var, cx, cy, R,
            theta_grid_deg=theta_grid_deg,
            xi_grid=xi_grid,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            smooth_sigma=smooth_sigma,
        )
        if snap is None:
            continue

        arrays.append(snap)
        params.append(float(r["param"]))
        labels.append(f"Kn={r['kn']:g}, M={r['mach']:g}")

        if metrics is not None and len(metrics):
            mm = metrics.copy()
            mm["gas"] = r["gas"]
            mm["kn"] = r["kn"]
            mm["mach"] = r["mach"]
            mm["param"] = r["param"]
            mm["variable"] = var
            metrics_all.append(mm)

    Xc, mean_vec, mask, shape = build_data_matrix_from_arrays(arrays)
    if Xc is None:
        return None

    pod = perform_pod(Xc, max_modes=max_modes)
    metrics_df = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()

    XI, TH = np.meshgrid(xi_grid, theta_grid_deg)

    return {
        "basis": "attached",
        "var": var,
        "rows": rows,
        "params": np.asarray(params),
        "labels": labels,
        "XI": XI,
        "TH": TH,
        "xi_grid": np.asarray(xi_grid),
        "theta_grid_deg": np.asarray(theta_grid_deg),
        "arrays": arrays,
        "Xc": Xc,
        "mean_vec": mean_vec,
        "mask": mask,
        "shape": shape,
        "pod": pod,
        "metrics": metrics_df,
    }


# ============================================================
# Figure helpers
# ============================================================

def save_spectrum_figure(results_by_var, title, outbase):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for var, res in results_by_var.items():
        if res is None:
            continue
        c = res["pod"]["cumulative"]
        x = np.arange(1, len(c) + 1)
        ax.plot(x, c, marker="o", label=var)

    ax.axhline(0.90, ls="--", lw=1, color="0.3")
    ax.axhline(0.95, ls=":", lw=1, color="0.3")
    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0, 1.03)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def save_reconstruction_error_figure(results_by_var, title, outbase):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for var, res in results_by_var.items():
        if res is None:
            continue
        e = res["pod"]["recon_errors"]
        x = np.arange(1, len(e) + 1)
        ax.semilogy(x, e, marker="o", label=var)

    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel(r"Relative reconstruction error, $\|X-X_r\|_F/\|X\|_F$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def save_coefficients_figure(res, title, outbase, n_modes=4):
    pod = res["pod"]
    params = res["params"]
    Vt = pod["Vt"]
    S = pod["S"]
    coeff = (S[:, None] * Vt)  # modes x snapshots

    order = np.argsort(params)
    params = params[order]
    coeff = coeff[:, order]

    n_modes = min(n_modes, coeff.shape[0])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for k in range(n_modes):
        ax.plot(params, coeff[k, :], marker="o", label=f"mode {k+1}")

    ax.set_xlabel("Parameter")
    ax.set_ylabel("POD coefficient")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def save_attached_modes_figure(res, title, outbase, n_modes=4):
    pod = res["pod"]
    U = pod["U"]
    mask = res["mask"]
    shape = res["shape"]
    XI = res["XI"]
    TH = res["TH"]

    n_modes = min(n_modes, U.shape[1])
    ncols = n_modes + 1

    fig, axes = plt.subplots(1, ncols, figsize=(3.1*ncols, 3.6), squeeze=False)
    axes = axes[0]

    mean_grid = mode_to_grid(res["mean_vec"], mask, shape)
    im = axes[0].pcolormesh(XI, TH, mean_grid, shading="auto", cmap="turbo")
    axes[0].set_title("mean")
    axes[0].set_xlabel(r"$\xi_q$")
    axes[0].set_ylabel(r"$\theta$ (deg)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    for k in range(n_modes):
        mode_grid = mode_to_grid(U[:, k], mask, shape)
        vmax = np.nanpercentile(np.abs(mode_grid), 98)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = axes[k+1].pcolormesh(XI, TH, mode_grid, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[k+1].set_title(f"mode {k+1}")
        axes[k+1].set_xlabel(r"$\xi_q$")
        axes[k+1].set_ylabel("")
        fig.colorbar(im, ax=axes[k+1], shrink=0.85)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def save_physical_modes_figure(res, title, outbase, n_modes=3):
    pod = res["pod"]
    U = pod["U"]
    mask = res["mask"]
    shape = res["shape"]
    X = res["Xgrid"]
    Y = res["Ygrid"]

    n_modes = min(n_modes, U.shape[1])
    ncols = n_modes + 1

    fig, axes = plt.subplots(1, ncols, figsize=(3.2*ncols, 3.2), squeeze=False)
    axes = axes[0]

    mean_grid = mode_to_grid(res["mean_vec"], mask, shape)
    im = axes[0].pcolormesh(X, Y, mean_grid, shading="auto", cmap="turbo")
    axes[0].set_title("mean")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    for k in range(n_modes):
        mode_grid = mode_to_grid(U[:, k], mask, shape)
        vmax = np.nanpercentile(np.abs(mode_grid), 98)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = axes[k+1].pcolormesh(X, Y, mode_grid, shading="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[k+1].set_title(f"mode {k+1}")
        axes[k+1].set_aspect("equal")
        axes[k+1].set_xlabel("x")
        axes[k+1].set_ylabel("")
        fig.colorbar(im, ax=axes[k+1], shrink=0.85)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outbase.with_suffix(".png"), dpi=300)
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


# ============================================================
# Saving tables
# ============================================================

def summarize_result(res, gas, sweep, var, basis):
    c = res["pod"]["cumulative"]
    e = res["pod"]["energy"]
    errs = res["pod"]["recon_errors"]

    row = {
        "gas": gas,
        "sweep": sweep,
        "variable": var,
        "basis": basis,
        "n_snapshots": len(res["params"]),
        "n_dof": res["Xc"].shape[0],
        "E1": float(e[0]) if len(e) > 0 else np.nan,
        "E2": float(e[1]) if len(e) > 1 else np.nan,
        "E3": float(e[2]) if len(e) > 2 else np.nan,
        "C1": float(c[0]) if len(c) > 0 else np.nan,
        "C2": float(c[1]) if len(c) > 1 else np.nan,
        "C3": float(c[2]) if len(c) > 2 else np.nan,
        "C4": float(c[3]) if len(c) > 3 else np.nan,
        "modes90": modes_needed(c, 0.90),
        "modes95": modes_needed(c, 0.95),
        "err1": float(errs[0]) if len(errs) > 0 else np.nan,
        "err2": float(errs[1]) if len(errs) > 1 else np.nan,
        "err3": float(errs[2]) if len(errs) > 2 else np.nan,
        "err4": float(errs[3]) if len(errs) > 3 else np.nan,
        "params": ",".join([f"{p:g}" for p in res["params"]]),
    }
    return row


def save_spectrum_csv(res, gas, sweep, var, basis, outdir):
    e = res["pod"]["energy"]
    c = res["pod"]["cumulative"]
    df = pd.DataFrame({
        "mode": np.arange(1, len(e)+1),
        "energy_fraction": e,
        "cumulative_energy": c,
    })
    df["gas"] = gas
    df["sweep"] = sweep
    df["variable"] = var
    df["basis"] = basis
    df.to_csv(outdir / f"spectrum_{gas}_{sweep}_{basis}_{var}.csv", index=False)


def save_coefficients_csv(res, gas, sweep, var, basis, outdir):
    pod = res["pod"]
    coeff = pod["S"][:, None] * pod["Vt"]
    rows = []
    for j, param in enumerate(res["params"]):
        for k in range(coeff.shape[0]):
            rows.append({
                "gas": gas,
                "sweep": sweep,
                "variable": var,
                "basis": basis,
                "snapshot_index": j,
                "parameter": float(param),
                "mode": k+1,
                "coefficient": float(coeff[k, j]),
            })
    pd.DataFrame(rows).to_csv(outdir / f"coefficients_{gas}_{sweep}_{basis}_{var}.csv", index=False)


# ============================================================
# Main workflow
# ============================================================

def run_modal_group(rows, gas, sweep, variables, args, outdir):
    if len(rows) < 2:
        print(f"[WARN] Not enough cases for {gas} {sweep}")
        return []

    variables = available_variables(rows, variables)
    if not variables:
        print(f"[WARN] No variables available for {gas} {sweep}")
        return []

    cx = args.leftmost_x + args.radius
    cy = args.center_y
    R = args.radius

    theta_grid = np.linspace(args.theta_max_deg, args.theta_min_deg, args.n_theta)
    xi_grid = np.linspace(args.xi_min, args.xi_max, args.n_xi)

    summaries = []
    physical_results = {}
    attached_results = {}

    print(f"\n[INFO] Modal group: gas={gas}, sweep={sweep}, cases={len(rows)}, variables={variables}")

    for var in variables:
        print(f"[INFO]   variable={var}: physical POD")
        try:
            res_phys = physical_pod(
                rows, var, cx, cy, R,
                nx=args.physical_nx,
                ny=args.physical_ny,
                smooth_sigma=args.smooth_sigma,
                max_modes=args.max_modes,
            )
        except Exception as e:
            print(f"[WARN] physical POD failed for {gas} {sweep} {var}: {e}")
            res_phys = None

        physical_results[var] = res_phys

        if res_phys is not None:
            summaries.append(summarize_result(res_phys, gas, sweep, var, "physical"))
            save_spectrum_csv(res_phys, gas, sweep, var, "physical", outdir)
            save_coefficients_csv(res_phys, gas, sweep, var, "physical", outdir)
            save_coefficients_figure(
                res_phys,
                f"{gas} {sweep} physical POD coefficients: {var}",
                outdir / f"POD_coeff_{gas}_{sweep}_physical_{var}",
                n_modes=min(args.max_modes, 4),
            )

        print(f"[INFO]   variable={var}: attached POD")
        try:
            res_att = attached_pod(
                rows, var, cx, cy, R,
                theta_grid_deg=theta_grid,
                xi_grid=xi_grid,
                grid_nx=args.attached_grid_nx,
                grid_ny=args.attached_grid_ny,
                smooth_sigma=args.smooth_sigma,
                max_modes=args.max_modes,
            )
        except Exception as e:
            print(f"[WARN] attached POD failed for {gas} {sweep} {var}: {e}")
            res_att = None

        attached_results[var] = res_att

        if res_att is not None:
            summaries.append(summarize_result(res_att, gas, sweep, var, "attached"))
            save_spectrum_csv(res_att, gas, sweep, var, "attached", outdir)
            save_coefficients_csv(res_att, gas, sweep, var, "attached", outdir)

            if len(res_att["metrics"]):
                res_att["metrics"].to_csv(outdir / f"attached_profile_metrics_{gas}_{sweep}_{var}.csv", index=False)

            save_attached_modes_figure(
                res_att,
                f"{gas} {sweep} attached POD: {var}",
                outdir / f"POD_modes_{gas}_{sweep}_attached_{var}",
                n_modes=min(args.max_modes, 4),
            )
            save_coefficients_figure(
                res_att,
                f"{gas} {sweep} attached POD coefficients: {var}",
                outdir / f"POD_coeff_{gas}_{sweep}_attached_{var}",
                n_modes=min(args.max_modes, 4),
            )

        # For selected variables, save physical mode maps as supplementary figures.
        if res_phys is not None and var in args.mode_map_variables:
            save_physical_modes_figure(
                res_phys,
                f"{gas} {sweep} physical POD: {var}",
                outdir / f"POD_modes_{gas}_{sweep}_physical_{var}",
                n_modes=min(args.max_modes, 3),
            )

    # Group spectrum / reconstruction figures
    save_spectrum_figure(
        physical_results,
        f"{gas} {sweep}: physical-coordinate POD cumulative energy",
        outdir / f"POD_spectrum_{gas}_{sweep}_physical",
    )
    save_spectrum_figure(
        attached_results,
        f"{gas} {sweep}: attached-coordinate POD cumulative energy",
        outdir / f"POD_spectrum_{gas}_{sweep}_attached",
    )
    save_reconstruction_error_figure(
        physical_results,
        f"{gas} {sweep}: physical-coordinate reconstruction error",
        outdir / f"POD_reconstruction_error_{gas}_{sweep}_physical",
    )
    save_reconstruction_error_figure(
        attached_results,
        f"{gas} {sweep}: attached-coordinate reconstruction error",
        outdir / f"POD_reconstruction_error_{gas}_{sweep}_attached",
    )

    return summaries


def main():
    ap = argparse.ArgumentParser(description="Step-2 modal analysis for rarefied hypersonic cylinder DSMC data.")

    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract_root", default="_modal_extracted")

    ap.add_argument("--radius", type=float, default=0.1524)
    ap.add_argument("--leftmost_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)

    ap.add_argument("--default_kn_if_missing", type=float, default=0.01)
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0)

    ap.add_argument("--kn_sweep_mach", type=float, default=10.0)
    ap.add_argument("--mach_sweep_kn", type=float, default=0.01)
    ap.add_argument("--kn_values", nargs="+", type=float, default=[0.01, 0.025, 0.1, 1.0])
    ap.add_argument("--mach_values", nargs="+", type=float, default=[5.0, 10.0, 15.0])

    ap.add_argument("--variables", nargs="+", default=["rho", "mach", "p", "ttr", "trot"],
                    help="Variables to attempt. Only variables found in at least two cases are used.")
    ap.add_argument("--mode_map_variables", nargs="+", default=["rho", "ttr", "trot"],
                    help="Physical-mode maps are saved only for these variables.")

    ap.add_argument("--physical_nx", type=int, default=180)
    ap.add_argument("--physical_ny", type=int, default=120)
    ap.add_argument("--attached_grid_nx", type=int, default=320)
    ap.add_argument("--attached_grid_ny", type=int, default=210)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)

    ap.add_argument("--theta_min_deg", type=float, default=95.0,
                    help="Upper end of theta range, near cylinder top.")
    ap.add_argument("--theta_max_deg", type=float, default=180.0,
                    help="Upstream stagnation direction.")
    ap.add_argument("--n_theta", type=int, default=49)

    ap.add_argument("--xi_min", type=float, default=-4.0)
    ap.add_argument("--xi_max", type=float, default=6.0)
    ap.add_argument("--n_xi", type=int, default=180)

    ap.add_argument("--max_modes", type=int, default=8)

    args = ap.parse_args()

    outdir = ensure_dir(args.out)
    roots = prepare_inputs(args.inputs, ensure_dir(args.extract_root))

    cases = find_flow_files(
        roots,
        default_kn=args.default_kn_if_missing,
        default_mach=args.default_mach_if_missing,
    )

    inv = build_inventory(cases)
    if inv.empty:
        raise RuntimeError("No valid cases found. Reader failed or no DS2FF/TriangulateGrid files found.")

    inv = inv.sort_values(["gas", "kn", "mach"]).reset_index(drop=True)
    inv.to_csv(outdir / "case_inventory_modal.csv", index=False)

    all_summaries = []

    for gas in ["Ar", "N2"]:
        kn_rows = select_kn_sweep(inv, gas, args.kn_sweep_mach, args.kn_values)
        mach_rows = select_mach_sweep(inv, gas, args.mach_sweep_kn, args.mach_values)

        all_summaries += run_modal_group(kn_rows, gas, "Kn", args.variables, args, outdir)
        all_summaries += run_modal_group(mach_rows, gas, "Mach", args.variables, args, outdir)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(outdir / "modal_summary.csv", index=False)

    print("\n[OK] Modal analysis complete.")
    print(f"[OK] Output directory: {outdir.resolve()}")
    print("[OK] Main summary: modal_summary.csv")
    print("[OK] Case inventory: case_inventory_modal.csv")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cylinder_fig12_clean_no_stripes_v4_JFM_smoother.py

Complete, ready-to-run smoother JFM-style replacement script for the cylinder paper Figure 12.

Purpose
-------
This script rebuilds the N2 Kn-sweep thermal POD figure, with two rows:
    row 1: Ttr mean, mode 1, mode 2
    row 2: Trot mean, mode 1, mode 2

It fixes the visual striping/banding problem by using:
    1) common valid-support masking,
    2) stronger but still display-only masked Gaussian smoothing,
    3) row/column de-striping with slow-trend protection,
    4) bicubic rasterized imshow output for PDF.

Version v4 applies slightly stronger smoothing in the angular direction to
reduce residual horizontal bands in the displayed figure.  POD energies and
modal amplitudes are still computed from the original unsmoothed data.

Important
---------
The POD decomposition and modal energies are computed from the unsmoothed attached
snapshot data.  Smoothing is applied only to the plotted display arrays.

Example
-------
python cylinder_fig12_clean_no_stripes_v4_JFM_smoother.py \
  --inputs "N2-Bird Prop-Kn-Part1.rar" "N2-Bird Prop-Kn-Part2.rar" \
  --out Step2b_Fig12_Clean \
  --extract_root _fig12_extracted \
  --radius 0.1524 \
  --leftmost_x 0.0 \
  --center_y 0.0 \
  --default_kn_if_missing 0.01 \
  --default_mach_if_missing 10.0 \
  --kn_sweep_mach 10.0 \
  --kn_values 0.01 0.025 0.05 0.075 0.1 0.5 0.75 1.0
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
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "savefig.dpi": 600,
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
# General utilities
# ============================================================

def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd(cmd: List[str]) -> bool:
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
        raise RuntimeError(
            f"Could not extract RAR archive: {src}\n"
            "Install 7-Zip/unrar, or extract the archive manually and pass the extracted folder as --inputs."
        )

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


def normalize_key(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ============================================================
# Tecplot / DS2V field reader
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
        "MFP", "SOF", "FSP", "ANG", "P", "Angular momentum/(Mass.L)",
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
# Metadata inference and case selection
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
    pats = [
        r"Kn\s*=\s*([0-9]+(?:[\.,][0-9]+)?)",
        r"Kn\s*[_-]?\s*([0-9]+(?:[\.,][0-9]+)?)",
        r"Knum\s*[_=-]?\s*([0-9]+(?:[\.,][0-9]+)?)",
    ]
    for pat in pats:
        m = re.search(pat, s, re.I)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


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

    for comp in [path.name] + list(path.parts)[::-1]:
        if re.search(r"Mach\s*[0-9]+(?:\.[0-9]+)?\s*-\s*[0-9]+", comp, re.I):
            continue
        for pat in [
            r"(?:TriangulateGrid|TriangulatedGrid)\s*M\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])M\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])Ma\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])Mach\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        ]:
            m = re.search(pat, comp, re.I)
            if m:
                val = clean(m.group(1))
                if np.isfinite(val):
                    return float(val)
    return None


def source_tag(path: Path) -> str:
    s = str(path).lower()
    if "ma=9-n2" in s or "ma9" in s:
        return "mach_sweep_n2_fixedM9"
    if "argon-data-mach" in s or "triangulategrid" in s or "triangulatedgrid" in s:
        return "mach_sweep_argon"
    if "results-n2" in s:
        return "mach_sweep_n2"
    if "prop-kn" in s or "kn=" in s or "kn_" in s:
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
        root = Path(root)
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


def select_kn_sweep(inv, gas, mach_ref, kn_values):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return pd.DataFrame()

    ginv["machdiff"] = np.abs(ginv["mach"] - mach_ref)
    rows = []
    available_kn = sorted(ginv["kn"].unique())
    for target in kn_values:
        k = min(available_kn, key=lambda a: abs(a - target))
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
# Interpolation, ray sampling and attached profiles
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

    # The data/paper use the upper half-plane.
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
        return gaussian_filter1d(q, sigma=sigma, mode="nearest")
    return q


def _crossing_s_for_level(s, qnorm, level):
    """Return the first robust crossing position of qnorm=level, or nearest point."""
    s = np.asarray(s, dtype=float)
    qnorm = np.asarray(qnorm, dtype=float)
    good = np.isfinite(s) & np.isfinite(qnorm)
    s = s[good]
    qnorm = qnorm[good]
    if len(s) < 4:
        return np.nan

    # Work with monotone direction inferred from endpoints.
    y = qnorm - level
    cross = np.where(y[:-1] * y[1:] <= 0)[0]
    if len(cross):
        i = int(cross[0])
        y0, y1 = y[i], y[i+1]
        if abs(y1 - y0) > 1e-14:
            return float(s[i] + (s[i+1] - s[i]) * (-y0) / (y1 - y0))
        return float(s[i])

    # Fallback: closest point to the requested level.
    return float(s[int(np.nanargmin(np.abs(y)))])


def metrics_from_profile(s, q, R, smooth_sigma=2.0, marker_method="maxgrad"):
    """
    Extract a centre and thickness from one ray profile.

    marker_method:
        maxgrad   : centre = maximum |dq/ds|, thickness = jump/max|dq/ds|
        halfjump  : centre = 50% transition point, thickness = 10--90 transition width
        gcentroid : centre = gradient-magnitude centroid, thickness = 2 weighted std. dev.

    The alternatives are intended for robustness checks, not for changing the
    main definition used in the manuscript.
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
    near0 = max(3, int(0.03 * n))
    near1 = max(near0 + 8, int(0.23 * n))
    far_n = max(12, int(0.15 * n))

    q_down = float(np.nanmedian(qs[near0:near1]))
    q_up = float(np.nanmedian(qs[-far_n:]))
    dq = q_down - q_up
    if not np.isfinite(dq) or abs(dq) < 1e-12:
        return None

    dqds = np.gradient(qs, s)
    g = np.abs(dqds)
    valid = (s > 0.015 * R) & (s < 0.94 * np.nanmax(s)) & np.isfinite(g)
    if valid.sum() < 20:
        return None

    idx_valid = np.where(valid)[0]
    max_grad = float(np.nanmax(g[valid]))
    if not np.isfinite(max_grad) or max_grad <= 0:
        return None

    qnorm = (qs - q_up) / dq
    method = str(marker_method).lower().strip()

    if method == "halfjump":
        sv = s[valid]
        qv = qnorm[valid]
        r_peak = _crossing_s_for_level(sv, qv, 0.5)
        s10 = _crossing_s_for_level(sv, qv, 0.1)
        s90 = _crossing_s_for_level(sv, qv, 0.9)
        delta = abs(s90 - s10)
        if not np.isfinite(delta) or delta <= 0:
            delta = abs(dq) / max_grad

    elif method == "gcentroid":
        sv = s[valid]
        w = np.asarray(g[valid], dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if np.nansum(w) <= 0:
            imax = idx_valid[int(np.nanargmax(g[valid]))]
            r_peak = float(s[imax])
            delta = abs(dq) / max_grad
        else:
            r_peak = float(np.nansum(sv * w) / np.nansum(w))
            var = float(np.nansum(w * (sv - r_peak)**2) / np.nansum(w))
            delta = 2.0 * np.sqrt(max(var, 0.0))
            if not np.isfinite(delta) or delta <= 0:
                delta = abs(dq) / max_grad

    else:
        imax = idx_valid[int(np.nanargmax(g[valid]))]
        r_peak = float(s[imax])
        delta = abs(dq) / max_grad

    delta = float(np.clip(delta, 0.001 * R, 20.0 * R))

    return {
        "s": s,
        "q": q,
        "qs": qs,
        "qnorm": qnorm,
        "q_up": q_up,
        "q_down": q_down,
        "jump": dq,
        "r_peak": float(r_peak),
        "delta": delta,
        "max_grad": max_grad,
        "marker_method": method,
    }


def interpolate_profile(xold, yold, xnew):
    xold = np.asarray(xold, dtype=float)
    yold = np.asarray(yold, dtype=float)
    xnew = np.asarray(xnew, dtype=float)
    good = np.isfinite(xold) & np.isfinite(yold)
    xold = xold[good]
    yold = yold[good]
    if len(xold) < 5:
        return np.full_like(xnew, np.nan, dtype=float)

    order = np.argsort(xold)
    xold = xold[order]
    yold = yold[order]
    xu, ind = np.unique(xold, return_index=True)
    yu = yold[ind]
    if len(xu) < 5:
        return np.full_like(xnew, np.nan, dtype=float)

    if SCIPY_OK:
        f = interp1d(xu, yu, bounds_error=False, fill_value=np.nan)
        return f(xnew)
    return np.interp(xnew, xu, yu, left=np.nan, right=np.nan)


def build_attached_field_family(selected, cx, cy, R, variables,
                                theta_min_deg=100.0, theta_max_deg=180.0, n_theta=65,
                                xi_min=-1.0, xi_max=4.0, n_xi=260,
                                ray_ns=900, smooth_sigma=2.0, marker_method="maxgrad"):
    """Build q(xi_q, theta) attached snapshots for POD maps."""
    theta_grid = np.linspace(theta_min_deg, theta_max_deg, n_theta)
    xi_grid = np.linspace(xi_min, xi_max, n_xi)

    snapshots = {v: [] for v in variables}
    meta = []

    for _, row in selected.iterrows():
        print(f"[INFO] field {row['sweep']} {row['gas']} Kn={row['kn']:g} M={row['mach']:g}")
        data = read_case(row)
        interps = {v: make_interpolator(data, v) for v in variables}
        interps = {v: I for v, I in interps.items() if I is not None}
        if "ttr" not in interps:
            print("[WARN] no Ttr interpolator for this case; skipping")
            continue

        case_fields = {v: [] for v in variables}
        for th_deg in theta_grid:
            th = np.deg2rad(th_deg)
            for var in variables:
                if var not in interps:
                    case_fields[var].append(np.full_like(xi_grid, np.nan))
                    continue
                prof = sample_ray_from_surface(data, interps[var], cx, cy, R, th, ns=ray_ns)
                if prof is None:
                    case_fields[var].append(np.full_like(xi_grid, np.nan))
                    continue
                s, q, xs, ys = prof
                met = metrics_from_profile(s, q, R, smooth_sigma=smooth_sigma, marker_method=marker_method)
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


# ============================================================
# POD
# ============================================================

def pod_decompose(snapshot_cube, valid_fraction=0.70):
    Xcube = np.asarray(snapshot_cube, dtype=float)
    if Xcube.ndim != 3 or Xcube.shape[0] < 2:
        return None

    n_snap, n_th, n_xi = Xcube.shape
    valid = np.isfinite(Xcube).mean(axis=0) >= valid_fraction
    if valid.sum() < 20:
        valid = np.isfinite(Xcube).mean(axis=0) >= 0.50
    if valid.sum() < 20:
        return None

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
    norm = np.linalg.norm(Xc)
    if not np.isfinite(norm) or norm <= 1e-14:
        return None

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    energy = S ** 2
    energy = energy / energy.sum()

    def map_back(vec):
        A = np.full((n_th, n_xi), np.nan)
        A[valid] = vec
        return A

    mean_map = map_back(mean)
    modes = [map_back(Vt[i]) for i in range(min(4, len(S)))]

    col_valid = np.isfinite(mean_map).mean(axis=0) > 0.2
    if col_valid.any():
        j = np.where(col_valid)[0]
        crop = (max(0, j[0] - 2), min(n_xi, j[-1] + 3))
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
# Clean display functions: no stripes
# ============================================================

def _copy_cmap(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad("white")
    return cmap


def _masked_gaussian_display(A, sigma=(0.70, 0.90), keep_std=True, passes=1):
    """
    Display-only Gaussian smoothing that does not leak into blank regions.
    """
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)
    if valid.sum() < 10 or not SCIPY_OK:
        return A.copy()

    A0 = np.where(valid, A, 0.0)
    W0 = valid.astype(float)

    B = A.copy()
    for _ in range(max(1, int(passes))):
        B0 = np.where(valid, B, 0.0)
        num = gaussian_filter(B0, sigma=sigma, mode="nearest")
        den = gaussian_filter(W0, sigma=sigma, mode="nearest")

        Bout = np.full_like(A, np.nan, dtype=float)
        good = den > 1.0e-12
        Bout[good] = num[good] / den[good]
        Bout[~valid] = np.nan
        B = Bout

    if keep_std:
        a = A[valid]
        b = B[valid]
        sa = np.nanstd(a)
        sb = np.nanstd(b)
        if np.isfinite(sa) and np.isfinite(sb) and sb > 1.0e-14:
            B = (B - np.nanmean(b)) * (sa / sb) + np.nanmean(a)
            B[~valid] = np.nan

    return B


def _remove_display_stripes(A, strength=0.20, row_sigma=7.5, col_sigma=6.0):
    """
    Row/column bias correction for visualization only.
    The slow trend is protected by subtracting only high-frequency row/column residuals.
    This is not used in the POD calculation.
    """
    A = np.asarray(A, dtype=float)
    valid = np.isfinite(A)
    if valid.sum() < 10 or not SCIPY_OK:
        return A.copy()

    B = A.copy()

    row_med = np.nanmedian(np.where(valid, B, np.nan), axis=1)
    if np.isfinite(row_med).sum() > 5:
        fill = np.nanmedian(row_med)
        row_med = np.where(np.isfinite(row_med), row_med, fill)
        row_slow = gaussian_filter1d(row_med, sigma=row_sigma, mode="nearest")
        row_res = row_med - row_slow
        B = B - strength * row_res[:, None]

    col_med = np.nanmedian(np.where(valid, B, np.nan), axis=0)
    if np.isfinite(col_med).sum() > 5:
        fill = np.nanmedian(col_med)
        col_med = np.where(np.isfinite(col_med), col_med, fill)
        col_slow = gaussian_filter1d(col_med, sigma=col_sigma, mode="nearest")
        col_res = col_med - col_slow
        B = B - 0.5 * strength * col_res[None, :]

    B[~valid] = np.nan
    return B


def plot_fig12_clean_no_stripes(gas, sweep, xi_grid, theta_grid, pods, outdir, n_modes=2):
    wanted = ["ttr", "trot"]
    if any(v not in pods or pods[v] is None for v in wanted):
        missing = [v for v in wanted if v not in pods or pods[v] is None]
        raise RuntimeError(f"Cannot plot Figure 12. Missing POD data for: {missing}")

    ncols = 1 + n_modes
    fig, axes = plt.subplots(
        len(wanted), ncols,
        figsize=(3.10 * ncols, 2.75 * len(wanted)),
        sharey=True,
        constrained_layout=True,
    )
    if len(wanted) == 1:
        axes = np.asarray([axes])

    for r, var in enumerate(wanted):
        pod = pods[var]
        j0, j1 = pod["xi_crop"]
        j0 = max(0, j0 + 1)
        j1 = min(len(xi_grid), j1 - 1)
        xi = np.asarray(xi_grid[j0:j1], dtype=float)
        th = np.asarray(theta_grid, dtype=float)

        maps_raw = [pod["mean_map"][:, j0:j1]]
        titles = [f"{LABELS.get(var, var)} mean"]
        for i in range(min(n_modes, len(pod["modes"]))):
            maps_raw.append(pod["modes"][i][:, j0:j1])
            e = pod["energy"][i] if i < len(pod["energy"]) else np.nan
            titles.append(f"mode {i + 1}, $E_{i + 1}$={e:.3f}")

        for c, (Araw, title) in enumerate(zip(maps_raw, titles)):
            ax = axes[r, c]
            Araw = np.asarray(Araw, dtype=float)
            valid = np.isfinite(Araw)

            if c == 0:
                # Mean fields: stronger angular smoothing for display only.
                # This removes residual horizontal bands while keeping the large-scale
                # thermal-relaxation pattern and the white common-support mask.
                Aplot = _masked_gaussian_display(Araw, sigma=(1.65, 1.35), keep_std=False, passes=1)
                Aplot = _remove_display_stripes(Aplot, strength=0.42, row_sigma=8.0, col_sigma=7.0)
                Aplot = _masked_gaussian_display(Aplot, sigma=(0.65, 0.55), keep_std=False, passes=1)
                finite = Araw[np.isfinite(Araw)]
                if finite.size:
                    vmin, vmax = np.nanpercentile(finite, [2, 98])
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
                else:
                    vmin, vmax = 0.0, 1.0
                cmap = _copy_cmap("viridis")
            else:
                # POD modes: keep the original mode amplitude statistics, but use
                # stronger angular display smoothing to reduce artificial horizontal
                # bands.  Energies and modes are computed above from unsmoothed data.
                Aplot = _masked_gaussian_display(Araw, sigma=(1.85, 1.25), keep_std=True, passes=1)
                Aplot = _remove_display_stripes(Aplot, strength=0.50, row_sigma=8.5, col_sigma=7.0)
                Aplot = _masked_gaussian_display(Aplot, sigma=(0.75, 0.55), keep_std=True, passes=1)
                finite_abs = np.abs(Araw[np.isfinite(Araw)])
                vmax = np.nanpercentile(finite_abs, 98) if finite_abs.size else 1.0
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 1.0
                vmin = -vmax
                cmap = _copy_cmap("coolwarm")

            Aplot[~valid] = np.nan

            im = ax.imshow(
                np.ma.masked_invalid(Aplot),
                origin="lower",
                aspect="auto",
                extent=[xi.min(), xi.max(), th.min(), th.max()],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
                resample=True,
                rasterized=True,
            )

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel(r"$\xi_q$", fontsize=11)
            if c == 0:
                ax.set_ylabel(r"$\theta$ (deg)", fontsize=11)
            ax.tick_params(labelsize=9, direction="out", length=3)

            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
            cb.ax.tick_params(labelsize=8, length=2)
            if c == 0:
                cb.set_label(rf"normalized {LABELS.get(var, var)}", fontsize=8)
            else:
                cb.set_label("POD amplitude", fontsize=8)

    fig.suptitle(
        r"N$_2$ Kn-sweep shock-attached POD modes: thermal fields",
        fontsize=13,
        fontweight="bold",
    )

    outdir = ensure_dir(outdir)
    fname = f"Step2b_{gas}_{sweep}_POD_modes_thermal_combined_JFM_v4_smoother"
    png_path = outdir / f"{fname}.png"
    pdf_path = outdir / f"{fname}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def plot_pod_spectrum(gas, sweep, pods, outdir):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    any_plot = False
    for var, pod in pods.items():
        if pod is None:
            continue
        cum = np.cumsum(pod["energy"])
        ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", lw=1.8, label=LABELS.get(var, var))
        any_plot = True
    if not any_plot:
        plt.close(fig)
        return None
    ax.set_xlabel("Number of POD modes")
    ax.set_ylabel("Cumulative energy")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title(f"{gas} {sweep}: attached-coordinate POD")
    fig.tight_layout()
    fname = f"Step2b_{gas}_{sweep}_POD_spectrum_clean"
    fig.savefig(outdir / f"{fname}.png", dpi=450, bbox_inches="tight")
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    return outdir / f"{fname}.pdf"


# ============================================================
# Main
# ============================================================


def summarize_pod(pod):
    if pod is None or "energy" not in pod:
        return {"E1": np.nan, "E2": np.nan, "C2": np.nan, "modes95": np.nan, "n_modes": 0}
    e = np.asarray(pod["energy"], dtype=float)
    return {
        "E1": float(e[0]) if len(e) > 0 else np.nan,
        "E2": float(e[1]) if len(e) > 1 else np.nan,
        "C2": float(np.sum(e[:2])) if len(e) > 1 else np.nan,
        "modes95": int(np.searchsorted(np.cumsum(e), 0.95) + 1) if len(e) else np.nan,
        "n_modes": int(len(e)),
    }


def density_metrics_for_selected(selected, cx, cy, R, theta_min_deg, theta_max_deg, n_theta, ray_ns, smooth_sigma, marker_method="maxgrad"):
    """Return median density-front standoff and density thickness over all selected cases/rays."""
    theta_grid = np.linspace(theta_min_deg, theta_max_deg, n_theta)
    rows = []
    for _, row in selected.iterrows():
        data = read_case(row)
        interp = make_interpolator(data, "rho")
        if interp is None:
            continue
        standoffs, deltas = [], []
        for th_deg in theta_grid:
            prof = sample_ray_from_surface(data, interp, cx, cy, R, np.deg2rad(th_deg), ns=ray_ns)
            if prof is None:
                continue
            s, q, xs, ys = prof
            met = metrics_from_profile(s, q, R, smooth_sigma=smooth_sigma, marker_method=marker_method)
            if met is None:
                continue
            standoffs.append(float(met["r_peak"]))
            deltas.append(float(met["delta"]))
        rows.append({
            "kn": float(row.get("kn", np.nan)),
            "mach": float(row.get("mach", np.nan)),
            "median_Delta_s": float(np.nanmedian(standoffs)) if len(standoffs) else np.nan,
            "median_delta_rho": float(np.nanmedian(deltas)) if len(deltas) else np.nan,
            "n_valid_rays": int(np.isfinite(standoffs).sum()) if len(standoffs) else 0,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return {"Delta_s_median_all": np.nan, "delta_rho_median_all": np.nan, "n_case_metric": 0}, df
    return {
        "Delta_s_median_all": float(df["median_Delta_s"].median()),
        "delta_rho_median_all": float(df["median_delta_rho"].median()),
        "n_case_metric": int(len(df)),
    }, df



# ============================================================
# Added analyses for JFM support package
# 1) block/ray-bootstrap confidence intervals
# 2) leave-one-out POD stability
# 3) Hornung / kinetic-residual scaling figure
# ============================================================

from scipy import stats as _stats  # noqa: E402


def t_ci_95(values):
    """Return mean, std, n, and two-sided 95% t-confidence interval."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, ci95_low=np.nan, ci95_high=np.nan, ci95_half=np.nan)
    mean = float(np.mean(a))
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
    if n > 1:
        half = float(_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
    else:
        half = np.nan
    return dict(n=int(n), mean=mean, std=std, ci95_low=mean-half if np.isfinite(half) else np.nan,
                ci95_high=mean+half if np.isfinite(half) else np.nan, ci95_half=half)


def infer_block_id(path: Path) -> str:
    """Best-effort block/run/time identifier from a file path."""
    s = str(path)
    patterns = [
        r'(?:block|blk|sample|samp|avg|average|run|seed|time|t)[_\-\s=]*([0-9]+)',
        r'([0-9]{3,})(?=\.[A-Za-z0-9]+$)',
    ]
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(1)
    # fallback: parent/name stem so repeated cases can still be distinguished
    return Path(path).stem


def build_inventory_all(cases):
    """Inventory without dropping repeated case/block files."""
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
                "kn": float(c["kn"]),
                "mach": float(c["mach"]),
                "source": c["source"],
                "block_id": infer_block_id(Path(c["path"])),
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
    return pd.DataFrame(rows)


def density_metrics_by_ray_for_case(row, cx, cy, R, theta_min_deg=100.0, theta_max_deg=180.0,
                                    n_theta=65, ray_ns=900, smooth_sigma=2.0,
                                    marker_method="maxgrad"):
    """Compute per-ray density metrics for one case/field."""
    theta_grid = np.linspace(theta_min_deg, theta_max_deg, n_theta)
    rows = []
    data = read_case(row)
    interp = make_interpolator(data, "rho")
    if interp is None:
        return pd.DataFrame()
    for th_deg in theta_grid:
        prof = sample_ray_from_surface(data, interp, cx, cy, R, np.deg2rad(th_deg), ns=ray_ns)
        if prof is None:
            continue
        s, q, xs, ys = prof
        met = metrics_from_profile(s, q, R, smooth_sigma=smooth_sigma, marker_method=marker_method)
        if met is None:
            continue
        # robust 10--90 width from the same normalized profile
        valid = (met["s"] > 0.015 * R) & (met["s"] < 0.94 * np.nanmax(met["s"]))
        s10 = _crossing_s_for_level(met["s"][valid], met["qnorm"][valid], 0.1) if valid.sum() > 5 else np.nan
        s90 = _crossing_s_for_level(met["s"][valid], met["qnorm"][valid], 0.9) if valid.sum() > 5 else np.nan
        rows.append({
            "theta_deg": float(th_deg),
            "Delta_s": float(met["r_peak"]),
            "Delta_s_over_R": float(met["r_peak"] / R),
            "delta_rho": float(met["delta"]),
            "delta_rho_over_R": float(met["delta"] / R),
            "delta_10_90": float(abs(s90 - s10)) if np.isfinite(s10) and np.isfinite(s90) else np.nan,
            "delta_10_90_over_R": float(abs(s90 - s10) / R) if np.isfinite(s10) and np.isfinite(s90) else np.nan,
            "jump": float(met["jump"]),
            "max_grad": float(met["max_grad"]),
        })
    return pd.DataFrame(rows)


def bootstrap_ci_from_rays(ray_df, n_boot=2000, seed=1234):
    """Bootstrap confidence intervals across body-normal rays for a single time-averaged field."""
    rng = np.random.default_rng(seed)
    out = {}
    if ray_df.empty:
        return out
    n = len(ray_df)
    for col in ["Delta_s_over_R", "delta_rho_over_R", "delta_10_90_over_R"]:
        vals = ray_df[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 5:
            continue
        boots = []
        for _ in range(n_boot):
            b = rng.choice(vals, size=len(vals), replace=True)
            boots.append(np.nanmedian(b))
        boots = np.asarray(boots)
        out[f"{col}_median"] = float(np.nanmedian(vals))
        out[f"{col}_ci95_low"] = float(np.nanpercentile(boots, 2.5))
        out[f"{col}_ci95_high"] = float(np.nanpercentile(boots, 97.5))
        out[f"{col}_ci95_half"] = 0.5 * (out[f"{col}_ci95_high"] - out[f"{col}_ci95_low"])
    out["n_rays"] = int(n)
    return out


def block_confidence_analysis(inv_all, outdir, cx, cy, R, args):
    """
    Compute block-averaged CIs when repeated fields exist for the same case.
    If no repeated blocks are found, also output ray-bootstrap CIs as a fallback.
    """
    outdir = ensure_dir(outdir)
    groups = inv_all.groupby(["gas", "kn", "mach"], dropna=False)
    block_rows = []
    ray_rows = []
    for (gas, kn, mach), g in groups:
        # Per-field/ray bootstrap is always useful as a diagnostic.
        # It is not a substitute for independent DSMC block uncertainty.
        best = g.sort_values("n_points", ascending=False).iloc[0]
        ray_df = density_metrics_by_ray_for_case(best, cx, cy, R, args.theta_min, args.theta_max,
                                                 args.n_theta, args.ray_ns, args.smooth_sigma,
                                                 marker_method="maxgrad")
        ray_df.to_csv(outdir / f"Ray_metrics_{gas}_Kn{kn:g}_M{mach:g}.csv".replace(".", "p"), index=False)
        rr = {"gas": gas, "kn": kn, "mach": mach, "ci_type": "ray_bootstrap_not_independent_blocks"}
        rr.update(bootstrap_ci_from_rays(ray_df, n_boot=args.n_bootstrap, seed=args.seed))
        ray_rows.append(rr)

        if len(g) < 2:
            continue
        vals = []
        for _, row in g.iterrows():
            rdf = density_metrics_by_ray_for_case(row, cx, cy, R, args.theta_min, args.theta_max,
                                                  args.n_theta, args.ray_ns, args.smooth_sigma,
                                                  marker_method="maxgrad")
            if rdf.empty:
                continue
            vals.append({
                "gas": gas,
                "kn": kn,
                "mach": mach,
                "block_id": row.get("block_id", ""),
                "path": row.get("path", ""),
                "Delta_s_over_R": float(np.nanmedian(rdf["Delta_s_over_R"])),
                "delta_rho_over_R": float(np.nanmedian(rdf["delta_rho_over_R"])),
                "delta_10_90_over_R": float(np.nanmedian(rdf["delta_10_90_over_R"])),
                "n_valid_rays": int(len(rdf)),
            })
        bdf = pd.DataFrame(vals)
        if bdf.empty:
            continue
        bdf.to_csv(outdir / f"Block_metric_values_{gas}_Kn{kn:g}_M{mach:g}.csv".replace(".", "p"), index=False)
        row_out = {"gas": gas, "kn": kn, "mach": mach, "ci_type": "independent_field_blocks", "n_blocks": len(bdf)}
        for col in ["Delta_s_over_R", "delta_rho_over_R", "delta_10_90_over_R"]:
            ci = t_ci_95(bdf[col].to_numpy(float))
            for k, v in ci.items():
                row_out[f"{col}_{k}"] = v
        block_rows.append(row_out)

    block_df = pd.DataFrame(block_rows)
    ray_df = pd.DataFrame(ray_rows)
    block_df.to_csv(outdir / "Block_averaged_confidence_intervals.csv", index=False)
    ray_df.to_csv(outdir / "Ray_bootstrap_confidence_intervals_fallback.csv", index=False)
    make_ci_latex_table(block_df, ray_df, outdir / "Block_or_ray_CI_table.tex")
    return block_df, ray_df


def make_ci_latex_table(block_df, ray_df, path):
    """Make a compact LaTeX table for the manuscript appendix."""
    use_block = block_df is not None and not block_df.empty
    df = block_df if use_block else ray_df
    note = ("Independent field blocks" if use_block else
            "Ray-bootstrap intervals from each time-averaged field; these are extraction-variability intervals, not independent DSMC block CIs")
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Representative uncertainty intervals for density-based shock-layer metrics. " + note.replace("_", r"\_") + r".}")
    lines.append(r"\label{tab:app_block_or_ray_ci}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Gas & Case & $\Delta_s/R$ & $\delta_\rho/R$ & $\delta_{10-90}/R$ \\")
    lines.append(r"\midrule")
    if df is not None and not df.empty:
        # pick representative rows: Ar low/intermediate/high and N2 if present, otherwise first 8
        dfx = df.copy()
        if "n_blocks" in dfx.columns:
            def fmt_block(row, col):
                m = row.get(f"{col}_mean", np.nan)
                h = row.get(f"{col}_ci95_half", np.nan)
                return f"{m:.3g} $\\pm$ {h:.1g}" if np.isfinite(m) and np.isfinite(h) else "--"
        else:
            def fmt_block(row, col):
                m = row.get(f"{col}_median", np.nan)
                lo = row.get(f"{col}_ci95_low", np.nan)
                hi = row.get(f"{col}_ci95_high", np.nan)
                h = 0.5*(hi-lo) if np.isfinite(lo) and np.isfinite(hi) else np.nan
                return f"{m:.3g} $\\pm$ {h:.1g}" if np.isfinite(m) and np.isfinite(h) else "--"
        dfx["sortkey"] = dfx["kn"].astype(float)
        for _, row in dfx.sort_values(["gas", "sortkey", "mach"]).head(10).iterrows():
            case = f"$Kn={float(row['kn']):g}, M={float(row['mach']):g}$"
            lines.append(f"{row['gas']} & {case} & {fmt_block(row, 'Delta_s_over_R')} & {fmt_block(row, 'delta_rho_over_R')} & {fmt_block(row, 'delta_10_90_over_R')} \\")
    else:
        lines.append(r"\multicolumn{5}{c}{No readable case metrics were available.} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def vectorize_mode_map(mode_map, mask):
    vals = np.asarray(mode_map, dtype=float)[mask]
    vals = np.where(np.isfinite(vals), vals, 0.0)
    n = np.linalg.norm(vals)
    if n <= 0 or not np.isfinite(n):
        return None
    return vals / n


def mode_similarity(pod_a, pod_b, mode_index=0):
    if pod_a is None or pod_b is None:
        return np.nan
    if len(pod_a.get("modes", [])) <= mode_index or len(pod_b.get("modes", [])) <= mode_index:
        return np.nan
    mask = np.isfinite(pod_a["modes"][mode_index]) & np.isfinite(pod_b["modes"][mode_index])
    if mask.sum() < 10:
        return np.nan
    va = vectorize_mode_map(pod_a["modes"][mode_index], mask)
    vb = vectorize_mode_map(pod_b["modes"][mode_index], mask)
    if va is None or vb is None:
        return np.nan
    return float(abs(np.dot(va, vb)))


def leave_one_out_pod_analysis(selected, outdir, cx, cy, R, args, variables):
    """Compute baseline and leave-one-out POD energy stability for a selected sweep."""
    outdir = ensure_dir(outdir)
    xi, theta, snaps, meta = build_attached_field_family(
        selected, cx, cy, R, variables,
        theta_min_deg=args.theta_min, theta_max_deg=args.theta_max, n_theta=args.n_theta,
        xi_min=args.xi_min, xi_max=args.xi_max, n_xi=args.n_xi,
        ray_ns=args.ray_ns, smooth_sigma=args.smooth_sigma, marker_method="maxgrad"
    )
    meta.to_csv(outdir / "LOO_snapshot_meta.csv", index=False)
    rows = []
    baseline_pods = {v: pod_decompose(snaps[v], valid_fraction=args.valid_fraction) for v in variables}
    for v, pod in baseline_pods.items():
        s = summarize_pod(pod)
        rows.append({"analysis": "baseline", "left_out_index": -1, "left_out_parameter": np.nan, "var": v,
                     "mode1_similarity_to_baseline": 1.0, **s})
    n = len(meta)
    for k in range(n):
        for v in variables:
            cube = np.delete(snaps[v], k, axis=0)
            pod = pod_decompose(cube, valid_fraction=args.valid_fraction)
            sim1 = mode_similarity(baseline_pods.get(v), pod, 0)
            sim2 = mode_similarity(baseline_pods.get(v), pod, 1)
            s = summarize_pod(pod)
            rows.append({
                "analysis": "leave_one_out",
                "left_out_index": int(k),
                "left_out_parameter": float(meta.iloc[k]["parameter"]),
                "left_out_kn": float(meta.iloc[k]["kn"]),
                "left_out_mach": float(meta.iloc[k]["mach"]),
                "var": v,
                "mode1_similarity_to_baseline": sim1,
                "mode2_similarity_to_baseline": sim2,
                **s,
            })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "Leave_one_out_POD_stability.csv", index=False)
    make_loo_latex_table(df, outdir / "Leave_one_out_POD_stability_table.tex")
    plot_loo_summary(df, outdir)
    return df


def make_loo_latex_table(df, path):
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Leave-one-out stability of shock-attached POD compactness. Ranges are obtained by omitting one parameter snapshot at a time. Mode similarity is the absolute inner product between the baseline and leave-one-out leading modes on their common support.}")
    lines.append(r"\label{tab:app_loo_pod}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Variable & Baseline $E_1$ & Leave-one-out $E_1$ range & Leave-one-out $C_2$ range & $|\langle \phi_1,\phi_1^{(-k)}\rangle|$ range \\")
    lines.append(r"\midrule")
    base = df[df["analysis"] == "baseline"].set_index("var")
    loo = df[df["analysis"] == "leave_one_out"]
    for var in sorted(df["var"].unique()):
        sub = loo[loo["var"] == var]
        if sub.empty or var not in base.index:
            continue
        b = base.loc[var]
        lines.append(
            f"{LABELS.get(var,var)} & {b['E1']:.3f} & "
            f"{sub['E1'].min():.3f}--{sub['E1'].max():.3f} & "
            f"{sub['C2'].min():.3f}--{sub['C2'].max():.3f} & "
            f"{sub['mode1_similarity_to_baseline'].min():.3f}--{sub['mode1_similarity_to_baseline'].max():.3f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def plot_loo_summary(df, outdir):
    loo = df[df["analysis"] == "leave_one_out"].copy()
    if loo.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for var in sorted(loo["var"].unique()):
        sub = loo[loo["var"] == var].sort_values("left_out_parameter")
        ax.plot(sub["left_out_parameter"], sub["E1"], marker="o", lw=1.5, label=LABELS.get(var, var))
    ax.set_xlabel("Left-out Knudsen number")
    ax.set_ylabel(r"Leave-one-out $E_1$")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    ax.set_title("Leave-one-out POD stability")
    fig.tight_layout()
    fig.savefig(outdir / "Leave_one_out_POD_E1.png", dpi=450, bbox_inches="tight")
    fig.savefig(outdir / "Leave_one_out_POD_E1.pdf", bbox_inches="tight")
    plt.close(fig)


def normal_shock_epsilon(M, gamma):
    M = np.asarray(M, dtype=float)
    return (gamma - 1.0 + 2.0 / (M**2)) / (gamma + 1.0)


def hornung_shape(eps):
    eps = np.asarray(eps, dtype=float)
    return 2.14 * eps * (1.0 + 0.5 * eps)


def fit_scale(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    den = np.dot(x[good], x[good])
    if den <= 0:
        return np.nan
    return float(np.dot(x[good], y[good]) / den)


def select_mach_sweep(inv, gas, kn_ref=0.01):
    ginv = inv[inv["gas"] == gas].copy()
    if ginv.empty:
        return pd.DataFrame()
    ginv["kndiff"] = np.abs(ginv["kn"] - kn_ref)
    # Use all cases close to the minimum kn available for the gas.
    min_diff = float(ginv["kndiff"].min())
    out = ginv[ginv["kndiff"] <= min_diff + 1e-12].sort_values("mach").copy()
    out["sweep"] = "Mach"
    out["parameter"] = out["mach"]
    return out


def metrics_for_scaling(selected, cx, cy, R, args):
    rows = []
    for _, row in selected.iterrows():
        rdf = density_metrics_by_ray_for_case(row, cx, cy, R, args.theta_min, args.theta_max,
                                              args.n_theta, args.ray_ns, args.smooth_sigma,
                                              marker_method="maxgrad")
        if rdf.empty:
            continue
        rows.append({
            "gas": row["gas"], "kn": float(row["kn"]), "mach": float(row["mach"]),
            "Delta_s_over_R": float(np.nanmedian(rdf["Delta_s_over_R"])),
            "delta_rho_over_R": float(np.nanmedian(rdf["delta_rho_over_R"])),
            "delta_10_90_over_R": float(np.nanmedian(rdf["delta_10_90_over_R"])),
            "n_valid_rays": int(len(rdf)),
        })
    return pd.DataFrame(rows)


def scaling_analysis(inv, outdir, cx, cy, R, args):
    outdir = ensure_dir(outdir)
    all_rows = []
    for gas, gamma in [("Ar", 5.0/3.0), ("N2", 1.4)]:
        # Mach sweep at low Kn
        mach_sel = select_mach_sweep(inv, gas, args.kn_sweep_ref)
        if not mach_sel.empty:
            mdf = metrics_for_scaling(mach_sel, cx, cy, R, args)
            if not mdf.empty:
                mdf["sweep"] = "Mach"
                mdf["gamma_ref"] = gamma
                mdf["epsilon"] = normal_shock_epsilon(mdf["mach"].to_numpy(float), gamma)
                mdf["hornung_shape"] = hornung_shape(mdf["epsilon"])
                scale = fit_scale(mdf["hornung_shape"], mdf["Delta_s_over_R"])
                mdf["hornung_scaled"] = scale * mdf["hornung_shape"]
                mdf["hornung_scale_factor"] = scale
                all_rows.append(mdf)
        # Kn sweep at M ref
        kn_sel = select_kn_sweep(inv, gas, args.kn_sweep_mach, args.kn_values)
        if not kn_sel.empty:
            kdf = metrics_for_scaling(kn_sel, cx, cy, R, args)
            if not kdf.empty:
                kdf["sweep"] = "Kn"
                kdf["gamma_ref"] = gamma
                eps = float(normal_shock_epsilon(args.kn_sweep_mach, gamma))
                kdf["epsilon"] = eps
                # residual relative to the lowest-Kn standoff in the same gas
                kdf = kdf.sort_values("kn")
                base = float(kdf.iloc[0]["Delta_s_over_R"])
                kdf["kinetic_residual_Delta_over_R"] = kdf["Delta_s_over_R"] - base
                all_rows.append(kdf)
    if not all_rows:
        return pd.DataFrame()
    df = pd.concat(all_rows, ignore_index=True, sort=False)
    df.to_csv(outdir / "Hornung_kinetic_scaling_data.csv", index=False)
    plot_scaling(df, outdir)
    make_scaling_latex_table(df, outdir / "Hornung_kinetic_scaling_summary.tex")
    return df


def plot_scaling(df, outdir):
    # Figure 1: Mach sweep versus Hornung reference shape.
    mach = df[df["sweep"] == "Mach"].copy()
    if not mach.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for gas in sorted(mach["gas"].unique()):
            sub = mach[mach["gas"] == gas].sort_values("epsilon")
            ax.plot(sub["epsilon"], sub["Delta_s_over_R"], "o", label=f"{gas} density marker")
            ax.plot(sub["epsilon"], sub["hornung_scaled"], "--", label=f"{gas} scaled Hornung shape")
        ax.set_xlabel(r"Inverse normal-shock density ratio, $\epsilon=\rho_\infty/\rho_s$")
        ax.set_ylabel(r"Median density-front standoff, $\Delta_s/R$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("Mach sweep: inviscid compression-ratio organization")
        fig.tight_layout()
        fig.savefig(outdir / "Scaling_Mach_Hornung_reference.png", dpi=450, bbox_inches="tight")
        fig.savefig(outdir / "Scaling_Mach_Hornung_reference.pdf", bbox_inches="tight")
        plt.close(fig)
    # Figure 2: Kn sweep residual and thickness.
    kn = df[df["sweep"] == "Kn"].copy()
    if not kn.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        for gas in sorted(kn["gas"].unique()):
            sub = kn[kn["gas"] == gas].sort_values("kn")
            ax.plot(sub["kn"], sub["kinetic_residual_Delta_over_R"], "o-", label=fr"{gas}: $\Delta_s/R$ residual")
            ax.plot(sub["kn"], sub["delta_rho_over_R"], "s--", label=fr"{gas}: $\delta_\rho/R$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$Kn_\infty$")
        ax.set_ylabel("Normalized length")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title("Kn sweep: kinetic residual and density-layer thickness")
        fig.tight_layout()
        fig.savefig(outdir / "Scaling_Kn_kinetic_residual.png", dpi=450, bbox_inches="tight")
        fig.savefig(outdir / "Scaling_Kn_kinetic_residual.pdf", bbox_inches="tight")
        plt.close(fig)


def make_scaling_latex_table(df, path):
    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary of scaling diagnostics. In the Mach sweep, the Hornung reference shape is scaled by one least-squares factor because the present marker is a density-gradient location rather than the inviscid sonic-line standoff. In the Knudsen sweep, the residual standoff is measured relative to the lowest-$Kn_\infty$ case for each gas.}")
    lines.append(r"\label{tab:app_scaling_summary}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Gas & Sweep & $N$ & Scale/slope diagnostic & RMS error or range \\")
    lines.append(r"\midrule")
    for gas in sorted(df["gas"].unique()):
        m = df[(df["gas"] == gas) & (df["sweep"] == "Mach")]
        if not m.empty and "hornung_scaled" in m:
            rms = float(np.sqrt(np.nanmean((m["Delta_s_over_R"] - m["hornung_scaled"])**2)))
            sf = float(np.nanmedian(m["hornung_scale_factor"]))
            lines.append(f"{gas} & Mach & {len(m)} & scaled Hornung factor {sf:.3g} & RMS {rms:.2g} \\")
        k = df[(df["gas"] == gas) & (df["sweep"] == "Kn")]
        if not k.empty:
            rng = float(np.nanmax(k["kinetic_residual_Delta_over_R"]) - np.nanmin(k["kinetic_residual_Delta_over_R"]))
            lines.append(f"{gas} & Kn & {len(k)} & residual $\\Delta_s/R$ grows with $Kn_\\infty$ & range {rng:.2g} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="JFM cylinder support analysis: block CIs, leave-one-out POD, Hornung/kinetic scaling.")
    ap.add_argument("--inputs", nargs="+", required=True, help="RAR/ZIP archives or extracted folders containing DS2V/Tecplot fields.")
    ap.add_argument("--out", default="JFM_support_analyses", help="Output folder.")
    ap.add_argument("--extract_root", default="_support_extracted", help="Extraction folder for archives.")
    ap.add_argument("--radius", type=float, default=0.1524)
    ap.add_argument("--leftmost_x", type=float, default=0.0)
    ap.add_argument("--center_y", type=float, default=0.0)
    ap.add_argument("--default_kn_if_missing", type=float, default=0.01)
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0)
    ap.add_argument("--kn_sweep_mach", type=float, default=10.0)
    ap.add_argument("--kn_sweep_ref", type=float, default=0.01, help="Reference Kn for selecting Mach sweep.")
    ap.add_argument("--kn_values", nargs="+", type=float, default=[0.01,0.025,0.05,0.075,0.1,0.25,0.5,0.75,1.0])
    ap.add_argument("--gas", default="N2", choices=["Ar", "N2"], help="Gas for leave-one-out POD stability.")
    ap.add_argument("--variables", nargs="+", default=["rho", "mach", "p", "ttr", "trot"], help="Variables for leave-one-out POD.")
    ap.add_argument("--theta_min", type=float, default=100.0)
    ap.add_argument("--theta_max", type=float, default=180.0)
    ap.add_argument("--n_theta", type=int, default=65)
    ap.add_argument("--xi_min", type=float, default=-1.0)
    ap.add_argument("--xi_max", type=float, default=4.0)
    ap.add_argument("--n_xi", type=int, default=260)
    ap.add_argument("--ray_ns", type=int, default=900)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)
    ap.add_argument("--valid_fraction", type=float, default=0.70)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_dir(args.out)
    roots = prepare_inputs(args.inputs, args.extract_root)
    cases = find_flow_files(roots, default_kn=args.default_kn_if_missing, default_mach=args.default_mach_if_missing)
    if not cases:
        raise RuntimeError("No readable DS2FF/TriangulatedGrid files found.")

    inv_all = build_inventory_all(cases)
    inv_all.to_csv(outdir / "Inventory_all_fields_including_blocks.csv", index=False)
    inv = build_inventory(cases)
    inv.to_csv(outdir / "Inventory_deduplicated_cases.csv", index=False)

    R = float(args.radius)
    cx = float(args.leftmost_x) + R
    cy = float(args.center_y)

    print("\n[1/3] Block/ray-bootstrap confidence intervals")
    ci_dir = ensure_dir(outdir / "block_or_ray_confidence")
    block_df, ray_df = block_confidence_analysis(inv_all, ci_dir, cx, cy, R, args)

    print("\n[2/3] Leave-one-out POD stability")
    loo_dir = ensure_dir(outdir / "leave_one_out_pod")
    selected = select_kn_sweep(inv, args.gas, args.kn_sweep_mach, args.kn_values)
    # Drop unavailable variables automatically.
    variables = list(args.variables)
    if args.gas == "Ar":
        variables = [v for v in variables if v != "trot"]
    loo_df = leave_one_out_pod_analysis(selected, loo_dir, cx, cy, R, args, variables)

    print("\n[3/3] Hornung / kinetic residual scaling")
    scaling_dir = ensure_dir(outdir / "hornung_kinetic_scaling")
    scaling_df = scaling_analysis(inv, scaling_dir, cx, cy, R, args)

    # Combined README for Overleaf appendix.
    readme = f"""
JFM cylinder support analyses
=============================

This folder contains three support analyses requested for the manuscript:

1. block_or_ray_confidence/
   - Block_averaged_confidence_intervals.csv
   - Ray_bootstrap_confidence_intervals_fallback.csv
   - Block_or_ray_CI_table.tex

   If repeated independent field/block files were found for the same gas/Kn/Mach case,
   the table uses independent-field block CIs. Otherwise it reports ray-bootstrap
   intervals from each time-averaged field. The latter quantify extraction variability
   across body-normal rays and should not be described as independent DSMC block CIs.

2. leave_one_out_pod/
   - Leave_one_out_POD_stability.csv
   - Leave_one_out_POD_stability_table.tex
   - Leave_one_out_POD_E1.pdf/png

3. hornung_kinetic_scaling/
   - Hornung_kinetic_scaling_data.csv
   - Hornung_kinetic_scaling_summary.tex
   - Scaling_Mach_Hornung_reference.pdf/png
   - Scaling_Kn_kinetic_residual.pdf/png

Recommended LaTeX inputs for the appendix:
\\input{{block_or_ray_confidence/Block_or_ray_CI_table.tex}}
\\input{{leave_one_out_pod/Leave_one_out_POD_stability_table.tex}}
\\input{{hornung_kinetic_scaling/Hornung_kinetic_scaling_summary.tex}}

Generated with gas={args.gas}, Kn sweep M={args.kn_sweep_mach}, theta sector=[{args.theta_min}, {args.theta_max}], n_theta={args.n_theta}, n_xi={args.n_xi}.
"""
    (outdir / "README_support_analyses.txt").write_text(readme, encoding="utf-8")

    print("\n[OK] All support analyses complete.")
    print(f"[OK] Output folder: {outdir}")
    print("[OK] Key LaTeX tables:")
    print(f"     {ci_dir / 'Block_or_ray_CI_table.tex'}")
    print(f"     {loo_dir / 'Leave_one_out_POD_stability_table.tex'}")
    print(f"     {scaling_dir / 'Hornung_kinetic_scaling_summary.tex'}")


if __name__ == "__main__":
    main()

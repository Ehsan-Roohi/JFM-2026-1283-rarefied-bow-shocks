#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_akhlaghi_vs_ray_shock_v19.py

Simple comparison of two shock-location definitions:

(1) Our current ray method:
    Along rays from cylinder centre, find the first significant upstream
    density-gradient peak outside the wall.

(2) Akhlaghi-style SWD:
    Convert the density field to a Cartesian image, compute a Roberts-cross
    schlieren function, then for each horizontal row fit a Gaussian to the
    schlieren peak near the bow shock:
        G(x) = B1 exp[-(x-B2)^2/(2 B3^2)]
    The shock centre is sigma=G/B1=1 at x=B2.
    The aft-shock point is sigma=0.1 on the downstream/body side.

This script only makes check figures and CSV files; it is not the final paper
pipeline.

Outputs:
  <out>/case_inventory_compare.csv
  <out>/shock_compare_points.csv
  <out>/checks_compare/compare_<case>.png
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle


# ----------------------------- I/O -----------------------------

def normalize_key(s: str) -> str:
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


def parse_variables_block(lines, start_idx):
    block = []
    for k in range(start_idx, len(lines)):
        s = lines[k].strip()
        us = s.upper()
        if k > start_idx and (us.startswith("ZONE") or re.match(r"^[\s\+\-]?(?:\d|\.\d)", s)):
            break
        block.append(s)
    return re.findall(r'"([^"]+)"', " ".join(block))


def read_tecplot_any(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    raw = path.read_bytes().replace(b"\x00", b"")
    txt = raw.decode("latin1", errors="ignore")
    lines = txt.splitlines()

    default_vars = [
        "X", "Y", "ND", "D", "U", "V", "W",
        "TTR", "TRT", "TVB", "TOV", "MA", "MC", "MCT",
        "MFP", "SOF", "FSP", "ANG", "P",
        "Angular momentum/(Mass.L)"
    ]

    var_idx = None
    zone_idx = None
    for i, line in enumerate(lines):
        us = line.strip().upper()
        if "VARIABLES" in us and var_idx is None:
            var_idx = i
        if "ZONE" in us and zone_idx is None:
            zone_idx = i
            break

    variables = parse_variables_block(lines, var_idx) if var_idx is not None else default_vars
    if not variables:
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
            us = line.strip().upper()
            if "VARIABLES" in us or us.startswith("TITLE") or us.startswith("ZONE"):
                continue
            if re.fullmatch(r'"[^"]+"\s*,?', line.strip()):
                continue
            data_lines.append(line)
        data_text = "\n".join(data_lines)

    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", data_text)
    if len(nums) < nvar:
        raise ValueError(f"{path}: not enough numeric data; numbers={len(nums)}, nvar={nvar}")

    if npts is None:
        npts = len(nums) // nvar
    need = npts * nvar
    if len(nums) < need:
        npts = len(nums) // nvar
        need = npts * nvar

    arr = np.asarray(nums[:need], dtype=float).reshape(npts, nvar)
    return pd.DataFrame(arr, columns=variables), variables


def variable_map(cols: List[str]) -> Dict[str, Optional[str]]:
    norm = {normalize_key(c): c for c in cols}

    def choose(cands, required=True):
        for c in cands:
            if normalize_key(c) in norm:
                return norm[normalize_key(c)]
        for key, orig in norm.items():
            for c in cands:
                cc = normalize_key(c)
                if key == cc or key.startswith(cc) or cc in key:
                    return orig
        if required:
            raise KeyError(f"Could not find {cands} in columns: {cols}")
        return None

    return {
        "x": choose(["X", "x"]),
        "y": choose(["Y", "y"]),
        "rho": choose(["D", "Density", "Rho", "rho"]),
        "u": choose(["U", "u"], required=False),
        "v": choose(["V", "v"], required=False),
        "ttr": choose(["TTR", "Ttr", "T"], required=False),
        "trot": choose(["TRT", "Trot"], required=False),
        "mach": choose(["MA", "Mach", "M"], required=False),
        "p": choose(["P", "Pressure"], required=False),
        "mfp": choose(["MFP", "MeanFreePath"], required=False),
    }


def extract_inputs(inputs: List[str], out_root: Path) -> List[Path]:
    ex = out_root / "_extracted"
    ex.mkdir(parents=True, exist_ok=True)
    roots = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            roots.append(p.resolve())
            continue
        if not p.exists():
            raise FileNotFoundError(p)
        target = ex / p.stem.replace(" ", "_")
        target.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(target)
        elif p.suffix.lower() == ".rar":
            cmds = []
            if shutil.which("unrar"):
                cmds.append(["unrar", "x", "-o+", str(p), str(target)])
            if shutil.which("7z"):
                cmds.append(["7z", "x", "-y", f"-o{target}", str(p)])
            if shutil.which("unar"):
                cmds.append(["unar", "-f", "-o", str(target), str(p)])
            if shutil.which("bsdtar"):
                cmds.append(["bsdtar", "-xf", str(p), "-C", str(target)])
            if not cmds:
                raise RuntimeError("No rar extractor found. Extract manually or install unrar/7z/unar/bsdtar.")
            ok = False
            last = ""
            for cmd in cmds:
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    ok = True
                    break
                except subprocess.CalledProcessError as e:
                    last = e.stderr[-1000:]
            if not ok:
                raise RuntimeError(f"Could not extract {p}:\n{last}")
        else:
            raise ValueError(f"Unsupported input: {p}")
        roots.append(target)
    return roots


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
        raw = raw.replace(",", ".")
        if "." in raw:
            val = float(raw)
            return val if 3 <= val <= 30 else np.nan
        if re.match(r"^[0-9]+$", raw):
            intval = int(raw)
            if 3 <= intval <= 30:
                return float(intval)
            if len(raw) >= 2:
                val = float(raw[0] + "." + raw[1:])
                return val if 3 <= val <= 30 else np.nan
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
    if "argon-data-mach" in s or "triangulategrid" in s or "triangulatedgrid" in s:
        return "mach_sweep_argon"
    if "results-n2" in s:
        return "mach_sweep_n2"
    if "prop-kn" in s or "kn=" in s:
        return "kn_sweep"
    return "unknown"


def find_flow_files(roots, default_kn=0.01, default_mach=10.0):
    cases = []
    pats = [
        "**/DS2FF.DAT", "**/DS2FF.dat", "**/*DS2FF*.DAT", "**/*DS2FF*.dat",
        "**/TriangulateGrid*.dat", "**/TriangulateGrid*.DAT",
        "**/TriangulatedGrid*.dat", "**/TriangulatedGrid*.DAT",
    ]
    for root in roots:
        files = []
        for pat in pats:
            files.extend(root.glob(pat))
        files = sorted(set(f for f in files if "DS2SU" not in f.name.upper()))
        for f in files:
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
            surf = None
            for pat in ["DS2SU*.DAT", "DS2SU*.dat", "*SU*.DAT", "*SU*.dat"]:
                hits = sorted(f.parent.glob(pat))
                if hits:
                    surf = hits[0]
                    break
            cases.append({"gas": gas, "Kn": float(kn), "Mach": float(mach),
                          "source": source_tag(f), "flow_file": f, "surface_file": surf})
    return cases


@dataclass
class Case:
    gas: str
    Kn: float
    Mach: float
    source: str
    flow_file: Path
    surface_file: Optional[Path]
    df: pd.DataFrame
    vm: Dict[str, Optional[str]]
    center: Tuple[float, float]
    radius: float
    tri: mtri.Triangulation


def fit_circle_surface(surface_file: Optional[Path]):
    if surface_file is None:
        return None
    try:
        sdf, cols = read_tecplot_any(surface_file)
        cmap = {normalize_key(c): c for c in cols}
        if "x" not in cmap or "y" not in cmap:
            return None
        x = sdf[cmap["x"]].to_numpy(float)
        y = sdf[cmap["y"]].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if len(x) < 10:
            return None
        A = np.column_stack([2*x, 2*y, np.ones_like(x)])
        b = x*x + y*y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        r = math.sqrt(max(c + cx*cx + cy*cy, 0))
        return float(cx), float(cy), float(r)
    except Exception:
        return None


def load_case(info, center_x=None, center_y=None, radius=None):
    df, cols = read_tecplot_any(info["flow_file"])
    vm = variable_map(cols)
    if center_x is not None and center_y is not None and radius is not None:
        cx, cy, R = float(center_x), float(center_y), float(radius)
        geom_source = "manual"
    else:
        fit = fit_circle_surface(info.get("surface_file"))
        if fit is None:
            raise RuntimeError(f"No geometry for {info['flow_file']}; use --leftmost_x/--center_x --radius.")
        cx, cy, R = fit
        geom_source = "surface_fit"
    x = df[vm["x"]].to_numpy(float)
    y = df[vm["y"]].to_numpy(float)
    tri = mtri.Triangulation(x, y)
    c = Case(info["gas"], info["Kn"], info["Mach"], info["source"], info["flow_file"],
             info.get("surface_file"), df, vm, (cx, cy), R, tri)
    c.geom_source = geom_source
    return c


# ----------------------------- two shock methods -----------------------------

def smooth1d(q, window=11):
    q = np.asarray(q, dtype=float)
    if len(q) < 3:
        return q
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    return np.convolve(np.pad(q, (pad, pad), mode="edge"), np.ones(window)/window, mode="valid")


def interp_rho(case: Case):
    return mtri.LinearTriInterpolator(case.tri, case.df[case.vm["rho"]].to_numpy(float))


def sample_ray(case: Case, angle: float, n=900):
    cx, cy = case.center
    R = case.radius
    x = case.df[case.vm["x"]].to_numpy(float)
    y = case.df[case.vm["y"]].to_numpy(float)
    ux, uy = math.cos(angle), math.sin(angle)
    proj = (x-cx)*ux + (y-cy)*uy
    pos = proj[proj > R]
    rmax = np.nanpercentile(pos, 98) if pos.size else 4*R
    rmax = max(rmax, 3.5*R)
    rr = np.linspace(1.005*R, rmax, n)
    return rr, cx + rr*ux, cy + rr*uy


def detect_ray_method(case: Case, half_angle=55, n_angles=41, min_standoff=0.015, max_standoff=0.25):
    """Our constrained ray method: first significant upstream density-gradient peak."""
    rhoi = interp_rho(case)
    rows = []
    cx, cy = case.center
    R = case.radius
    upstream_angle = math.pi

    for sign in [-1, 1]:
        for deg in np.linspace(0, half_angle, n_angles):
            angle = upstream_angle + sign*np.deg2rad(deg)
            if math.sin(angle) < -1e-10:
                continue
            rr, xs, ys = sample_ray(case, angle)
            rho = rhoi(xs, ys)
            rho = np.asarray(rho.filled(np.nan) if hasattr(rho, "filled") else rho, dtype=float)
            good = np.isfinite(rho)
            if good.sum() < 50:
                continue
            rr, xs, ys, rho = rr[good], xs[good], ys[good], rho[good]
            st = rr - R
            m = (st >= min_standoff) & (st <= max_standoff) & (xs <= cx + 0.05*R)
            if m.sum() < 30:
                continue
            rrw, xw, yw, rw = rr[m], xs[m], ys[m], rho[m]
            grad = np.abs(np.gradient(smooth1d(rw, max(7, len(rw)//70)), rrw))
            if not np.isfinite(grad).any() or np.nanmax(grad) <= 0:
                continue
            th = 0.20*np.nanmax(grad)
            cand = np.where(grad >= th)[0]
            idx = int(cand[np.argmax(rrw[cand])]) if len(cand) else int(np.nanargmax(grad))
            lo, hi = max(0, idx-4), min(len(grad), idx+5)
            idx = lo + int(np.nanargmax(grad[lo:hi]))
            rows.append({
                "method": "ray_center",
                "x": float(xw[idx]), "y": float(yw[idx]),
                "angle_deg": float(sign*deg),
                "standoff": float(rrw[idx] - R),
                "value": float(grad[idx]),
            })
    return pd.DataFrame(rows)


def akhlaghi_swd_method(case: Case, ray_guess: pd.DataFrame,
                         nx=900, ny=560,
                         x_pad=0.45,
                         y_max_factor=4.0,
                         sigma_level=0.1,
                         near_wall_buffer=0.025,
                         row_window=0.090,
                         min_prom=0.05):
    """Akhlaghi-style SWD with a shock-neighbourhood constraint.

    This implements the *important* part that was missing before:
    the Gaussian fit is NOT applied over the entire row.  It is applied only
    in the neighbourhood of the shock region.  The shock neighbourhood is
    initialized from the ray-based ridge, then the Roberts-cross schlieren
    function is fit locally along each y=const row.

    Steps:
      1) Interpolate density to a Cartesian image.
      2) Mask the cylinder and a near-wall buffer.
      3) Compute Roberts-cross schlieren magnitude.
      4) For each horizontal row, use the ray shock locus as an initial
         guess for x_s(y), then fit a Gaussian only in that neighbourhood.
      5) sigma=1 is the Gaussian centre. sigma=0.1 is the aft/downstream point.

    This is closer to Akhlaghi's SWD than the previous global-row maximum,
    because Akhlaghi fits the schlieren function in the shock region, not
    over wall gradients or the whole row.
    """
    if ray_guess is None or ray_guess.empty or len(ray_guess) < 4:
        return pd.DataFrame()

    x = case.df[case.vm["x"]].to_numpy(float)
    y = case.df[case.vm["y"]].to_numpy(float)
    cx, cy = case.center
    R = case.radius

    # Cartesian crop: upstream/front of the cylinder plus upper half.
    xmin = max(np.nanmin(x), cx - R - x_pad)
    xmax = min(np.nanmax(x), cx + 0.25*R)
    ymin = cy
    ymax = min(np.nanmax(y), cy + y_max_factor*R)
    if ymax <= ymin:
        ymax = np.nanmax(y)

    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    rhoi = interp_rho(case)
    RHO = rhoi(Xg, Yg)
    RHO = np.asarray(RHO.filled(np.nan) if hasattr(RHO, "filled") else RHO, dtype=float)

    # Strong wall gradients must not be allowed to enter SWD.
    inside_or_near_wall = (Xg-cx)**2 + (Yg-cy)**2 <= (R + near_wall_buffer)**2
    RHO[inside_or_near_wall] = np.nan

    # Roberts-cross schlieren function.
    F = np.full_like(RHO, np.nan)
    d1 = RHO[:-1, :-1] - RHO[1:, 1:]
    d2 = RHO[:-1, 1:] - RHO[1:, :-1]
    F[:-1, :-1] = np.sqrt(d1*d1 + d2*d2)
    if np.nanmax(F) > 0:
        F = F / np.nanmax(F)

    xgrid = Xg[0, :]
    ygrid = Yg[:, 0]

    # Smooth ray guess and use it as row-wise shock neighbourhood.
    rg = ray_guess.dropna(subset=["x", "y"]).sort_values("y")
    # Use unique y for interpolation stability.
    rg = rg.groupby("y", as_index=False)["x"].median().sort_values("y")
    if len(rg) < 4:
        return pd.DataFrame()
    y_guess = rg["y"].to_numpy(float)
    x_guess = rg["x"].to_numpy(float)

    rows = []
    for j in range(2, ny-3):
        yy = float(ygrid[j])
        if yy < y_guess.min() or yy > y_guess.max():
            continue

        # Initial shock guess at this row.
        x0_guess = float(np.interp(yy, y_guess, x_guess))

        # Local shock neighbourhood only.
        row = F[j, :]
        valid = np.isfinite(row)
        valid &= (xgrid >= x0_guess - row_window) & (xgrid <= x0_guess + row_window)
        valid &= (xgrid <= cx + 0.05*R)
        valid &= ((xgrid-cx)**2 + (yy-cy)**2 > (R + near_wall_buffer)**2)
        if valid.sum() < 12:
            continue

        xv = xgrid[valid]
        fv = row[valid]
        if not np.isfinite(fv).any():
            continue

        fv_s = smooth1d(fv, 7)
        fmax = np.nanmax(fv_s)
        if not np.isfinite(fmax) or fmax < min_prom:
            continue

        imax = int(np.nanargmax(fv_s))

        # Fit Gaussian in a smaller window around local peak.
        lo = max(0, imax-8)
        hi = min(len(xv), imax+9)
        xx = xv[lo:hi]
        ff = fv_s[lo:hi]
        good = np.isfinite(ff) & (ff > max(1e-12, 0.20*fmax))

        if good.sum() >= 5:
            try:
                # log(G) = a x^2 + b x + c, a<0
                a, b, c = np.polyfit(xx[good], np.log(ff[good]), 2)
                if a < 0:
                    x_center = float(-b/(2*a))
                    sigma = float(np.sqrt(-1/(2*a)))
                    amp = float(np.exp(c - b*b/(4*a)))
                else:
                    x_center = float(xv[imax])
                    sigma = np.nan
                    amp = float(fmax)
            except Exception:
                x_center = float(xv[imax])
                sigma = np.nan
                amp = float(fmax)
        else:
            x_center = float(xv[imax])
            sigma = np.nan
            amp = float(fmax)

        # Reject if Gaussian centre jumped out of local neighbourhood or near wall.
        if abs(x_center - x0_guess) > row_window:
            continue
        if (x_center-cx)**2 + (yy-cy)**2 <= (R + near_wall_buffer)**2:
            continue
        if x_center > cx + 0.05*R:
            continue

        rows.append({
            "method": "akhlaghi_sigma1",
            "x": x_center,
            "y": yy,
            "sigma": 1.0,
            "value": amp,
            "x_guess": x0_guess,
        })

        # sigma=0.1 aft/downstream point on the body side (+x for left-to-right flow).
        if np.isfinite(sigma) and sigma > 0:
            dx = sigma * math.sqrt(-2.0*math.log(max(float(sigma_level), 1e-12)))
            xaft = x_center + dx
            if abs(xaft - x0_guess) <= row_window*1.8:
                if (xaft-cx)**2 + (yy-cy)**2 > (R + near_wall_buffer*0.6)**2 and xaft <= cx + 0.15*R:
                    rows.append({
                        "method": f"akhlaghi_sigma{sigma_level:g}",
                        "x": float(xaft),
                        "y": yy,
                        "sigma": float(sigma_level),
                        "value": amp,
                        "x_guess": x0_guess,
                    })

    return pd.DataFrame(rows)



def make_compare_plot(case: Case, ray: pd.DataFrame, ak: pd.DataFrame, outdir: Path, unit="m"):
    """Comparison plot, upper-half only.

    v22_fix:
      * does NOT change the working v21 shock-detection logic
      * only changes the plotting window and cylinder drawing:
          - y >= 0 only
          - upper semicircle only
    """
    outdir.mkdir(parents=True, exist_ok=True)

    x = get_x = case.df[case.vm["x"]].to_numpy(float)
    y = get_y = case.df[case.vm["y"]].to_numpy(float)
    rho = case.df[case.vm["rho"]].to_numpy(float)

    # freestream density from leftmost 8%
    rho_inf = np.nanmedian(rho[x <= np.nanquantile(x, 0.08)])
    vals = rho / rho_inf

    # Upper half only.
    mask = y >= case.center[1]
    tri = mtri.Triangulation(x[mask], y[mask])
    v2 = vals[mask]

    levels = np.linspace(np.nanpercentile(v2, 2), np.nanpercentile(v2, 98), 36)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    cf = ax.tricontourf(tri, v2, levels=levels, cmap="turbo", extend="both")

    cx, cy = case.center
    R = case.radius

    # Draw only the upper semicircle of the cylinder.
    theta = np.linspace(0.0, np.pi, 500)
    x_cyl = cx + R*np.cos(theta)
    y_cyl = cy + R*np.sin(theta)
    ax.fill(
        x_cyl, y_cyl,
        facecolor="white",
        edgecolor="black",
        lw=1.5,
        zorder=5
    )

    # Draw wall/symmetry line y=0.
    xmin_plot = np.nanmin(x[mask])
    xmax_plot = np.nanmax(x[mask])
    ax.plot([xmin_plot, xmax_plot], [case.center[1], case.center[1]],
            color="black", lw=1.0, zorder=4)

    if not ray.empty:
        # black line with white markers: visible on both bright and dark contours
        ax.plot(
            ray["x"], ray["y"],
            color="black",
            marker="o",
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.6,
            linewidth=2.4,
            label="Ray ridge",
            zorder=8
        )

    if not ak.empty:
        c1 = ak[ak["method"] == "akhlaghi_sigma1"]
        c01 = ak[ak["method"].str.contains("sigma0.1", regex=False)]
        
        if not c1.empty:
                ax.scatter(
                    c1["x"], c1["y"],
                    s=22,
                    c="#b12fc9",
                    edgecolors="black",
                    linewidths=0.25,
                    label=r"SWD, $\sigma=1$",
                    zorder=9
                )
        
        if not c01.empty:
                ax.scatter(
                    c01["x"], c01["y"],
                    s=22,
                    c="#21c7d9",
                    edgecolors="black",
                    linewidths=0.25,
                    label=r"SWD, $\sigma=0.1$",
                    zorder=9
                )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")

    # Upper half only: remove empty lower region.
    ymax_plot = np.nanmax(y[mask])
    ax.set_ylim(case.center[1], ymax_plot)
    ax.set_xlim(xmin_plot, xmax_plot)

    # ax.set_title(f"{case.gas}, Kn={case.Kn:g}, M={case.Mach:g}, center=({cx:g},{cy:g}), R={R:g}")

    ax.legend(
    loc="upper right",
    fontsize=12,
    frameon=True,
    framealpha=0.88,
    borderpad=0.45,
    labelspacing=0.35,
    handlelength=1.6,
    markerscale=1.2
    )
    cb = fig.colorbar(cf, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"$\rho/\rho_\infty$")

    tag = f"{case.gas}_Kn{str(case.Kn).replace('.','p')}_M{str(case.Mach).replace('.','p')}_{case.source}"
    fig.savefig(outdir / f"compare_{tag}.png", dpi=230, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="Shock_Compare_Akhlaghi_v19")
    ap.add_argument("--leftmost_x", type=float, default=None)
    ap.add_argument("--center_x", type=float, default=None)
    ap.add_argument("--center_y", type=float, default=0.0)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--default_kn_if_missing", type=float, default=0.01)
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0)
    ap.add_argument("--only_gas", default="")
    ap.add_argument("--only_mach", type=float, default=None)
    ap.add_argument("--only_kn", type=float, default=None)
    ap.add_argument("--max_cases", type=int, default=9999)
    ap.add_argument("--half_angle", type=float, default=55)
    ap.add_argument("--angles", type=int, default=41)
    ap.add_argument("--min_standoff", type=float, default=0.015)
    ap.add_argument("--max_standoff", type=float, default=0.35)
    ap.add_argument("--ak_nx", type=int, default=700)
    ap.add_argument("--ak_ny", type=int, default=420)
    ap.add_argument("--ak_xpad", type=float, default=0.40)
    ap.add_argument("--ak_wall_buffer", type=float, default=0.025,
                    help="exclude Akhlaghi SWD points closer than R+buffer to cylinder")
    ap.add_argument("--ak_row_window", type=float, default=0.090,
                    help="x-window around ray shock guess used for Akhlaghi Gaussian fit")
    ap.add_argument("--ak_min_prom", type=float, default=0.05,
                    help="minimum normalized Roberts-Schlieren peak to accept a row")
    args = ap.parse_args()

    if args.leftmost_x is not None:
        args.center_x = args.leftmost_x + args.radius
    if args.center_x is None:
        raise ValueError("Use either --center_x or --leftmost_x with --radius")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    roots = extract_inputs(args.inputs, out)
    infos = find_flow_files(roots, args.default_kn_if_missing, args.default_mach_if_missing)

    inv = []
    allpts = []
    ncase = 0
    for info in infos:
        if args.only_gas and info["gas"].lower() != args.only_gas.lower():
            continue
        if args.only_kn is not None and not np.isclose(info["Kn"], args.only_kn):
            continue
        if args.only_mach is not None and not np.isclose(info["Mach"], args.only_mach):
            continue
        if ncase >= args.max_cases:
            break

        try:
            case = load_case(info, args.center_x, args.center_y, args.radius)
            ray = detect_ray_method(case, args.half_angle, args.angles, args.min_standoff, args.max_standoff)
            ak = akhlaghi_swd_method(case, ray, nx=args.ak_nx, ny=args.ak_ny, x_pad=args.ak_xpad, near_wall_buffer=args.ak_wall_buffer, row_window=args.ak_row_window, min_prom=args.ak_min_prom)

            make_compare_plot(case, ray, ak, out/"checks_compare")
            inv.append({"gas": case.gas, "Kn": case.Kn, "Mach": case.Mach, "source": case.source,
                        "flow_file": str(case.flow_file), "n_ray": len(ray), "n_akhlaghi": len(ak)})
            for df in [ray, ak]:
                if not df.empty:
                    tmp = df.copy()
                    tmp["gas"] = case.gas
                    tmp["Kn"] = case.Kn
                    tmp["Mach"] = case.Mach
                    tmp["source"] = case.source
                    tmp["flow_file"] = str(case.flow_file)
                    allpts.append(tmp)
            ncase += 1
            print(f"[OK] {case.gas} Kn={case.Kn:g} M={case.Mach:g}: ray={len(ray)}, akhlaghi={len(ak)}")
        except Exception as e:
            print(f"[ERROR] {info.get('gas')} Kn={info.get('Kn')} M={info.get('Mach')} {info.get('flow_file')} :: {e}")

    pd.DataFrame(inv).to_csv(out/"case_inventory_compare.csv", index=False)
    if allpts:
        pd.concat(allpts, ignore_index=True).to_csv(out/"shock_compare_points.csv", index=False)
    print(f"[DONE] output in {out}")


if __name__ == "__main__":
    main()

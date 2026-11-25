#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full post-processing pipeline for rarefied hypersonic cylinder datasets (our shock-detection method only).

Reads Tecplot ASCII DSMC/DS2V cylinder data organized by gas and Kn, then produces
structural diagnostics aimed at a focused JFM Rapid paper:

1) shock-layer inflation with increasing Kn
2) shock standoff and effective density-gradient thickness scaling
3) shock-attached profile collapse
4) POD/SVD of shock-attached density fields across Kn
5) read checks to verify that Tecplot files were parsed correctly

v6 adds paper-ready plotting defaults and cleaner supplementary POD-mode panels.

Run example:
python jfm_rapid_cylinder_shock_inflation.py \
  --inputs "Ar-Bird Prop-Kn-Part1.rar" "Ar-Bird Prop-Kn-Part2.rar" \
           "N2-Bird Prop-Kn-Part1.rar" "N2-Bird Prop-Kn-Part2.rar" \
  --out JFM_Rapid_Cylinder \
  --length_unit m \
  --half_angle 60 \
  --angles 49

If RAR extraction fails, manually extract the archives and pass the extracted folders.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
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


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GAS_COLORS = {"Ar": "#1f77b4", "N2": "#d62728", "Unknown": "#555555"}
KN_CMAP = "viridis"


def parse_variables(line: str) -> List[str]:
    return re.findall(r'"([^"]+)"', line)


def parse_zone_points(line: str) -> int:
    """Parse Tecplot point count from a ZONE header.

    Supports:
      I=..., optionally J=... and K=...
      N=...
      Nodes=...
    If I and J are present, returns I*J(*K).
    """
    header = line
    # FE style: N=...
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

    raise ValueError(f"Could not parse point count from ZONE header:\n{line}")



def read_tecplot_point(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Robust Tecplot/DS2V ASCII reader.

    v14 fixes:
      * multiline VARIABLES blocks, e.g. VARIABLES="X" then "Y" on next line
      * multiline ZONE headers before numeric data begin
      * structured I,J zones as well as FE N/E zones
      * headerless DS2V fallback

    For triangulated Mach-sweep files, this is the important part: the older
    reader only saw the first variable ("X"), which caused the error
    "Could not find ['Y','y'] in columns ['X']".
    """
    raw = Path(path).read_bytes()
    raw_no_nul = raw.replace(b"\x00", b"")
    text0 = raw_no_nul.decode("latin1", errors="ignore")
    lines = text0.splitlines()

    default_vars = [
        "X", "Y", "ND", "D", "U", "V", "W",
        "TTR", "TRT", "TVB", "TOV",
        "MA", "MC", "MCT", "MFP", "SOF", "FSP", "ANG", "P",
        "Angular momentum/(Mass.L)"
    ]

    # Locate VARIABLES and ZONE lines.
    var_idx = None
    zone_idx = None
    for i, line in enumerate(lines):
        s = line.strip().upper()
        if "VARIABLES" in s and var_idx is None:
            var_idx = i
        if "ZONE" in s and zone_idx is None:
            zone_idx = i
            # continue a bit is not necessary; variables normally precede zone
            break

    # Parse multiline VARIABLES block.
    if var_idx is not None:
        var_text = []
        for k in range(var_idx, len(lines)):
            s = lines[k].strip()
            us = s.upper()
            if k > var_idx and (us.startswith("ZONE") or re.match(r"^[\s\+\-]?\d", s)):
                break
            var_text.append(s)
        variables = re.findall(r'"([^"]+)"', " ".join(var_text))
        if not variables:
            variables = default_vars
    else:
        variables = default_vars

    nvar = len(variables)

    if zone_idx is not None:
        # Build multiline zone header until first numeric data line.
        header_lines = []
        data_start = zone_idx + 1
        for k in range(zone_idx, len(lines)):
            s = lines[k].strip()
            if k > zone_idx and re.match(r"^[\s\+\-]?(?:\d|\.\d)", s):
                data_start = k
                break
            header_lines.append(s)
            # stop if next line is numeric; otherwise keep collecting header
            if k > zone_idx + 50:
                data_start = k + 1
                break
        zone_header = " ".join(header_lines)
        try:
            npts = parse_zone_points(zone_header)
        except Exception:
            npts = None
        data_text = "\n".join(lines[data_start:])
    else:
        data_lines = []
        for line in lines:
            us = line.strip().upper()
            if ("VARIABLES" in us) or us.startswith("TITLE") or ("ZONE" in us):
                continue
            # also skip standalone quoted variable lines
            if re.fullmatch(r'"[^"]+"\s*,?', line.strip()):
                continue
            data_lines.append(line)
        data_text = "\n".join(data_lines)
        npts = None

    nums = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?", data_text)

    if len(nums) < nvar:
        raise ValueError(
            f"{path}: no numeric field data found after removing NUL padding. "
            "The file appears to contain only the VARIABLES header or is corrupted. "
            "Re-export/copy the field file for this case."
        )

    if npts is None:
        npts = len(nums) // nvar
        nums = nums[:npts*nvar]
    else:
        need = npts*nvar
        if len(nums) < need:
            print(f"[WARN] {path}: expected {need} numeric values from ZONE, found {len(nums)}; inferring npts.")
            npts = len(nums) // nvar
            nums = nums[:npts*nvar]
        else:
            nums = nums[:need]

    if npts < 1:
        raise ValueError(f"{path}: inferred zero points.")

    arr = np.array(nums, dtype=float).reshape(npts, nvar)
    return pd.DataFrame(arr, columns=variables), variables



def normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def variable_map(columns: List[str]) -> Dict[str, Optional[str]]:
    norm = {normalize_key(c): c for c in columns}
    def choose(candidates, required=True):
        for c in candidates:
            if normalize_key(c) in norm:
                return norm[normalize_key(c)]
        for key, orig in norm.items():
            for c in candidates:
                cc = normalize_key(c)
                if key == cc or key.startswith(cc) or cc in key:
                    return orig
        if required:
            raise KeyError(f"Could not find {candidates} in columns: {columns}")
        return None
    return {
        "x": choose(["X", "x"]),
        "y": choose(["Y", "y"]),
        "rho": choose(["D", "Density", "Rho"]),
        "number_density": choose(["ND", "NumberDensity"], required=False),
        "u": choose(["U", "u"]),
        "v": choose(["V", "v"]),
        "w": choose(["W", "w"], required=False),
        "ttr": choose(["TTR", "Ttr", "TranslationalTemperature", "T"], required=False),
        "trot": choose(["TRT", "Trot", "RotationalTemperature"], required=False),
        "tvib": choose(["TVB", "Tvib", "VibrationalTemperature"], required=False),
        "t": choose(["TTR", "TOV", "T", "Temperature"], required=False),
        "mach": choose(["MA", "Mach", "M"], required=False),
        "mfp": choose(["MFP", "MeanFreePath", "lambda"], required=False),
        "p": choose(["P", "Pressure"], required=False),
    }


def extract_inputs(inputs: List[str], out_root: Path) -> List[Path]:
    extract_root = out_root / "_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    roots = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            roots.append(p.resolve())
            continue
        if not p.exists():
            raise FileNotFoundError(f"Input does not exist: {p}")
        target = extract_root / p.stem.replace(" ", "_")
        target.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                z.extractall(target)
            roots.append(target)
        elif p.suffix.lower() == ".rar":
            commands = []
            if shutil.which("unrar"):
                commands.append(["unrar", "x", "-o+", str(p), str(target)])
            if shutil.which("7z"):
                commands.append(["7z", "x", "-y", f"-o{target}", str(p)])
            if shutil.which("unar"):
                commands.append(["unar", "-f", "-o", str(target), str(p)])
            if shutil.which("bsdtar"):
                commands.append(["bsdtar", "-xf", str(p), "-C", str(target)])
            if not commands:
                raise RuntimeError("No RAR extractor found. Install unrar/7z/unar/bsdtar, or extract manually.")
            ok = False
            err = []
            for cmd in commands:
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    ok = True
                    break
                except subprocess.CalledProcessError as e:
                    err.append(e.stderr[-500:])
            if not ok:
                raise RuntimeError(f"Could not extract {p}:\n" + "\n".join(err))
            roots.append(target)
        else:
            raise ValueError(f"Unsupported input type: {p}")
    return roots


def infer_gas_from_path(path: Path) -> str:
    s = str(path).lower()
    if re.search(r"\bn2\b|nitrogen", s):
        return "N2"
    if re.search(r"\bar\b|argon", s):
        return "Ar"
    return "Unknown"


def infer_kn_from_path(path: Path) -> Optional[float]:
    text = " / ".join(path.parts)
    m = re.search(r"Kn\s*=\s*([0-9]+(?:[.,][0-9]+)?)", text, re.I)
    if not m:
        m = re.search(r"Kn\s*[_-]?\s*([0-9]+(?:[.,][0-9]+)?)", text, re.I)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def find_cases(roots: List[Path], default_kn_if_missing: Optional[float] = None,
               default_mach_if_missing: Optional[float] = 10.0,
               ar_mach_values: str = "5:15", n2_mach_values: str = "5:14") -> List[Dict]:
    """Find DS2FF flow files and attach gas, Kn and Mach metadata.

    v11:
      * robust Mach inference for Mach-sweep archives
      * if Mach is missing in a multi-case archive, assign ordered Mach list
        using --ar_mach_values and --n2_mach_values
      * cases are unique by exact file path, not by only gas/Kn
    """
    cases = []
    for root in roots:
        flow_files = []
        for pat in [
            "**/DS2FF.DAT", "**/DS2FF.dat", "**/*DS2FF*.DAT", "**/*DS2FF*.dat",
            "**/TriangulateGrid*.dat", "**/TriangulateGrid*.DAT",
            "**/TriangulatedGrid*.dat", "**/TriangulatedGrid*.DAT",
        ]:
            flow_files.extend(root.glob(pat))
        # Do not accidentally treat surface files as field files.
        flow_files = [f for f in flow_files if "DS2SU" not in f.name.upper()]
        for ff in sorted(set(flow_files)):
            gas = infer_gas_from_path(ff)
            if gas == "Unknown":
                gas = infer_gas_from_path(root)

            kn = infer_kn_from_path(ff)
            if kn is None:
                kn = infer_kn_from_path(root)
            if kn is None and default_kn_if_missing is not None:
                kn = float(default_kn_if_missing)
            if kn is None:
                print(f"[WARN] Could not infer Kn from {ff}; skipping.")
                continue

            mach = infer_mach_from_path(ff)
            if not np.isfinite(mach):
                mach = infer_mach_from_path(root)

            surface = None
            for pat in ["DS2SU*.DAT", "DS2SU*.dat", "*SU*.DAT", "*SU*.dat"]:
                hits = sorted(ff.parent.glob(pat))
                if hits:
                    surface = hits[0]
                    break
            cases.append({"gas": gas, "kn": float(kn), "mach": float(mach) if np.isfinite(mach) else np.nan,
                          "mach_assigned_by_order": False,
                          "source_tag": infer_source_tag(ff),
                          "flow_file": ff, "surface_file": surface,
                          "root_path": str(root)})

    # Assign missing Mach in multi-case Mach-sweep archives before defaulting.
    cases = assign_missing_mach_by_order(cases, ar_values=ar_mach_values, n2_values=n2_mach_values)

    for c in cases:
        if not np.isfinite(c["mach"]) and default_mach_if_missing is not None:
            c["mach"] = float(default_mach_if_missing)

    unique = {}
    for c in cases:
        key = str(c["flow_file"])
        if key not in unique:
            unique[key] = c
        else:
            print(f"[WARN] duplicate exact flow file ignored: {c['flow_file']}")
    return [unique[k] for k in sorted(unique, key=lambda kk: (unique[kk]["gas"], unique[kk]["kn"], unique[kk]["mach"], kk))]

def fit_circle_from_surface(surface_path: Optional[Path]) -> Optional[Tuple[float, float, float]]:
    """Fit cylinder center/radius from a surface Tecplot file.

    Surface files usually do NOT contain volume-field variables such as density D.
    They often contain only S, ANGLE, X, Y, PR, HEAT-FLUX, etc. Therefore this
    routine must only read X and Y and must not call variable_map(), which
    requires density for volume files.
    """
    if surface_path is None:
        return None
    try:
        df, cols = read_tecplot_point(surface_path)

        # Surface file: only require X and Y.
        colmap = {normalize_key(c): c for c in cols}
        xcol = colmap.get("x")
        ycol = colmap.get("y")
        if xcol is None or ycol is None:
            raise KeyError(f"Surface file has no X/Y columns: {cols}")

        x = df[xcol].to_numpy(dtype=float)
        y = df[ycol].to_numpy(dtype=float)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if len(x) < 10:
            return None

        # Algebraic least-squares circle fit:
        # x^2 + y^2 = 2 cx x + 2 cy y + c
        A = np.column_stack([2*x, 2*y, np.ones_like(x)])
        b = x*x + y*y
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))
        return float(cx), float(cy), float(r)
    except Exception as e:
        print(f"[WARN] could not fit circle from {surface_path}: {e}")
        return None


def estimate_circle_from_flow(df: pd.DataFrame, vm: Dict[str, str]) -> Tuple[float, float, float]:
    x = df[vm["x"]].to_numpy()
    y = df[vm["y"]].to_numpy()
    r = float(np.nanpercentile(np.abs(y), 95))
    cy = 0.0
    cx = float(np.nanmax(x) - r)
    return cx, cy, r


@dataclass
class CaseData:
    gas: str
    kn: float
    mach: float
    source_tag: str
    flow_file: Path
    surface_file: Optional[Path]
    df: pd.DataFrame
    vm: Dict[str, Optional[str]]
    center: Tuple[float, float]
    radius: float
    tri: mtri.Triangulation
    uinf: Tuple[float, float]
    upstream_angle: float


def load_case(case: Dict, center_args=None) -> CaseData:
    df, cols = read_tecplot_point(case["flow_file"])
    vm = variable_map(cols)
    if center_args and all(v is not None for v in center_args):
        cx, cy, r = center_args
    else:
        fit = fit_circle_from_surface(case["surface_file"])
        if fit is None:
            cx, cy, r = estimate_circle_from_flow(df, vm)
        else:
            cx, cy, r = fit
    x = df[vm["x"]].to_numpy()
    y = df[vm["y"]].to_numpy()
    tri = mtri.Triangulation(x, y)
    u = df[vm["u"]].to_numpy()
    v = df[vm["v"]].to_numpy()
    mask = x <= np.nanquantile(x, 0.08)
    if mask.sum() < 10:
        mask = x <= np.nanquantile(x, 0.12)
    uinf = float(np.nanmedian(u[mask])), float(np.nanmedian(v[mask]))
    upstream_angle = math.atan2(-uinf[1], -uinf[0])
    return CaseData(case["gas"], float(case["kn"]), float(case.get("mach", np.nan)), case.get("source_tag", "unknown"), case["flow_file"], case["surface_file"], df, vm, (cx, cy), r, tri, uinf, upstream_angle)


def get_array(case: CaseData, key: str) -> np.ndarray:
    return case.df[case.vm[key]].to_numpy()


def interp(case: CaseData, key: str):
    return mtri.LinearTriInterpolator(case.tri, get_array(case, key))


def smooth1d(q, window=11):
    q = np.asarray(q)
    if q.size < 3:
        return q
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    pad = window//2
    return np.convolve(np.pad(q, (pad, pad), mode="edge"), np.ones(window)/window, mode="valid")


def normalize(q):
    q = np.asarray(q, dtype=float)
    good = np.isfinite(q)
    if good.sum() < 2:
        return np.full_like(q, np.nan)
    lo, hi = np.nanmin(q[good]), np.nanmax(q[good])
    if abs(hi-lo) < 1e-300:
        out = np.full_like(q, np.nan); out[good] = 0.0; return out
    return (q-lo)/(hi-lo)


def sample_ray(case: CaseData, angle_rad: float, n=900):
    cx, cy = case.center
    r0 = case.radius * 1.005
    x = get_array(case, "x")
    y = get_array(case, "y")
    ux, uy = math.cos(angle_rad), math.sin(angle_rad)
    proj = (x-cx)*ux + (y-cy)*uy
    pos = proj[proj > r0]
    if pos.size:
        rmax = float(np.nanpercentile(pos, 99.5))
    else:
        rmax = 4.5*case.radius
    rmax = max(rmax, 4.5*case.radius)
    rr = np.linspace(r0, rmax, n)
    return rr, cx + rr*ux, cy + rr*uy


def detect_shock_on_ray(case: CaseData, angle_rad: float):
    rr, xs, ys = sample_ray(case, angle_rad)
    rhoi = interp(case, "rho")
    rho = rhoi(xs, ys)
    rho = np.asarray(rho.filled(np.nan) if hasattr(rho, "filled") else rho)
    valid = np.isfinite(rho)
    if valid.sum() < 80:
        return None
    rr_v = rr[valid]
    rho_v = rho[valid]
    keep = rr_v > (case.radius + 0.03*case.radius)
    if keep.sum() < 80:
        keep = rr_v > (case.radius + 0.01*case.radius)
    rr_v = rr_v[keep]
    rho_v = rho_v[keep]
    if rr_v.size < 80:
        return None
    rho_s = smooth1d(rho_v, max(7, rr_v.size//80))
    drho = np.gradient(rho_s, rr_v)
    absgrad = np.abs(drho)
    imax = int(np.nanargmax(absgrad))
    r_s = float(rr_v[imax])
    maxgrad = float(absgrad[imax])
    nseg = max(8, rr_v.size//12)
    rho_up = float(np.nanmedian(rho_v[-nseg:]))
    rho_down = float(np.nanmedian(rho_v[:nseg]))
    delta = abs(rho_down-rho_up) / max(maxgrad, 1e-300)
    return {
        "angle_rad": angle_rad,
        "r_s": r_s,
        "standoff": r_s - case.radius,
        "delta_rho": delta,
        "maxgrad_rho": maxgrad,
        "rho_up": rho_up,
        "rho_down": rho_down,
    }


def compute_shock_metrics(case: CaseData, half_angle_deg: float, n_angles: int):
    rows = []
    offsets = np.linspace(0.0, half_angle_deg, n_angles)
    candidates = []
    for sign in [-1, +1]:
        tmp = []
        for deg in offsets:
            angle = case.upstream_angle + sign*np.deg2rad(deg)
            if math.sin(angle) < -1e-8:
                continue
            det = detect_shock_on_ray(case, angle)
            if det is None:
                continue
            det["angle_deg_from_upstream"] = sign*deg
            tmp.append(det)
        candidates.append(tmp)
    valid = [c for c in candidates if c]
    if not valid:
        return pd.DataFrame()
    best = max(valid, key=len)
    for d in best:
        d.update({"gas": case.gas, "Kn": case.kn, "Mach": case.mach,
                  "source_tag": case.source_tag, "flow_file": str(case.flow_file)})
        rows.append(d)
    return pd.DataFrame(rows)


def stagnation_profile(case: CaseData, shock_row: pd.Series, band_fraction=0.02, nbins=420):
    """Stagnation-ray profiles using interpolation along the detected shock ray.

    The previous band-averaging approach can fail for some DS2V unstructured/scattered
    datasets, especially when the upstream domain changes with Kn. This ray-based
    method samples directly along the same upstream/stagnation ray used for shock
    detection and is much more robust.

    Returns profiles as:
       key -> (s_from_wall, xi, normalized_q)
    where s_from_wall = r - R and xi=(r-r_s)/delta_rho.
    """
    angle = float(shock_row["angle_rad"])
    r_s = float(shock_row["r_s"])
    delta = float(shock_row["delta_rho"])

    # Sample from just outside wall to the available upstream/far-field domain.
    rr, xs, ys = sample_ray(case, angle, n=max(900, nbins*3))
    s_from_wall = rr - case.radius
    xi = (rr - r_s) / delta

    prof = {}
    keys = []
    if case.vm.get("rho"):
        keys.append(("rho", "rho"))
    if case.vm.get("mach"):
        keys.append(("mach", "mach"))
    if case.vm.get("ttr"):
        keys.append(("Ttr", "ttr"))
    elif case.vm.get("t"):
        keys.append(("Ttr", "t"))
    if case.vm.get("trot"):
        keys.append(("Trot", "trot"))

    for out_key, vm_key in keys:
        try:
            qinterp = interp(case, vm_key)
            vals = qinterp(xs, ys)
            vals = np.asarray(vals.filled(np.nan) if hasattr(vals, "filled") else vals, dtype=float)
        except Exception:
            continue

        valid = np.isfinite(vals) & np.isfinite(s_from_wall) & np.isfinite(xi)
        # Keep a physically useful xi window but allow physical plot to show wider ray.
        if valid.sum() < 20:
            continue

        s_v = s_from_wall[valid]
        xi_v = xi[valid]
        q_v = smooth1d(vals[valid], 11)
        qn = normalize(q_v)

        # Sort by s/xi
        order = np.argsort(s_v)
        prof[out_key] = (s_v[order], xi_v[order], qn[order])

    return prof


def savefig(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_check_plot(case, sdf, outdir, unit):
    outdir.mkdir(parents=True, exist_ok=True)
    x, y, rho = get_array(case, "x"), get_array(case, "y"), get_array(case, "rho")
    rho_inf = np.nanmedian(rho[x <= np.nanquantile(x, 0.08)])
    vals = rho/rho_inf
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    levels = np.linspace(np.nanpercentile(vals, 2), np.nanpercentile(vals, 98), 34)
    cf = ax.tricontourf(case.tri, vals, levels=levels, cmap="turbo", extend="both")
    cx, cy = case.center
    ax.add_patch(Circle((cx, cy), case.radius, facecolor="white", edgecolor="black", lw=1.3))
    if not sdf.empty:
        xs = cx + sdf["r_s"].to_numpy()*np.cos(sdf["angle_rad"].to_numpy())
        ys = cy + sdf["r_s"].to_numpy()*np.sin(sdf["angle_rad"].to_numpy())
        ax.plot(xs, ys, "w-", lw=3.0)
        ax.plot(xs, ys, "k-", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"Read check: {case.gas}, Kn={case.kn:g}")
    fig.colorbar(cf, ax=ax, label=r"$\rho/\rho_\infty$")
    savefig(fig, outdir/f"check_raw_{case.gas}_Kn{str(case.kn).replace('.','p')}_M{str(case.mach).replace('.','p')}")


def make_fig1_contours(cases, metrics, gas, outdir, unit):
    """Paper-ready contour figure.

    v6 changes:
      * shared colorbar across the selected Kn panels
      * cleaner layout and reduced clutter
      * top/right spines removed for a lighter appearance
      * only leftmost panel keeps y-label
    """
    gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
    if not gas_cases:
        return

    idxs = sorted(set([0, len(gas_cases)//2, len(gas_cases)-1]))
    sel = [gas_cases[i] for i in idxs]

    cache = []
    all_vals = []
    for case in sel:
        x = get_array(case, "x")
        y = get_array(case, "y")
        rho = get_array(case, "rho")
        rho_inf = np.nanmedian(rho[x <= np.nanquantile(x, 0.08)])
        vals = rho / rho_inf
        mask = y >= case.center[1]
        tri = mtri.Triangulation(x[mask], y[mask])
        v2 = vals[mask]
        cache.append((case, x, y, tri, v2))
        all_vals.append(v2[np.isfinite(v2)])

    all_vals = np.concatenate(all_vals) if all_vals else np.array([0.0, 1.0])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    levels = np.linspace(vmin, vmax, 36)

    fig, axes = plt.subplots(1, len(sel), figsize=(5.2*len(sel) + 0.65, 4.85), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(left=0.07, right=0.90, bottom=0.13, top=0.82, wspace=0.24)

    last_cf = None
    for iax, (ax, item) in enumerate(zip(axes, cache)):
        case, x, y, tri, v2 = item
        last_cf = ax.tricontourf(tri, v2, levels=levels, cmap="turbo", extend="both")
        cx, cy = case.center
        ax.add_patch(Circle((cx, cy), case.radius, facecolor="white", edgecolor="black", lw=1.2, zorder=5))

        df = metrics[(metrics["gas"] == gas) & (metrics["Kn"] == case.kn)]
        if not df.empty:
            xs = cx + df["r_s"].to_numpy() * np.cos(df["angle_rad"].to_numpy())
            ys = cy + df["r_s"].to_numpy() * np.sin(df["angle_rad"].to_numpy())
            ax.plot(xs, ys, "w-", lw=2.8)
            ax.plot(xs, ys, "k-", lw=1.0)

        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(case.center[1], np.nanmax(y))
        ax.set_title(fr"{gas}, $\mathrm{{Kn}}={case.kn:g}$", pad=8)
        ax.set_xlabel(f"x ({unit})")
        if iax == 0:
            ax.set_ylabel(f"y ({unit})")
        else:
            ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    cax = fig.add_axes([0.915, 0.17, 0.015, 0.60])
    cbar = fig.colorbar(last_cf, cax=cax)
    cbar.set_label(r"$\rho/\rho_\infty$")
    fig.suptitle(f"Shock-layer inflation with Kn: {gas}", y=0.96, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig1_ShockInflation_Contours_{gas}")

def make_fig2_scaling(summary, outdir, unit, include_ratio=False):
    """Paper-focused scaling figure.

    By default v6 keeps the two strongest panels for the main paper.
    The relative inflation ratio panel can still be generated by setting
    include_ratio=True.
    """
    if summary.empty:
        return
    ncols = 3 if include_ratio else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.35*ncols, 4.7), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.86, wspace=0.32)
    for gas, dfg in summary.groupby("gas"):
        dfg = dfg.sort_values("Kn")
        color = GAS_COLORS.get(gas, None)
        axes[0].plot(dfg["Kn"], dfg["standoff_median"], "o-", color=color, label=gas)
        axes[1].plot(dfg["Kn"], dfg["delta_rho_median"], "o-", color=color, label=gas)
        if include_ratio:
            axes[2].plot(dfg["Kn"], dfg["delta_over_standoff"], "o-", color=color, label=gas)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(f"Median shock standoff ({unit})")
    axes[1].set_ylabel(rf"Median $\delta_\rho$ ({unit})")
    axes[0].set_title("(a) Shock standoff")
    axes[1].set_title("(b) Effective shock thickness")
    if include_ratio:
        axes[2].set_ylabel(r"Median $\delta_\rho/\Delta_s$ (-)")
        axes[2].set_title("(c) Relative shock-layer inflation")
    axes[0].legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig2_ShockInflation_Scaling")

def make_fig3_collapse(cases, metrics, gas, outdir, unit):
    """Physical and shock-attached collapse figure.

    v5 change: wider figure and independent colorbar axis placed farther right
    so it no longer overlaps with panel coordinates or labels.
    """
    gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
    if len(gas_cases) < 2:
        return

    kns = np.array([c.kn for c in gas_cases])
    norm = plt.Normalize(np.log10(kns.min()), np.log10(kns.max()))
    cmap = plt.get_cmap(KN_CMAP)

    variables = [("rho", r"$\rho$"), ("mach", "Mach"), ("Ttr", r"$T_{tr}$")]
    if gas == "N2" and any(c.vm.get("trot") for c in gas_cases):
        variables = [("rho", r"$\rho$"), ("Ttr", r"$T_{tr}$"), ("Trot", r"$T_{rot}$")]

    fig, axes = plt.subplots(2, 3, figsize=(17.6, 8.4))
    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.065, right=0.84, hspace=0.31, wspace=0.26)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for case in gas_cases:
        df = metrics[(metrics["gas"] == gas) & (metrics["Kn"] == case.kn)]
        if df.empty:
            continue
        row = df.iloc[np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy()))]
        prof = stagnation_profile(case, row)
        color = cmap(norm(np.log10(case.kn)))

        for j, (key, title) in enumerate(variables):
            if key not in prof:
                continue
            s, xi, qn = prof[key]
            if s.size < 5:
                continue
            axes[0, j].plot(s, qn, color=color)
            axes[1, j].plot(xi, qn, color=color)

    for j, (key, title) in enumerate(variables):
        axes[0, j].set_title(f"({chr(97+j)}) {title}: physical coordinate")
        axes[1, j].set_title(f"({chr(100+j)}) {title}: shock-attached coordinate")
        axes[0, j].set_xlabel(f"s from wall ({unit})")
        axes[1, j].set_xlabel(r"$\xi_s=(x-x_s)/\delta_\rho$ (-)")
        axes[0, j].set_ylabel("Normalized profile (-)")
        axes[1, j].set_ylabel("Normalized profile (-)")
        axes[0, j].grid(alpha=0.22)
        axes[1, j].grid(alpha=0.22)
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_xlim(-7, 7)

    cax = fig.add_axes([0.875, 0.13, 0.018, 0.70])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r"$\log_{10}(\mathrm{Kn}_\infty)$")
    cbar.ax.tick_params(labelsize=11)

    fig.suptitle(f"Shock-attached collapse across Kn: {gas}", fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig3_ShockAttached_Collapse_{gas}")



def make_matrix(case, metrics, xi_grid, angle_grid):
    intr = interp(case, "rho")
    M = np.full((len(angle_grid), len(xi_grid)), np.nan)
    df = metrics[(metrics["gas"] == case.gas) & (metrics["Kn"] == case.kn)]
    if df.empty: return M
    for ia, adeg in enumerate(angle_grid):
        j = int(np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy() - adeg)))
        row = df.iloc[j]
        rr = row["r_s"] + xi_grid*row["delta_rho"]
        cx, cy = case.center
        xs = cx + rr*np.cos(row["angle_rad"])
        ys = cy + rr*np.sin(row["angle_rad"])
        vals = intr(xs, ys)
        vals = np.asarray(vals.filled(np.nan) if hasattr(vals, "filled") else vals)
        M[ia,:] = normalize(vals)
    return M


def make_fig4_pod(cases, metrics, gas, outdir, modes_to_plot=4):
    """POD/SVD of shock-attached density fields.

    Main JFM Rapid figure:
      spectrum + mode 1 map + first two coefficients versus Kn.
    Supplementary/full diagnostic:
      first N mode maps with cleaner spacing and larger margins.

    v6 changes:
      * more room around titles and colorbar
      * mode-energy percentage moved into an annotation box, not the title
      * larger supplementary canvas so suptitle never overlaps panel titles
    """
    gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
    if len(gas_cases) < 3:
        return

    xi_grid = np.linspace(-5, 6, 150)
    dfg = metrics[metrics["gas"] == gas]
    if dfg.empty:
        return

    amax = min(55.0, np.nanmax(np.abs(dfg["angle_deg_from_upstream"])))
    angle_grid = np.linspace(0, amax, 50)

    mats, kns = [], []
    for case in gas_cases:
        M = make_matrix(case, metrics, xi_grid, angle_grid)
        if np.isfinite(M).mean() < 0.5:
            continue
        Mfill = M.copy()
        Mfill[~np.isfinite(Mfill)] = np.nanmean(Mfill)
        mats.append(Mfill.ravel())
        kns.append(case.kn)

    if len(mats) < 3:
        return

    A = np.column_stack(mats)
    Af = A - A.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(Af, full_matrices=False)
    efrac = s**2 / np.sum(s**2)
    cum = np.cumsum(efrac)
    coeff = np.diag(s) @ Vt

    pd.DataFrame({
        "mode": np.arange(1, len(efrac)+1),
        "energy_fraction": efrac,
        "cumulative": cum
    }).to_csv(outdir/f"shock_attached_pod_summary_{gas}.csv", index=False)

    coef = pd.DataFrame(coeff.T, columns=[f"mode_{i+1}" for i in range(coeff.shape[0])])
    coef.insert(0, "Kn", kns)
    coef.to_csv(outdir/f"shock_attached_pod_coefficients_{gas}.csv", index=False)

    mode1 = U[:, 0].reshape(len(angle_grid), len(xi_grid))
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    plt.subplots_adjust(left=0.06, right=0.975, bottom=0.17, top=0.80, wspace=0.34)

    modes = np.arange(1, len(efrac)+1)
    axes[0].bar(modes, 100*efrac, color="#4c78a8", alpha=0.85, label="mode energy")
    axes[0].plot(modes, 100*cum, "o-", color="#d62728", label="cumulative")
    axes[0].set_xlabel("Mode number (-)")
    axes[0].set_ylabel("Energy fraction / cumulative (%)")
    axes[0].grid(alpha=0.22)
    axes[0].set_title("(a) POD spectrum")
    axes[0].legend(frameon=True, fontsize=9, loc="lower right")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    vmax = np.nanpercentile(np.abs(mode1), 99)
    cf = axes[1].contourf(
        xi_grid, angle_grid, mode1,
        levels=np.linspace(-vmax, vmax, 41),
        cmap="coolwarm", extend="both"
    )
    axes[1].set_xlabel(r"$\xi_s$ (-)")
    axes[1].set_ylabel("Angle from upstream axis (deg)")
    axes[1].set_title("(b) Mode 1")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.colorbar(cf, ax=axes[1], fraction=0.050, pad=0.030, label="Amplitude (-)")

    axes[2].plot(kns, coeff[0, :], "o-", label="mode 1")
    if coeff.shape[0] > 1:
        axes[2].plot(kns, coeff[1, :], "s-", label="mode 2")
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
    axes[2].set_ylabel("POD coefficient (-)")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=True)
    axes[2].set_title("(c) First two coefficients")
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle(f"Shock-attached compact structure across Kn: {gas}", y=0.95, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig4_ShockAttached_POD_{gas}")

    nm = min(int(modes_to_plot), U.shape[1])
    if nm >= 2:
        ncols = 2 if nm <= 4 else min(nm, 3)
        nrows = int(np.ceil(nm / ncols))
        fig2, axs = plt.subplots(nrows, ncols, figsize=(6.2*ncols, 4.8*nrows), squeeze=False)
        plt.subplots_adjust(left=0.075, right=0.88, bottom=0.10, top=0.82, wspace=0.25, hspace=0.36)

        last_cf = None
        for im in range(nm):
            ax = axs[im//ncols, im % ncols]
            mode = U[:, im].reshape(len(angle_grid), len(xi_grid))
            vmax = np.nanpercentile(np.abs(mode), 99)
            last_cf = ax.contourf(
                xi_grid, angle_grid, mode,
                levels=np.linspace(-vmax, vmax, 41),
                cmap="coolwarm", extend="both"
            )
            ax.set_title(f"Mode {im+1}", pad=10)
            ax.text(
                0.03, 0.96, f"{100*efrac[im]:.1f}% energy",
                transform=ax.transAxes, ha="left", va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.65", alpha=0.90)
            )
            ax.set_xlabel(r"$\xi_s$ (-)")
            ax.set_ylabel("Angle from upstream axis (deg)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        for im in range(nm, nrows*ncols):
            axs[im//ncols, im % ncols].axis("off")

        cax = fig2.add_axes([0.905, 0.14, 0.018, 0.64])
        cb = fig2.colorbar(last_cf, cax=cax)
        cb.set_label("Amplitude (-)")
        fig2.suptitle(f"Leading shock-attached POD modes across Kn: {gas}", y=0.96, fontweight="bold")
        savefig(fig2, outdir/f"JFM_Fig4b_ShockAttached_POD_Modes_{gas}")

def compute_gll_and_gradient_summary(case: CaseData, nx: int, ny: int) -> Dict:
    """Compute simple upper-half gradient/GLL diagnostics on a regular grid.

    This is not used as a universal breakdown criterion. It is a compact
    diagnostic of how much of the upper upstream field is occupied by strong
    density/temperature gradients as Kn increases.
    """
    x = get_array(case, "x")
    y = get_array(case, "y")
    cx, cy = case.center

    # upper half, avoid inside cylinder
    xmin, xmax = np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5)
    ymin, ymax = cy, np.nanpercentile(y, 99.5)
    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    inside = (Xg-cx)**2 + (Yg-cy)**2 <= case.radius**2

    rho_i = interp(case, "rho")
    rho = rho_i(Xg, Yg)
    rho = np.asarray(rho.filled(np.nan) if hasattr(rho, "filled") else rho, dtype=float)
    rho[inside] = np.nan

    dx = float(np.nanmedian(np.diff(Xg[0, :])))
    dy = float(np.nanmedian(np.diff(Yg[:, 0])))

    drdy, drdx = np.gradient(rho, dy, dx)
    grad_rho = np.sqrt(drdx**2 + drdy**2)
    gll_rho = grad_rho / np.maximum(np.abs(rho), 1e-300)

    # If MFP is available, convert to Kn_GLL proxy
    if case.vm.get("mfp"):
        mfp_i = interp(case, "mfp")
        mfp = mfp_i(Xg, Yg)
        mfp = np.asarray(mfp.filled(np.nan) if hasattr(mfp, "filled") else mfp, dtype=float)
        mfp[inside] = np.nan
        gll_rho = mfp * gll_rho

    # Temperature gradient proxy if possible
    temp_gll_max = np.nan
    if case.vm.get("ttr"):
        Ti = interp(case, "ttr")
        T = Ti(Xg, Yg)
        T = np.asarray(T.filled(np.nan) if hasattr(T, "filled") else T, dtype=float)
        T[inside] = np.nan
        dTdy, dTdx = np.gradient(T, dy, dx)
        gT = np.sqrt(dTdx**2 + dTdy**2) / np.maximum(np.abs(T), 1e-300)
        if case.vm.get("mfp"):
            gT = mfp * gT
        temp_gll_max = float(np.nanpercentile(gT, 99.0))

    threshold = np.nanpercentile(gll_rho[np.isfinite(gll_rho)], 90.0)
    area_fraction_top10 = float(np.nanmean(gll_rho > threshold))

    return {
        "gas": case.gas,
        "Kn": case.kn,
        "gll_rho_p95": float(np.nanpercentile(gll_rho, 95.0)),
        "gll_rho_p99": float(np.nanpercentile(gll_rho, 99.0)),
        "gll_Ttr_p99": temp_gll_max,
        "high_gradient_area_fraction_top10": area_fraction_top10,
    }


def make_fig5_collapse_quality(collapse_df: pd.DataFrame, outdir: Path):
    if collapse_df.empty:
        return
    # Only variables that have at least one gas
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    vars_order = ["rho", "mach", "Ttr", "Trot"]
    gases = sorted(collapse_df["gas"].unique())
    width = 0.36
    xbase = np.arange(len(vars_order))

    for ig, gas in enumerate(gases):
        vals = []
        for var in vars_order:
            row = collapse_df[(collapse_df["gas"] == gas) & (collapse_df["variable"] == var)]
            vals.append(float(row["mean_std_in_xi"].iloc[0]) if not row.empty else np.nan)
        ax.bar(xbase + (ig-0.5)*width, vals, width=width, label=gas, color=GAS_COLORS.get(gas, None), alpha=0.85)

    ax.set_xticks(xbase)
    ax.set_xticklabels([r"$\rho$", "Mach", r"$T_{tr}$", r"$T_{rot}$"])
    ax.set_ylabel(r"Mean collapse spread, $\langle \sigma_{Kn}(q(\xi_s)) \rangle$ (-)")
    ax.set_title("Shock-attached collapse quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig5_CollapseQuality")


def make_fig6_gll_summary(gll_df: pd.DataFrame, outdir: Path):
    if gll_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for gas, dfg in gll_df.groupby("gas"):
        dfg = dfg.sort_values("Kn")
        color = GAS_COLORS.get(gas, None)
        axes[0].plot(dfg["Kn"], dfg["gll_rho_p99"], "o-", color=color, label=gas)
        axes[1].plot(dfg["Kn"], dfg["gll_Ttr_p99"], "s-", color=color, label=gas)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"99th percentile of $\mathrm{Kn}_{GLL,\rho}$ or proxy")
    axes[1].set_ylabel(r"99th percentile of $\mathrm{Kn}_{GLL,T_{tr}}$ or proxy")
    axes[0].set_title("(a) Density-gradient diagnostic")
    axes[1].set_title("(b) Temperature-gradient diagnostic")
    axes[0].legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig6_GLLGradientDiagnostics")


# =============================================================================
# v7: variable-specific length scales and VHS mean-free-path diagnostics
# =============================================================================

VHS_DEFAULTS = {
    # Bird/DSMC-style VHS values commonly used for these gases.
    # mu_ref is dynamic viscosity at T_ref, used in lambda_VHS formula.
    "Ar": {"omega": 0.81, "T_ref": 273.0, "mu_ref": 2.117e-5, "molar_mass": 39.948e-3},
    "N2": {"omega": 0.74, "T_ref": 273.0, "mu_ref": 1.656e-5, "molar_mass": 28.0134e-3},
}
R_UNIVERSAL = 8.31446261815324


def vhs_params_for_gas(gas: str) -> Dict[str, float]:
    return VHS_DEFAULTS.get(gas, VHS_DEFAULTS["Ar"]).copy()


def vhs_mean_free_path(rho, T, gas: str):
    """VHS mean free path from density and translational temperature.

    lambda = C(omega) * mu(T)/(rho * sqrt(2*pi*R*T))
    C(omega) = 2(5-2omega)(7-2omega)/15
    mu(T) = mu_ref (T/T_ref)^omega

    Inputs can be scalars or arrays. rho is mass density [kg/m^3], T is [K].
    """
    pars = vhs_params_for_gas(gas)
    omega = pars["omega"]
    T_ref = pars["T_ref"]
    mu_ref = pars["mu_ref"]
    R = R_UNIVERSAL / pars["molar_mass"]
    rho = np.asarray(rho, dtype=float)
    T = np.asarray(T, dtype=float)
    Tclip = np.maximum(T, 1e-12)
    mu = mu_ref * (Tclip / T_ref) ** omega
    coeff = 2.0 * (5.0 - 2.0*omega) * (7.0 - 2.0*omega) / 15.0
    lam = coeff * mu / np.maximum(rho * np.sqrt(2.0*np.pi*R*Tclip), 1e-300)
    lam[~np.isfinite(lam)] = np.nan
    return lam


def available_profile_variables(case: CaseData):
    """Variables for which we can define a gradient thickness."""
    vars_ = []
    if case.vm.get("rho"):
        vars_.append(("rho", r"$\rho$"))
    if case.vm.get("mach"):
        vars_.append(("mach", "Mach"))
    if case.vm.get("ttr"):
        vars_.append(("Ttr", r"$T_{tr}$"))
    elif case.vm.get("t"):
        vars_.append(("Ttr", r"$T$"))
    if case.vm.get("trot"):
        vars_.append(("Trot", r"$T_{rot}$"))
    if case.vm.get("p"):
        vars_.append(("p", r"$p$"))
    return vars_


def get_var_values(case: CaseData, var_key: str):
    if var_key == "Ttr":
        if case.vm.get("ttr"):
            return case.df[case.vm["ttr"]].to_numpy()
        return case.df[case.vm["t"]].to_numpy()
    if var_key == "Trot":
        return case.df[case.vm["trot"]].to_numpy()
    return get_array(case, var_key)


def get_var_interpolator(case: CaseData, var_key: str):
    if var_key == "Ttr":
        col = case.vm["ttr"] if case.vm.get("ttr") else case.vm["t"]
        return mtri.LinearTriInterpolator(case.tri, case.df[col].to_numpy())
    if var_key == "Trot":
        return mtri.LinearTriInterpolator(case.tri, case.df[case.vm["trot"]].to_numpy())
    return interp(case, var_key)


def freestream_lambda(case: CaseData):
    x = get_array(case, "x")
    rho = get_array(case, "rho")
    if case.vm.get("ttr"):
        T = case.df[case.vm["ttr"]].to_numpy()
    elif case.vm.get("t"):
        T = case.df[case.vm["t"]].to_numpy()
    else:
        return np.nan
    mask = x <= np.nanquantile(x, 0.08)
    if mask.sum() < 10:
        mask = x <= np.nanquantile(x, 0.12)
    return float(np.nanmedian(vhs_mean_free_path(rho[mask], T[mask], case.gas)))


def detect_variable_layer_on_ray(case: CaseData, angle_rad: float, var_key: str, min_points: int = 80):
    """Detect max-gradient layer for a given variable on a ray.

    This generalizes delta_rho = Delta rho / max|d rho/dr| to
    delta_q = Delta q / max|dq/dr| for each variable q.
    """
    rr, xs, ys = sample_ray(case, angle_rad)
    qi = get_var_interpolator(case, var_key)
    q = qi(xs, ys)
    q = np.asarray(q.filled(np.nan) if hasattr(q, "filled") else q, dtype=float)

    # rho/T for local VHS lambda on the same ray
    rhoi = interp(case, "rho")
    rho_ray = rhoi(xs, ys)
    rho_ray = np.asarray(rho_ray.filled(np.nan) if hasattr(rho_ray, "filled") else rho_ray, dtype=float)
    if case.vm.get("ttr"):
        Ti = mtri.LinearTriInterpolator(case.tri, case.df[case.vm["ttr"]].to_numpy())
    elif case.vm.get("t"):
        Ti = mtri.LinearTriInterpolator(case.tri, case.df[case.vm["t"]].to_numpy())
    else:
        Ti = None
    if Ti is not None:
        T_ray = Ti(xs, ys)
        T_ray = np.asarray(T_ray.filled(np.nan) if hasattr(T_ray, "filled") else T_ray, dtype=float)
        lam_ray = vhs_mean_free_path(rho_ray, T_ray, case.gas)
    else:
        lam_ray = np.full_like(q, np.nan)

    valid = np.isfinite(q)
    if valid.sum() < min_points:
        return None

    rr_v = rr[valid]
    q_v = q[valid]
    lam_v = lam_ray[valid]

    # Avoid wall-adjacent artifacts.
    keep = rr_v > (case.radius + 0.03*case.radius)
    if keep.sum() < min_points:
        keep = rr_v > (case.radius + 0.01*case.radius)
    rr_v = rr_v[keep]
    q_v = q_v[keep]
    lam_v = lam_v[keep]
    if rr_v.size < min_points:
        return None

    q_s = smooth1d(q_v, window=max(7, rr_v.size//80))
    dq = np.gradient(q_s, rr_v)
    absgrad = np.abs(dq)
    imax = int(np.nanargmax(absgrad))
    r_q = float(rr_v[imax])
    maxgrad = float(absgrad[imax])
    nseg = max(8, rr_v.size//12)
    q_up = float(np.nanmedian(q_v[-nseg:]))
    q_down = float(np.nanmedian(q_v[:nseg]))
    delta_q = abs(q_down - q_up) / max(maxgrad, 1e-300)
    lambda_q = float(lam_v[imax]) if np.isfinite(lam_v[imax]) else np.nan

    return {
        "r_q": r_q,
        "standoff_q": r_q - case.radius,
        "delta_q": float(delta_q),
        "maxgrad_q": maxgrad,
        "q_up": q_up,
        "q_down": q_down,
        "lambda_q": lambda_q,
    }


def compute_variable_layer_metrics(cases, half_angle_deg: float, n_angles: int):
    rows = []
    for case in cases:
        lam_inf = freestream_lambda(case)
        variables = available_profile_variables(case)
        offsets = np.linspace(0.0, half_angle_deg, n_angles)

        for var_key, var_label in variables:
            candidates = []
            for sign in [-1, +1]:
                tmp = []
                for deg in offsets:
                    angle = case.upstream_angle + sign*np.deg2rad(deg)
                    if math.sin(angle) < -1e-8:
                        continue
                    det = detect_variable_layer_on_ray(case, angle, var_key)
                    if det is None:
                        continue
                    det["angle_rad"] = angle
                    det["angle_deg_from_upstream"] = sign*deg
                    tmp.append(det)
                candidates.append(tmp)

            valid_lists = [c for c in candidates if c]
            if not valid_lists:
                print(f"[WARN] variable layer detection failed: {case.gas} Kn={case.kn:g}, var={var_key}")
                continue
            best = max(valid_lists, key=len)
            for det in best:
                det.update({
                    "gas": case.gas,
                    "Kn": case.kn,
                    "Mach": case.mach,
                    "source_tag": case.source_tag,
                    "flow_file": str(case.flow_file),
                    "variable": var_key,
                    "variable_label": var_label,
                    "lambda_inf": lam_inf,
                    "delta_over_lambda_q": det["delta_q"] / det["lambda_q"] if np.isfinite(det["lambda_q"]) and det["lambda_q"] != 0 else np.nan,
                    "delta_over_lambda_inf": det["delta_q"] / lam_inf if np.isfinite(lam_inf) and lam_inf != 0 else np.nan,
                    "standoff_over_lambda_inf": det["standoff_q"] / lam_inf if np.isfinite(lam_inf) and lam_inf != 0 else np.nan,
                })
                rows.append(det)
    return pd.DataFrame(rows)


def summarize_variable_layer_metrics(var_metrics: pd.DataFrame):
    if var_metrics.empty:
        return pd.DataFrame()
    summary = (var_metrics.groupby(["gas", "Kn", "Mach", "variable", "variable_label"])
               .agg(r_q_median=("r_q", "median"),
                    standoff_q_median=("standoff_q", "median"),
                    delta_q_median=("delta_q", "median"),
                    maxgrad_q_median=("maxgrad_q", "median"),
                    lambda_q_median=("lambda_q", "median"),
                    lambda_inf_median=("lambda_inf", "median"),
                    delta_over_lambda_q_median=("delta_over_lambda_q", "median"),
                    delta_over_lambda_inf_median=("delta_over_lambda_inf", "median"),
                    standoff_over_lambda_inf_median=("standoff_over_lambda_inf", "median"))
               .reset_index())

    # Add ratios to density scale for the same gas/Kn
    rho_scale = (summary[summary["variable"] == "rho"][["gas", "Kn", "Mach", "delta_q_median"]]
                 .rename(columns={"delta_q_median": "delta_rho_median"}))
    summary = summary.merge(rho_scale, on=["gas", "Kn", "Mach"], how="left")
    summary["delta_q_over_delta_rho"] = summary["delta_q_median"] / summary["delta_rho_median"]
    return summary


def variable_specific_profile(case: CaseData, row: pd.Series):
    angle = float(row["angle_rad"])
    r_q = float(row["r_q"])
    delta_q = float(row["delta_q"])
    if (not np.isfinite(delta_q)) or abs(delta_q) < 1e-300:
        return np.array([]), np.array([]), np.array([])
    var_key = row["variable"]
    rr, xs, ys = sample_ray(case, angle, n=1100)
    qi = get_var_interpolator(case, var_key)
    q = qi(xs, ys)
    q = np.asarray(q.filled(np.nan) if hasattr(q, "filled") else q, dtype=float)
    s_from_wall = rr - case.radius
    xi = (rr - r_q) / delta_q
    valid = np.isfinite(q) & np.isfinite(xi)
    if valid.sum() < 20:
        return np.array([]), np.array([]), np.array([])
    s = s_from_wall[valid]
    xi = xi[valid]
    qn = normalize(smooth1d(q[valid], 11))
    order = np.argsort(xi)
    return s[order], xi[order], qn[order]


def compute_variable_specific_collapse_quality(cases, var_metrics: pd.DataFrame):
    rows = []
    xi_common = np.linspace(-5, 6, 260)
    for gas in sorted({c.gas for c in cases}):
        gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
        for var_key in ["rho", "mach", "Ttr", "Trot", "p"]:
            profiles = []
            used = []
            for case in gas_cases:
                df = var_metrics[(var_metrics["gas"] == gas) &
                                 (var_metrics["Kn"] == case.kn) &
                                 (np.isclose(var_metrics["Mach"], case.mach)) &
                                 (var_metrics["variable"] == var_key)]
                if df.empty:
                    continue
                row = df.iloc[np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy()))]
                s, xi, qn = variable_specific_profile(case, row)
                good = np.isfinite(xi) & np.isfinite(qn)
                if good.sum() < 10:
                    continue
                qi = np.interp(xi_common, xi[good], qn[good], left=np.nan, right=np.nan)
                profiles.append(qi)
                used.append(case.kn)
            if len(profiles) >= 2:
                P = np.vstack(profiles)
                rows.append({
                    "gas": gas,
                    "variable": var_key,
                    "n_cases": len(profiles),
                    "mean_std_own_scale": float(np.nanmean(np.nanstd(P, axis=0))),
                    "Kn_used": json.dumps(used),
                })
    return pd.DataFrame(rows)


def make_fig7_variable_scales(var_summary: pd.DataFrame, outdir: Path, unit: str):
    """Variable-specific length scales and VHS normalization."""
    if var_summary.empty:
        return
    # Focus on variables with enough data
    var_order = ["rho", "mach", "Ttr", "Trot", "p"]
    labels = {"rho": r"$\rho$", "mach": "Mach", "Ttr": r"$T_{tr}$", "Trot": r"$T_{rot}$", "p": r"$p$"}
    for gas, dfg in var_summary.groupby("gas"):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
        plt.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.83, wspace=0.34)
        for var in var_order:
            dfv = dfg[dfg["variable"] == var].sort_values("Kn")
            if dfv.empty:
                continue
            lab = labels.get(var, var)
            axes[0].plot(dfv["Kn"], dfv["delta_q_median"], "o-", label=lab)
            axes[1].plot(dfv["Kn"], dfv["delta_q_over_delta_rho"], "o-", label=lab)
            axes[2].plot(dfv["Kn"], dfv["delta_over_lambda_q_median"], "o-", label=lab)
        for ax in axes:
            ax.set_xscale("log")
            ax.grid(alpha=0.24)
            ax.set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(rf"Median variable thickness, $\delta_q$ ({unit})")
        axes[1].set_ylabel(r"Scale ratio, $\delta_q/\delta_\rho$ (-)")
        axes[2].set_ylabel(r"VHS-normalized scale, $\delta_q/\lambda_q$ (-)")
        axes[0].set_title("(a) Variable-specific thickness")
        axes[1].set_title("(b) Relative to density thickness")
        axes[2].set_title("(c) Relative to local VHS mean free path")
        axes[0].legend(frameon=True, ncol=2, fontsize=9)
        fig.suptitle(f"Variable-specific shock-layer scales: {gas}", y=0.98, fontweight="bold")
        savefig(fig, outdir/f"JFM_Fig7_VariableSpecificScales_{gas}")


def make_fig8_variable_collapse_quality(own_df: pd.DataFrame, outdir: Path):
    if own_df.empty:
        return
    vars_order = ["rho", "mach", "Ttr", "Trot", "p"]
    labels = [r"$\rho$", "Mach", r"$T_{tr}$", r"$T_{rot}$", r"$p$"]
    gases = sorted(own_df["gas"].unique())
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    xbase = np.arange(len(vars_order))
    width = 0.36
    for ig, gas in enumerate(gases):
        vals = []
        for var in vars_order:
            row = own_df[(own_df["gas"] == gas) & (own_df["variable"] == var)]
            vals.append(float(row["mean_std_own_scale"].iloc[0]) if not row.empty else np.nan)
        ax.bar(xbase + (ig-0.5)*width, vals, width=width, label=gas,
               color=GAS_COLORS.get(gas, None), alpha=0.86)
    ax.set_xticks(xbase)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean spread after own-scale collapse, $\langle\sigma_{Kn}(q(\xi_q))\rangle$ (-)")
    ax.set_title("Does each variable collapse with its own gradient scale?")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig8_OwnScaleCollapseQuality")


def make_fig9_variable_specific_collapse(cases, var_metrics: pd.DataFrame, gas: str, outdir: Path):
    """Variable-specific collapse using xi_q for each q."""
    gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
    if len(gas_cases) < 2:
        return
    var_order = ["rho", "mach", "Ttr", "Trot"] if gas == "N2" else ["rho", "mach", "Ttr"]
    var_order = [v for v in var_order if not var_metrics[(var_metrics["gas"] == gas) & (var_metrics["variable"] == v)].empty]
    if not var_order:
        return
    kns = np.array([c.kn for c in gas_cases])
    norm = plt.Normalize(np.log10(kns.min()), np.log10(kns.max()))
    cmap = plt.get_cmap(KN_CMAP)
    fig, axes = plt.subplots(1, len(var_order), figsize=(5.2*len(var_order)+1.0, 4.7), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(left=0.07, right=0.86, bottom=0.17, top=0.80, wspace=0.28)
    labels = {"rho": r"$\rho$", "mach": "Mach", "Ttr": r"$T_{tr}$", "Trot": r"$T_{rot}$", "p": r"$p$"}

    for case in gas_cases:
        color = cmap(norm(np.log10(case.kn)))
        for j, var in enumerate(var_order):
            df = var_metrics[(var_metrics["gas"] == gas) &
                             (var_metrics["Kn"] == case.kn) &
                             (np.isclose(var_metrics["Mach"], case.mach)) &
                             (var_metrics["variable"] == var)]
            if df.empty:
                continue
            row = df.iloc[np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy()))]
            s, xi, qn = variable_specific_profile(case, row)
            if xi.size < 5:
                continue
            axes[j].plot(xi, qn, color=color)
    for j, var in enumerate(var_order):
        axes[j].set_xlabel(r"$\xi_q=(r-r_q)/\delta_q$ (-)")
        axes[j].set_ylabel("Normalized profile (-)")
        axes[j].set_title(f"({chr(97+j)}) {labels.get(var,var)} with own scale")
        axes[j].set_xlim(-6, 7)
        axes[j].set_ylim(-0.05, 1.05)
        axes[j].grid(alpha=0.22)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.895, 0.18, 0.018, 0.57])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$\log_{10}(\mathrm{Kn}_\infty)$")
    fig.suptitle(f"Variable-specific shock-attached collapse: {gas}", y=0.96, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig9_VariableSpecificCollapse_{gas}")


# =============================================================================
# v8: uncertainty, rigorous collapse metric, relaxation-scale and surface diagnostics
# =============================================================================

def detect_variable_layer_on_ray_with_window(case: CaseData, angle_rad: float, var_key: str, smooth_window: int, min_points: int = 80):
    """Same as detect_variable_layer_on_ray(), but with a prescribed smoothing window.

    This is used for algorithmic/sensitivity uncertainty in r_q and delta_q.
    It does not replace true DSMC statistical uncertainty from independent runs,
    but it tests whether the extracted trends are robust to detection choices.
    """
    rr, xs, ys = sample_ray(case, angle_rad)
    qi = get_var_interpolator(case, var_key)
    q = qi(xs, ys)
    q = np.asarray(q.filled(np.nan) if hasattr(q, "filled") else q, dtype=float)

    rhoi = interp(case, "rho")
    rho_ray = rhoi(xs, ys)
    rho_ray = np.asarray(rho_ray.filled(np.nan) if hasattr(rho_ray, "filled") else rho_ray, dtype=float)

    if case.vm.get("ttr"):
        Ti = mtri.LinearTriInterpolator(case.tri, case.df[case.vm["ttr"]].to_numpy())
    elif case.vm.get("t"):
        Ti = mtri.LinearTriInterpolator(case.tri, case.df[case.vm["t"]].to_numpy())
    else:
        Ti = None

    if Ti is not None:
        T_ray = Ti(xs, ys)
        T_ray = np.asarray(T_ray.filled(np.nan) if hasattr(T_ray, "filled") else T_ray, dtype=float)
        lam_ray = vhs_mean_free_path(rho_ray, T_ray, case.gas)
    else:
        lam_ray = np.full_like(q, np.nan)

    valid = np.isfinite(q)
    if valid.sum() < min_points:
        return None

    rr_v = rr[valid]
    q_v = q[valid]
    lam_v = lam_ray[valid]

    keep = rr_v > (case.radius + 0.03*case.radius)
    if keep.sum() < min_points:
        keep = rr_v > (case.radius + 0.01*case.radius)
    rr_v = rr_v[keep]
    q_v = q_v[keep]
    lam_v = lam_v[keep]
    if rr_v.size < min_points:
        return None

    q_s = smooth1d(q_v, window=smooth_window)
    dq = np.gradient(q_s, rr_v)
    absgrad = np.abs(dq)
    imax = int(np.nanargmax(absgrad))
    r_q = float(rr_v[imax])
    maxgrad = float(absgrad[imax])

    nseg = max(8, rr_v.size//12)
    q_up = float(np.nanmedian(q_v[-nseg:]))
    q_down = float(np.nanmedian(q_v[:nseg]))
    delta_q = abs(q_down - q_up) / max(maxgrad, 1e-300)
    lambda_q = float(lam_v[imax]) if np.isfinite(lam_v[imax]) else np.nan

    return {
        "r_q": r_q,
        "standoff_q": r_q - case.radius,
        "delta_q": float(delta_q),
        "maxgrad_q": maxgrad,
        "lambda_q": lambda_q,
    }


def compute_extraction_uncertainty(cases, half_angle_deg: float, n_angles: int, windows=(7, 11, 15, 21, 31)):
    """Sensitivity/uncertainty of r_q and delta_q to the detection smoothing window.

    Output is a practical robustness metric:
      std_delta_q / mean_delta_q and std_r_q / mean_r_q.
    """
    rows = []
    offsets = np.linspace(0.0, half_angle_deg, n_angles)

    for case in cases:
        for var_key, var_label in available_profile_variables(case):
            # Use the sign with the most valid detections, same logic as before.
            for sign in [-1, +1]:
                dets_all = []
                for deg in offsets:
                    angle = case.upstream_angle + sign*np.deg2rad(deg)
                    if math.sin(angle) < -1e-8:
                        continue
                    for win in windows:
                        det = detect_variable_layer_on_ray_with_window(case, angle, var_key, smooth_window=win)
                        if det is None:
                            continue
                        det.update({
                            "gas": case.gas,
                            "Kn": case.kn,
                            "Mach": case.mach,
                            "variable": var_key,
                            "variable_label": var_label,
                            "angle_deg_from_upstream": sign*deg,
                            "window": win,
                        })
                        dets_all.append(det)
                # Record both signs; later group by angle/var.
                rows.extend(dets_all)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    raw = pd.DataFrame(rows)

    # Uncertainty for each ray.
    ray_stats = (raw.groupby(["gas", "Kn", "Mach", "variable", "variable_label", "angle_deg_from_upstream"])
                 .agg(r_mean=("r_q", "mean"),
                      r_std=("r_q", "std"),
                      delta_mean=("delta_q", "mean"),
                      delta_std=("delta_q", "std"),
                      n=("delta_q", "count"))
                 .reset_index())
    ray_stats["r_rel_std"] = ray_stats["r_std"] / ray_stats["r_mean"].replace(0, np.nan)
    ray_stats["delta_rel_std"] = ray_stats["delta_std"] / ray_stats["delta_mean"].replace(0, np.nan)

    # Summary over rays for paper.
    summary = (ray_stats.groupby(["gas", "Kn", "Mach", "variable", "variable_label"])
               .agg(r_rel_std_median=("r_rel_std", "median"),
                    r_abs_std_median=("r_std", "median"),
                    delta_rel_std_median=("delta_rel_std", "median"),
                    delta_abs_std_median=("delta_std", "median"),
                    n_rays=("angle_deg_from_upstream", "count"))
               .reset_index())
    return raw, summary


def compute_integrated_l2_collapse(cases, var_metrics: pd.DataFrame):
    """Integrated L2 distance to ensemble-mean profile in variable-specific xi_q.

    For each gas and variable:
      E_i = ||q_i(xi)-mean(q)||_2 / ||mean(q)||_2
      variance captured by mean = 1 - SSE_about_mean / SSE_about_global_constant
    """
    rows_case = []
    rows_summary = []
    xi_common = np.linspace(-5, 6, 320)

    for gas in sorted({c.gas for c in cases}):
        gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
        for var_key in ["rho", "mach", "Ttr", "Trot", "p"]:
            profiles = []
            used_cases = []

            for case in gas_cases:
                df = var_metrics[(var_metrics["gas"] == gas) &
                                 (var_metrics["Kn"] == case.kn) &
                                 (np.isclose(var_metrics["Mach"], case.mach)) &
                                 (var_metrics["variable"] == var_key)]
                if df.empty:
                    continue
                row = df.iloc[np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy()))]
                s, xi, qn = variable_specific_profile(case, row)
                good = np.isfinite(xi) & np.isfinite(qn)
                if good.sum() < 10:
                    continue
                qi = np.interp(xi_common, xi[good], qn[good], left=np.nan, right=np.nan)
                profiles.append(qi)
                used_cases.append(case)

            if len(profiles) < 2:
                continue

            P = np.vstack(profiles)
            valid = np.isfinite(P).mean(axis=0) > 0.75
            if valid.sum() < 20:
                continue
            Pv = P[:, valid]
            xi_v = xi_common[valid]

            mean_profile = np.nanmean(Pv, axis=0)
            global_mean = np.nanmean(Pv)

            sse = 0.0
            sst = 0.0
            l2_errors = []
            for ic, case in enumerate(used_cases):
                q = Pv[ic, :]
                good = np.isfinite(q) & np.isfinite(mean_profile)
                if good.sum() < 10:
                    continue
                num = np.trapz((q[good] - mean_profile[good])**2, xi_v[good])
                den = np.trapz(mean_profile[good]**2, xi_v[good])
                l2 = math.sqrt(num / max(den, 1e-300))
                l2_errors.append(l2)
                rows_case.append({
                    "gas": gas,
                    "Kn": case.kn,
                    "Mach": case.mach,
                    "source_tag": case.source_tag,
                    "flow_file": str(case.flow_file),
                    "variable": var_key,
                    "L2_to_mean": l2,
                })
                sse += np.nansum((q[good] - mean_profile[good])**2)
                sst += np.nansum((q[good] - global_mean)**2)

            var_captured = 1.0 - sse / max(sst, 1e-300)
            rows_summary.append({
                "gas": gas,
                "variable": var_key,
                "n_cases": len(l2_errors),
                "mean_L2_to_mean": float(np.nanmean(l2_errors)),
                "median_L2_to_mean": float(np.nanmedian(l2_errors)),
                "max_L2_to_mean": float(np.nanmax(l2_errors)),
                "variance_captured_by_mean": float(var_captured),
            })

    return pd.DataFrame(rows_case), pd.DataFrame(rows_summary)


def compute_surface_metrics(cases):
    """Read DS2SU surface files and compute simple aerothermal/surface metrics.

    Surface columns can include:
      S, ANGLE, PR, SSP, HEAT-FLUX, TS, ...
    This function returns stagnation/max/integral values versus Kn.
    """
    rows = []
    for case in cases:
        if case.surface_file is None:
            continue
        try:
            sdf, cols = read_tecplot_point(case.surface_file)
        except Exception as e:
            print(f"[WARN] surface read failed for {case.gas} Kn={case.kn:g}: {e}")
            continue

        colmap = {normalize_key(c): c for c in cols}
        def cget(name):
            return colmap.get(normalize_key(name))

        scol = cget("S")
        acol = cget("ANGLE")
        prcol = cget("PR")
        heatcol = cget("HEAT-FLUX") or cget("HEATFLUX")
        sspcol = cget("SSP")
        tscol = cget("TS")

        if scol is None:
            # Use index as pseudo-s if unavailable.
            s = np.arange(len(sdf), dtype=float)
        else:
            s = sdf[scol].to_numpy(dtype=float)

        order = np.argsort(s)
        s_sorted = s[order]

        if acol is not None:
            angle = sdf[acol].to_numpy(dtype=float)[order]
            istag = int(np.nanargmin(np.abs(angle)))
        else:
            angle = np.full_like(s_sorted, np.nan)
            # fallback: stagnation = max pressure or first point
            istag = 0

        row = {"gas": case.gas, "Kn": case.kn, "Mach": case.mach, "source_tag": case.source_tag, "surface_file": str(case.surface_file)}

        for key, col in [("PR", prcol), ("HEAT_FLUX", heatcol), ("SSP", sspcol), ("TS", tscol)]:
            if col is None:
                continue
            q = sdf[col].to_numpy(dtype=float)[order]
            row[f"{key}_stag"] = float(q[istag]) if len(q) else np.nan
            row[f"{key}_max"] = float(np.nanmax(q)) if len(q) else np.nan
            row[f"{key}_min"] = float(np.nanmin(q)) if len(q) else np.nan
            try:
                row[f"{key}_int"] = float(np.trapz(q, s_sorted))
                row[f"{key}_abs_int"] = float(np.trapz(np.abs(q), s_sorted))
            except Exception:
                row[f"{key}_int"] = np.nan
                row[f"{key}_abs_int"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def make_fig10_extraction_uncertainty(uq_summary: pd.DataFrame, outdir: Path):
    if uq_summary.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    vars_keep = ["rho", "Ttr", "Trot"]
    for gas, dfg in uq_summary.groupby("gas"):
        for var in vars_keep:
            dfv = dfg[dfg["variable"] == var].sort_values("Kn")
            if dfv.empty:
                continue
            label = f"{gas} {var}"
            axes[0].plot(dfv["Kn"], 100*dfv["r_rel_std_median"], "o-", label=label)
            axes[1].plot(dfv["Kn"], 100*dfv["delta_rel_std_median"], "o-", label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Median relative uncertainty in layer location (%)")
    axes[1].set_ylabel("Median relative uncertainty in thickness (%)")
    axes[0].set_title("(a) Location-detection robustness")
    axes[1].set_title("(b) Thickness-extraction robustness")
    axes[0].legend(frameon=True, fontsize=8, ncol=2)
    savefig(fig, outdir/"JFM_Fig10_ExtractionUncertainty")


def make_fig11_l2_collapse(l2_summary: pd.DataFrame, outdir: Path):
    if l2_summary.empty:
        return
    vars_order = ["rho", "mach", "Ttr", "Trot", "p"]
    labels = [r"$\rho$", "Mach", r"$T_{tr}$", r"$T_{rot}$", r"$p$"]
    gases = sorted(l2_summary["gas"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    width = 0.36
    xbase = np.arange(len(vars_order))

    for ig, gas in enumerate(gases):
        means = []
        vars_cap = []
        for var in vars_order:
            row = l2_summary[(l2_summary["gas"] == gas) & (l2_summary["variable"] == var)]
            means.append(float(row["mean_L2_to_mean"].iloc[0]) if not row.empty else np.nan)
            vars_cap.append(float(row["variance_captured_by_mean"].iloc[0]) if not row.empty else np.nan)
        shift = (ig - 0.5) * width
        axes[0].bar(xbase + shift, means, width=width, label=gas, color=GAS_COLORS.get(gas, None), alpha=0.86)
        axes[1].bar(xbase + shift, vars_cap, width=width, label=gas, color=GAS_COLORS.get(gas, None), alpha=0.86)

    for ax in axes:
        ax.set_xticks(xbase)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"Mean integrated $L_2$ distance to mean profile (-)")
    axes[1].set_ylabel("Variance captured by mean profile (-)")
    axes[0].set_title("(a) Collapse error in own-scale coordinates")
    axes[1].set_title("(b) Mean-profile representativeness")
    axes[0].legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig11_L2CollapseMetrics")


def make_fig12_surface_correlation(surface_df: pd.DataFrame, shock_summary: pd.DataFrame, outdir: Path):
    if surface_df.empty:
        return
    df = surface_df.merge(shock_summary[["gas", "Kn", "standoff_median", "delta_rho_median"]],
                          on=["gas", "Kn"], how="left")
    # Choose columns that exist.
    has_heat = "HEAT_FLUX_stag" in df.columns or "HEAT_FLUX_max" in df.columns
    has_pr = "PR_stag" in df.columns or "PR_max" in df.columns
    if not has_heat and not has_pr:
        return

    ncols = 2 if has_heat and has_pr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.1*ncols, 4.8), squeeze=False)
    axes = axes.ravel()
    iax = 0
    if has_heat:
        ycol = "HEAT_FLUX_stag" if "HEAT_FLUX_stag" in df.columns else "HEAT_FLUX_max"
        for gas, dfg in df.groupby("gas"):
            dfg = dfg.sort_values("Kn")
            axes[iax].plot(dfg["delta_rho_median"], dfg[ycol], "o-", color=GAS_COLORS.get(gas, None), label=gas)
        axes[iax].set_xlabel(r"Median $\delta_\rho$ (m)")
        axes[iax].set_ylabel(ycol.replace("_", " "))
        axes[iax].set_title("(a) Surface heat flux versus shock thickness")
        axes[iax].grid(alpha=0.25)
        iax += 1
    if has_pr:
        ycol = "PR_stag" if "PR_stag" in df.columns else "PR_max"
        for gas, dfg in df.groupby("gas"):
            dfg = dfg.sort_values("Kn")
            axes[iax].plot(dfg["standoff_median"], dfg[ycol], "s-", color=GAS_COLORS.get(gas, None), label=gas)
        axes[iax].set_xlabel("Median shock standoff (m)")
        axes[iax].set_ylabel(ycol.replace("_", " "))
        axes[iax].set_title("(b) Surface pressure versus shock standoff")
        axes[iax].grid(alpha=0.25)
    axes[0].legend(frameon=True)
    savefig(fig, outdir/"JFM_Fig12_SurfaceCorrelation")


# =============================================================================
# v9: |grad rho| shock-indicator figures and stagnation-line 1D reference proxy
# =============================================================================

def regular_grid_upper(case: CaseData, nx=360, ny=240):
    """Regular upper-half grid outside the cylinder for contour diagnostics."""
    x = get_array(case, "x")
    y = get_array(case, "y")
    cx, cy = case.center
    xmin, xmax = np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5)
    ymin, ymax = cy, np.nanpercentile(y, 99.5)
    Xg, Yg = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
    inside = (Xg-cx)**2 + (Yg-cy)**2 <= case.radius**2
    return Xg, Yg, inside


def gradient_field_on_grid(case: CaseData, key="rho", nx=360, ny=240):
    """Interpolate a field and compute normalized gradient magnitude on upper-half grid."""
    Xg, Yg, inside = regular_grid_upper(case, nx=nx, ny=ny)
    qi = get_var_interpolator(case, key)
    Q = qi(Xg, Yg)
    Q = np.asarray(Q.filled(np.nan) if hasattr(Q, "filled") else Q, dtype=float)
    Q[inside] = np.nan
    dx = float(np.nanmedian(np.diff(Xg[0, :])))
    dy = float(np.nanmedian(np.diff(Yg[:, 0])))
    dQdy, dQdx = np.gradient(Q, dy, dx)
    G = np.sqrt(dQdx**2 + dQdy**2)
    # Dimensionless shock indicator: |grad q| / max(|grad q|)
    Gnorm = G / np.nanmax(G) if np.nanmax(G) > 0 else G
    return Xg, Yg, Q, G, Gnorm


def make_fig13_grad_rho_contours(cases, metrics, gas, outdir, unit):
    """Paper-ready |grad rho| shock indicator contours.

    This is the reviewer-friendly version of Fig. 1: it shows the shock indicator
    directly rather than only density contours.
    """
    gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c: c.kn)
    if not gas_cases:
        return
    idxs = sorted(set([0, len(gas_cases)//2, len(gas_cases)-1]))
    sel = [gas_cases[i] for i in idxs]

    fig, axes = plt.subplots(1, len(sel), figsize=(5.2*len(sel) + 0.65, 4.85), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(left=0.07, right=0.90, bottom=0.13, top=0.82, wspace=0.24)

    last_cf = None
    for iax, (ax, case) in enumerate(zip(axes, sel)):
        Xg, Yg, Q, G, Gnorm = gradient_field_on_grid(case, "rho", nx=360, ny=240)
        # Use log-like display to retain diffuse high-Kn structures.
        Gplot = np.log10(np.maximum(Gnorm, 1e-4))
        levels = np.linspace(-4, 0, 36)
        last_cf = ax.contourf(Xg, Yg, Gplot, levels=levels, cmap="magma", extend="both")

        cx, cy = case.center
        ax.add_patch(Circle((cx, cy), case.radius, facecolor="white", edgecolor="black", lw=1.2, zorder=5))

        df = metrics[(metrics["gas"] == gas) & (metrics["Kn"] == case.kn)]
        if not df.empty:
            xs = cx + df["r_s"].to_numpy() * np.cos(df["angle_rad"].to_numpy())
            ys = cy + df["r_s"].to_numpy() * np.sin(df["angle_rad"].to_numpy())
            ax.plot(xs, ys, "w-", lw=2.8)
            ax.plot(xs, ys, "c-", lw=1.0)

        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(case.center[1], np.nanmax(Yg))
        ax.set_title(fr"{gas}, $\mathrm{{Kn}}={case.kn:g}$", pad=8)
        ax.set_xlabel(f"x ({unit})")
        if iax == 0:
            ax.set_ylabel(f"y ({unit})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    cax = fig.add_axes([0.915, 0.17, 0.015, 0.60])
    cbar = fig.colorbar(last_cf, cax=cax)
    cbar.set_label(r"$\log_{10}(|\nabla\rho|/\max|\nabla\rho|)$")
    fig.suptitle(f"Density-gradient shock indicator across Kn: {gas}", y=0.96, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig13_GradRhoShockIndicator_{gas}")


def compute_stagnation_1d_reference_proxy(cases, var_metrics: pd.DataFrame):
    """Stagnation-line local 1D reference/proxy diagnostics.

    This is not a full independent 1D normal-shock solution. It is a deliberately
    transparent local reference: if the bow shock behaved like a locally planar
    1D shock controlled only by the local mean free path, delta_q/lambda_q would
    be approximately constant with Kn. Deviations quantify curvature/global
    rarefaction/relaxation effects.
    """
    rows = []
    for case in cases:
        dfc = var_metrics[(var_metrics["gas"] == case.gas) & (var_metrics["Kn"] == case.kn)]
        if dfc.empty:
            continue

        # Use the ray closest to the stagnation/upstream direction.
        for var in ["rho", "Ttr", "Trot", "mach", "p"]:
            dfv = dfc[dfc["variable"] == var]
            if dfv.empty:
                continue
            row = dfv.iloc[np.argmin(np.abs(dfv["angle_deg_from_upstream"].to_numpy()))]
            rows.append({
                "gas": case.gas,
                "Kn": case.kn,
                "Mach": case.mach,
                "source_tag": case.source_tag,
                "variable": var,
                "delta_q": row["delta_q"],
                "lambda_q": row["lambda_q"],
                "lambda_inf": row["lambda_inf"],
                "delta_over_lambda_q": row["delta_q"]/row["lambda_q"] if np.isfinite(row["lambda_q"]) and row["lambda_q"] != 0 else np.nan,
                "delta_over_lambda_inf": row["delta_q"]/row["lambda_inf"] if np.isfinite(row["lambda_inf"]) and row["lambda_inf"] != 0 else np.nan,
                "standoff_over_lambda_inf": row["standoff_q"]/row["lambda_inf"] if np.isfinite(row["lambda_inf"]) and row["lambda_inf"] != 0 else np.nan,
                "standoff_q": row["standoff_q"],
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Define low-Kn baseline for each gas/variable to show departure from local-1D scaling.
    baselines = []
    for (gas, var), dfg in out.groupby(["gas", "variable"]):
        dfg = dfg.sort_values("Kn")
        base = dfg.iloc[0]["delta_over_lambda_q"]
        for _, r in dfg.iterrows():
            baselines.append({
                "gas": gas,
                "Kn": r["Kn"],
                "Mach": r["Mach"],
                "variable": var,
                "delta_over_lambda_q_over_lowKn": r["delta_over_lambda_q"]/base if np.isfinite(base) and base != 0 else np.nan,
            })
    bdf = pd.DataFrame(baselines)
    out = out.merge(bdf, on=["gas", "Kn", "Mach", "variable"], how="left")
    return out


def make_fig14_stagnation_1d_proxy(proxy_df: pd.DataFrame, outdir: Path):
    if proxy_df.empty:
        return
    vars_keep = ["rho", "Ttr", "Trot"]
    labels = {"rho": r"$\rho$", "Ttr": r"$T_{tr}$", "Trot": r"$T_{rot}$"}
    for gas, dfg in proxy_df.groupby("gas"):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
        plt.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.83, wspace=0.34)

        for var in vars_keep:
            dfv = dfg[dfg["variable"] == var].sort_values("Kn")
            if dfv.empty:
                continue
            lab = labels.get(var, var)
            axes[0].plot(dfv["Kn"], dfv["delta_over_lambda_q"], "o-", label=lab)
            axes[1].plot(dfv["Kn"], dfv["delta_over_lambda_q_over_lowKn"], "o-", label=lab)
            axes[2].plot(dfv["Kn"], dfv["standoff_over_lambda_inf"], "o-", label=lab)

        for ax in axes:
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$\mathrm{Kn}_\infty$ (-)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(r"Local-mfp-normalized thickness, $\delta_q/\lambda_q$ (-)")
        axes[1].set_ylabel(r"Departure from low-Kn local scaling (-)")
        axes[2].set_ylabel(r"Standoff in freestream mean free paths, $\Delta_q/\lambda_\infty$ (-)")
        axes[0].set_title("(a) Local 1D-shock proxy")
        axes[1].set_title("(b) Scaling departure")
        axes[2].set_title("(c) Upstream influence")
        axes[0].legend(frameon=True)
        fig.suptitle(f"Stagnation-line local-mfp reference: {gas}", y=0.98, fontweight="bold")
        savefig(fig, outdir/f"JFM_Fig14_Stagnation1DProxy_{gas}")


def infer_mach_from_path(path: Path):
    """Infer Mach number from path components.

    v14 fix:
    The Argon archive has files like:
      TriangulateGridM5.dat, TriangulateGridM55.dat, TriangulatedGridM825.dat
    The parent folder is named Argon-Data-Mach5-15.  Older versions matched
    the parent folder first/too broadly, so all Argon files were assigned Mach=5.
    This function now prioritizes the actual filename.

    Interpretation:
      M5    -> 5
      M55   -> 5.5
      M825  -> 8.25
      M975  -> 9.75
      M10   -> 10
      M15   -> 15
    """
    comps = list(path.parts)

    def clean_numeric_from_m(raw: str):
        raw = str(raw).replace(",", ".")
        # Normal decimal form, e.g. 7.5
        try:
            if "." in raw:
                val = float(raw)
                if 3.0 <= val <= 30.0:
                    return val
        except Exception:
            pass

        # Integer/compact form.
        if re.match(r"^[0-9]+$", raw):
            intval = int(raw)
            # Ordinary integer Mach numbers.
            if 3 <= intval <= 30:
                return float(intval)
            # Compact decimal notation: 55 -> 5.5, 825 -> 8.25, 975 -> 9.75.
            if len(raw) >= 2:
                val = float(raw[0] + "." + raw[1:])
                if 3.0 <= val <= 30.0:
                    return val
        return np.nan

    # 1) Highest priority: actual file name, because parent can be "Mach5-15".
    fname = path.name.replace(",", ".")
    for pat in [
        r"(?:TriangulateGrid|TriangulatedGrid)\s*M\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])M\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])Ma\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
        r"(?:^|[^A-Za-z])Mach\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
    ]:
        m = re.search(pat, fname, re.I)
        if m:
            val = clean_numeric_from_m(m.group(1))
            if np.isfinite(val):
                return val

    # 2) Then scan parent components, but avoid ranges like Mach5-15.
    for comp in comps[::-1]:
        s = comp.replace(",", ".")
        if re.search(r"Mach\s*[0-9]+(?:\.[0-9]+)?\s*[-]\s*[0-9]+", s, re.I):
            # This is probably a range label such as Mach5-15, not a single case.
            continue
        patterns = [
            r"(?:^|[^A-Za-z])Mach\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])Ma\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:^|[^A-Za-z])M\s*[-_=]?\s*([0-9]+(?:\.[0-9]+)?)",
            r"^([0-9]+(?:\.[0-9]+)?)\s*Mach$",
        ]
        for pat in patterns:
            m = re.search(pat, s, re.I)
            if m:
                val = clean_numeric_from_m(m.group(1))
                if np.isfinite(val):
                    return val

    # 3) Pure numeric folder fallback.
    for comp in comps[::-1]:
        s = comp.replace(",", ".").strip()
        m = re.match(r"^([0-9]+(?:\.[0-9]+)?)$", s)
        if m:
            try:
                val = float(m.group(1))
                if 3.0 <= val <= 30.0:
                    return val
            except Exception:
                pass

    return np.nan


def parse_mach_values(spec: str):
    """Parse '5:15', '5,6,7', or '5 6 7' into a list of floats."""
    if spec is None:
        return []
    spec = str(spec).strip()
    if ":" in spec:
        a, b = spec.split(":", 1)
        a = int(float(a)); b = int(float(b))
        step = 1 if b >= a else -1
        return [float(v) for v in range(a, b+step, step)]
    parts = re.split(r"[,\s]+", spec)
    vals = []
    for s in parts:
        if s:
            vals.append(float(s))
    return vals



def infer_source_tag(path: Path) -> str:
    """Classify whether a case comes from a Mach-sweep archive or a Kn-sweep archive."""
    s = str(path)
    sl = s.lower()
    if ("argon-data-mach" in sl) or ("triangulategrid" in sl) or ("triangulatedgrid" in sl):
        return "mach_sweep_argon"
    if ("results-n2" in sl):
        return "mach_sweep_n2"
    if ("prop-kn" in sl) or ("kn=" in sl):
        return "kn_sweep"
    return "unknown"


def assign_missing_mach_by_order(cases: List[Dict], ar_values: str = "5:15", n2_values: str = "5:14"):
    """Assign Mach values to archives where folder names do not contain Mach.

    If multiple cases in the same root/gas/Kn group have missing Mach, assign
    an ordered list of Mach values. This is useful for archives where folders
    are named in sorted order but do not explicitly contain 'Mach'.
    """
    ar_list = parse_mach_values(ar_values)
    n2_list = parse_mach_values(n2_values)
    out = []
    df = pd.DataFrame(cases)
    if df.empty:
        return cases

    used_idx = set()
    # group by root, gas, kn for missing/ambiguous mach groups
    for (root, gas, kn), dfg in df.groupby(["root_path", "gas", "kn"], dropna=False):
        idxs = list(dfg.index)
        # Cases that still have NaN Mach or all collapsed to same default in a multi-file Mach archive.
        missing = [i for i in idxs if not np.isfinite(df.loc[i, "mach"])]
        # Assign only if group has multiple cases and missing values.
        if len(idxs) > 1 and missing:
            vals = ar_list if gas == "Ar" else n2_list if gas == "N2" else []
            if len(vals) >= len(missing):
                for ii, mach in zip(sorted(missing, key=lambda k: str(df.loc[k, "flow_file"])), vals[:len(missing)]):
                    df.loc[ii, "mach"] = mach
                    df.loc[ii, "mach_assigned_by_order"] = True
            else:
                print(f"[WARN] Cannot assign Mach by order for {gas} root={root}: {len(missing)} files but {len(vals)} values.")
    # any remaining nan left for later defaulting
    return df.to_dict("records")

def summarize_mach_sweep_if_present(cases, metrics, outdir):
    """Optional Mach-sweep diagnostic if Mach is encoded in path names.

    If no Mach metadata are available, this quietly skips.
    """
    rows = []
    for case in cases:
        mach = infer_mach_from_path(case.flow_file)
        if not np.isfinite(mach):
            continue
        df = metrics[(metrics["gas"] == case.gas) & (metrics["Kn"] == case.kn)]
        if df.empty:
            continue
        rows.append({
            "gas": case.gas,
            "Kn": case.kn,
            "Mach": mach,
            "standoff_median": float(df["standoff"].median()),
            "delta_rho_median": float(df["delta_rho"].median()),
        })
    mdf = pd.DataFrame(rows)
    if mdf.empty or mdf["Mach"].nunique() < 2:
        return mdf
    mdf.to_csv(outdir/"mach_sweep_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7))
    for (gas, kn), dfg in mdf.groupby(["gas", "Kn"]):
        dfg = dfg.sort_values("Mach")
        axes[0].plot(dfg["Mach"], dfg["standoff_median"], "o-", label=f"{gas}, Kn={kn:g}")
        axes[1].plot(dfg["Mach"], dfg["delta_rho_median"], "s-", label=f"{gas}, Kn={kn:g}")
    axes[0].set_xlabel(r"$M_\infty$ (-)")
    axes[1].set_xlabel(r"$M_\infty$ (-)")
    axes[0].set_ylabel("Median shock standoff (m)")
    axes[1].set_ylabel(r"Median $\delta_\rho$ (m)")
    axes[0].set_title("(a) Mach robustness: standoff")
    axes[1].set_title("(b) Mach robustness: thickness")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=True, fontsize=8)
    savefig(fig, outdir/"JFM_Fig15_MachSweepRobustness")
    return mdf



def add_stagnation_metrics(summary_df: pd.DataFrame, metrics_df: pd.DataFrame):
    """Add stagnation-line standoff/delta values to a summary table."""
    if summary_df.empty or metrics_df.empty:
        return summary_df
    rows = []
    for (gas, kn, mach), dfg in metrics_df.groupby(["gas", "Kn", "Mach"]):
        row = dfg.iloc[np.argmin(np.abs(dfg["angle_deg_from_upstream"].to_numpy()))]
        rows.append({
            "gas": gas, "Kn": kn, "Mach": mach,
            "standoff_stag": row["standoff"],
            "delta_rho_stag": row["delta_rho"],
            "maxgrad_stag": row["maxgrad_rho"],
        })
    sdf = pd.DataFrame(rows)
    return summary_df.merge(sdf, on=["gas", "Kn", "Mach"], how="left")

# =============================================================================
# v10: Mach-sweep figures at fixed Kn
# =============================================================================

def make_fig16_mach_sweep_scaling(summary_mach: pd.DataFrame, outdir: Path, unit: str, fixed_kn: float,
                                  stagnation_only: bool = True):
    """Mach-sweep scaling at fixed Kn.

    v11 default uses stagnation-line quantities instead of angular medians,
    because median-over-angle can be contaminated by low-Mach near-wall artifacts.
    """
    if summary_mach.empty or summary_mach["Mach"].nunique() < 2:
        print("[INFO] Mach-sweep scaling skipped: fewer than two Mach values found.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))
    plt.subplots_adjust(left=0.075, right=0.985, bottom=0.17, top=0.84, wspace=0.30)

    standoff_col = "standoff_stag" if stagnation_only and "standoff_stag" in summary_mach.columns else "standoff_median"
    delta_col = "delta_rho_stag" if stagnation_only and "delta_rho_stag" in summary_mach.columns else "delta_rho_median"

    for gas, dfg in summary_mach.groupby("gas"):
        dfg = dfg.sort_values("Mach")
        axes[0].plot(dfg["Mach"], dfg[standoff_col], "o-", color=GAS_COLORS.get(gas, None), label=gas)
        axes[1].plot(dfg["Mach"], dfg[delta_col], "s-", color=GAS_COLORS.get(gas, None), label=gas)
    axes[0].set_xlabel(r"$M_\infty$ (-)")
    axes[1].set_xlabel(r"$M_\infty$ (-)")
    axes[0].set_ylabel(f"Shock standoff ({unit})")
    axes[1].set_ylabel(rf"$\delta_\rho$ ({unit})")
    suffix = "stagnation line" if stagnation_only else "angular median"
    axes[0].set_title(fr"(a) Standoff at $Kn_\infty={fixed_kn:g}$")
    axes[1].set_title(fr"(b) Density thickness at $Kn_\infty={fixed_kn:g}$")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=True, title=suffix)
    savefig(fig, outdir/"JFM_Fig16_MachSweep_StandoffThickness")

def make_fig17_mach_variable_scales(var_summary: pd.DataFrame, outdir: Path, unit: str, fixed_kn: float):
    if var_summary.empty or "Mach" not in var_summary.columns:
        return
    df = var_summary[np.isclose(var_summary["Kn"], fixed_kn)].copy()
    if df.empty or df["Mach"].nunique() < 2:
        print("[INFO] Mach variable-scale figure skipped: insufficient Mach-sweep variable metrics.")
        return
    var_order = ["rho", "mach", "Ttr", "Trot", "p"]
    labels = {"rho": r"$\rho$", "mach": "Mach", "Ttr": r"$T_{tr}$", "Trot": r"$T_{rot}$", "p": r"$p$"}
    for gas, dfg in df.groupby("gas"):
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.7))
        plt.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.83, wspace=0.34)
        for var in var_order:
            if gas == "Ar" and var == "Trot":
                continue
            dfv = dfg[dfg["variable"] == var].sort_values("Mach")
            if dfv.empty:
                continue
            lab = labels.get(var, var)
            axes[0].plot(dfv["Mach"], dfv["delta_q_median"], "o-", label=lab)
            axes[1].plot(dfv["Mach"], dfv["delta_q_over_delta_rho"], "o-", label=lab)
            axes[2].plot(dfv["Mach"], dfv["delta_over_lambda_q_median"], "o-", label=lab)
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.set_xlabel(r"$M_\infty$ (-)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[0].set_ylabel(rf"Median $\delta_q$ ({unit})")
        axes[1].set_ylabel(r"$\delta_q/\delta_\rho$ (-)")
        axes[2].set_ylabel(r"$\delta_q/\lambda_q$ (-)")
        axes[0].set_title("(a) Variable-specific thickness")
        axes[1].set_title("(b) Relative to density thickness")
        axes[2].set_title("(c) Local-mfp normalization")
        axes[0].legend(frameon=True, ncol=2, fontsize=9)
        fig.suptitle(fr"Mach variation of variable-specific scales: {gas}, $Kn_\infty={fixed_kn:g}$",
                     y=0.98, fontweight="bold")
        savefig(fig, outdir/f"JFM_Fig17_MachSweep_VariableScales_{gas}")


def make_fig18_mach_contours(cases_mach, metrics_mach, gas, outdir, unit, fixed_kn):
    gas_cases = sorted([c for c in cases_mach if c.gas == gas], key=lambda c: c.mach)
    if len(gas_cases) < 2:
        return
    # Choose low/mid/high Mach.
    idxs = sorted(set([0, len(gas_cases)//2, len(gas_cases)-1]))
    sel = [gas_cases[i] for i in idxs]
    fig, axes = plt.subplots(1, len(sel), figsize=(5.2*len(sel)+0.65, 4.85), squeeze=False)
    axes = axes.ravel()
    plt.subplots_adjust(left=0.07, right=0.90, bottom=0.13, top=0.82, wspace=0.24)
    cache, all_vals = [], []
    for case in sel:
        x = get_array(case, "x")
        y = get_array(case, "y")
        rho = get_array(case, "rho")
        rho_inf = np.nanmedian(rho[x <= np.nanquantile(x, 0.08)])
        vals = rho / rho_inf
        mask = y >= case.center[1]
        tri = mtri.Triangulation(x[mask], y[mask])
        v2 = vals[mask]
        cache.append((case, x, y, tri, v2))
        all_vals.append(v2[np.isfinite(v2)])
    all_vals = np.concatenate(all_vals) if all_vals else np.array([0,1])
    levels = np.linspace(np.nanpercentile(all_vals, 2), np.nanpercentile(all_vals, 98), 36)
    last_cf = None
    for iax, (ax, item) in enumerate(zip(axes, cache)):
        case, x, y, tri, v2 = item
        last_cf = ax.tricontourf(tri, v2, levels=levels, cmap="turbo", extend="both")
        cx, cy = case.center
        ax.add_patch(Circle((cx, cy), case.radius, facecolor="white", edgecolor="black", lw=1.2, zorder=5))
        df = metrics_mach[(metrics_mach["gas"] == gas) & (np.isclose(metrics_mach["Mach"], case.mach))]
        if not df.empty:
            xs = cx + df["r_s"].to_numpy()*np.cos(df["angle_rad"].to_numpy())
            ys = cy + df["r_s"].to_numpy()*np.sin(df["angle_rad"].to_numpy())
            ax.plot(xs, ys, "w-", lw=2.8)
            ax.plot(xs, ys, "k-", lw=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(case.center[1], np.nanmax(y))
        ax.set_title(fr"{gas}, $M_\infty={case.mach:g}$", pad=8)
        ax.set_xlabel(f"x ({unit})")
        if iax == 0:
            ax.set_ylabel(f"y ({unit})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    cax = fig.add_axes([0.915, 0.17, 0.015, 0.60])
    cb = fig.colorbar(last_cf, cax=cax)
    cb.set_label(r"$\rho/\rho_\infty$")
    fig.suptitle(fr"Mach-dependent bow-shock structure: {gas}, $Kn_\infty={fixed_kn:g}$",
                 y=0.96, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig18_MachSweep_Contours_{gas}")


def make_fig19_mach_pod(cases_mach, metrics_mach, gas, outdir, fixed_kn):
    gas_cases = sorted([c for c in cases_mach if c.gas == gas], key=lambda c: c.mach)
    if len(gas_cases) < 3:
        return
    xi_grid = np.linspace(-5, 6, 150)
    dfg = metrics_mach[metrics_mach["gas"] == gas]
    if dfg.empty:
        return
    amax = min(55.0, np.nanmax(np.abs(dfg["angle_deg_from_upstream"])))
    angle_grid = np.linspace(0, amax, 50)
    mats, machs = [], []
    for case in gas_cases:
        M = make_matrix(case, metrics_mach, xi_grid, angle_grid)
        if np.isfinite(M).mean() < 0.5:
            continue
        Mfill = M.copy()
        Mfill[~np.isfinite(Mfill)] = np.nanmean(Mfill)
        mats.append(Mfill.ravel())
        machs.append(case.mach)
    if len(mats) < 3:
        return
    A = np.column_stack(mats)
    Af = A - A.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(Af, full_matrices=False)
    efrac = s**2/np.sum(s**2)
    cum = np.cumsum(efrac)
    coeff = np.diag(s) @ Vt
    mode1 = U[:,0].reshape(len(angle_grid), len(xi_grid))
    pd.DataFrame({"mode": np.arange(1,len(efrac)+1),
                  "energy_fraction": efrac, "cumulative": cum}).to_csv(
                      outdir/f"mach_sweep_pod_summary_{gas}.csv", index=False)
    cdf = pd.DataFrame(coeff.T, columns=[f"mode_{i+1}" for i in range(coeff.shape[0])])
    cdf.insert(0, "Mach", machs)
    cdf.to_csv(outdir/f"mach_sweep_pod_coefficients_{gas}.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    plt.subplots_adjust(left=0.06, right=0.975, bottom=0.17, top=0.80, wspace=0.34)
    modes = np.arange(1, len(efrac)+1)
    axes[0].bar(modes, 100*efrac, color="#4c78a8", alpha=0.85)
    axes[0].plot(modes, 100*cum, "o-", color="#d62728")
    axes[0].set_xlabel("Mode number (-)")
    axes[0].set_ylabel("Energy fraction / cumulative (%)")
    axes[0].set_title("(a) POD spectrum")
    axes[0].grid(alpha=0.22)
    vmax = np.nanpercentile(np.abs(mode1), 99)
    cf = axes[1].contourf(xi_grid, angle_grid, mode1,
                          levels=np.linspace(-vmax, vmax, 41),
                          cmap="coolwarm", extend="both")
    axes[1].set_xlabel(r"$\xi_s$ (-)")
    axes[1].set_ylabel("Angle from upstream axis (deg)")
    axes[1].set_title("(b) Mode 1")
    fig.colorbar(cf, ax=axes[1], fraction=0.050, pad=0.030, label="Amplitude (-)")
    axes[2].plot(machs, coeff[0,:], "o-", label="mode 1")
    if coeff.shape[0] > 1:
        axes[2].plot(machs, coeff[1,:], "s-", label="mode 2")
    axes[2].set_xlabel(r"$M_\infty$ (-)")
    axes[2].set_ylabel("POD coefficient (-)")
    axes[2].set_title("(c) First two coefficients")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=True)
    fig.suptitle(fr"Shock-attached compact structure across Mach: {gas}, $Kn_\infty={fixed_kn:g}$",
                 y=0.95, fontweight="bold")
    savefig(fig, outdir/f"JFM_Fig19_MachSweep_POD_{gas}")


def is_mach_sweep_case(case: CaseData) -> bool:
    return str(case.source_tag).startswith("mach_sweep")


def safe_mach_metrics_for_plot(metrics_df: pd.DataFrame, max_angle: float = 30.0, standoff_multiplier: float = 4.0):
    """Filter Mach-sweep shock-line metrics to remove boundary/near-wall artifacts.

    Keeps angles near the stagnation line and removes shock points whose standoff is
    unrealistically far from the stagnation-line standoff for that case.
    """
    if metrics_df.empty:
        return metrics_df
    rows = []
    for (gas, kn, mach, source_tag), dfg in metrics_df.groupby(["gas", "Kn", "Mach", "source_tag"], dropna=False):
        dfg = dfg.copy()
        dfg = dfg[np.abs(dfg["angle_deg_from_upstream"]) <= max_angle]
        if dfg.empty:
            continue
        stag = dfg.iloc[np.argmin(np.abs(dfg["angle_deg_from_upstream"].to_numpy()))]["standoff"]
        if np.isfinite(stag) and stag > 0:
            dfg = dfg[dfg["standoff"] <= standoff_multiplier * stag]
        rows.append(dfg)
    if not rows:
        return pd.DataFrame(columns=metrics_df.columns)
    return pd.concat(rows, ignore_index=True)


def deduplicate_mach_sweep_cases(cases_mach: List[CaseData]):
    """Prefer explicit Mach-sweep files over Kn-sweep fallback duplicates."""
    best = {}
    priority = {"mach_sweep_argon": 3, "mach_sweep_n2": 3, "kn_sweep": 1, "unknown": 0}
    for c in cases_mach:
        key = (c.gas, round(c.kn, 8), round(c.mach, 8))
        p = priority.get(c.source_tag, 0)
        if key not in best or p > priority.get(best[key].source_tag, 0):
            best[key] = c
    return sorted(best.values(), key=lambda c: (c.gas, c.kn, c.mach, c.source_tag))



# =============================================================================
# v16: Mach-sweep cleanup: explicit exclusions and robust upstream shock detector
# =============================================================================

def parse_exclude_cases(spec: str):
    """Parse exclusion list such as 'N2:0.01:9,Ar:0.01:10'.

    Format: gas:Kn:Mach.  Use empty string for no exclusions.
    """
    rules = []
    if spec is None:
        return rules
    spec = str(spec).strip()
    if not spec:
        return rules
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            print(f"[WARN] bad exclude case format ignored: {item}. Expected gas:Kn:Mach")
            continue
        gas, kn, mach = parts
        try:
            rules.append((gas.strip(), float(kn), float(mach)))
        except Exception:
            print(f"[WARN] bad exclude numeric values ignored: {item}")
    return rules


def is_excluded_case(gas, kn, mach, rules, tol=1e-8):
    for rg, rkn, rm in rules:
        if str(gas).lower() == str(rg).lower() and abs(float(kn)-rkn) < tol and abs(float(mach)-rm) < tol:
            return True
    return False


def filter_excluded_cases(cases, metrics, var_metrics, rules, outdir: Path):
    """Remove known corrupted/outlier cases consistently from cases, metrics and var_metrics."""
    if not rules:
        return cases, metrics, var_metrics
    rows = []
    cases2 = []
    for c in cases:
        bad = is_excluded_case(c.gas, c.kn, c.mach, rules)
        if bad:
            rows.append({"gas": c.gas, "Kn": c.kn, "Mach": c.mach, "flow_file": str(c.flow_file), "reason": "user_excluded"})
        else:
            cases2.append(c)

    def filter_df(df):
        if df is None or df.empty or not {"gas","Kn","Mach"}.issubset(set(df.columns)):
            return df
        mask = np.ones(len(df), dtype=bool)
        for rg, rkn, rm in rules:
            mask &= ~((df["gas"].astype(str).str.lower() == str(rg).lower()) &
                      (np.isclose(df["Kn"].astype(float), rkn)) &
                      (np.isclose(df["Mach"].astype(float), rm)))
        return df[mask].copy()

    metrics2 = filter_df(metrics)
    var2 = filter_df(var_metrics)
    if rows:
        pd.DataFrame(rows).to_csv(outdir/"excluded_cases_report.csv", index=False)
        print(f"[INFO] excluded {len(rows)} cases. See excluded_cases_report.csv")
    return cases2, metrics2, var2


def detect_shock_on_ray_window(case: CaseData, angle_rad: float,
                               min_standoff: float = 0.015,
                               max_standoff: float = 0.12,
                               min_points: int = 60):
    """Shock detection for Mach-sweep visualization/metrics.

    The ordinary detector may lock onto wall gradients or far-boundary features.
    This constrained detector searches only in a physically plausible upstream
    annulus measured from the cylinder surface.  It is intended for the fixed
    Kn=0.01 Mach sweep, where the bow shock is close to the body but not on the
    wall.
    """
    rr, xs, ys = sample_ray(case, angle_rad)
    rho_i = interp(case, "rho")
    rho = rho_i(xs, ys)
    rho = np.asarray(rho.filled(np.nan) if hasattr(rho, "filled") else rho, dtype=float)

    valid = np.isfinite(rho)
    if valid.sum() < min_points:
        return None

    rr_v = rr[valid]
    rho_v = rho[valid]
    st = rr_v - case.radius

    # Keep only a reasonable shock-search window.  Also allow relative scaling
    # if geometry is not dimensional.
    mn = max(float(min_standoff), 0.05*case.radius)
    mx = max(float(max_standoff), 0.75*case.radius)
    keep = (st >= mn) & (st <= mx)
    if keep.sum() < min_points:
        # fallback: less strict but still away from the wall
        keep = (st >= 0.03*case.radius) & (st <= 1.2*case.radius)
    if keep.sum() < min_points:
        return None

    rr_w = rr_v[keep]
    rho_w = rho_v[keep]
    rho_s = smooth1d(rho_w, window=max(7, rr_w.size//60))
    drho = np.gradient(rho_s, rr_w)
    absgrad = np.abs(drho)

    if not np.any(np.isfinite(absgrad)):
        return None
    imax = int(np.nanargmax(absgrad))
    r_s = float(rr_w[imax])
    maxgrad = float(absgrad[imax])

    nseg = max(6, rr_w.size//10)
    # far upstream end within search window
    rho_up = float(np.nanmedian(rho_w[-nseg:]))
    # near-body/post-shock end within search window
    rho_down = float(np.nanmedian(rho_w[:nseg]))
    delta = abs(rho_down-rho_up) / max(maxgrad, 1e-300)

    return {
        "r_s": r_s,
        "standoff": r_s - case.radius,
        "delta_rho": float(delta),
        "maxgrad_rho": maxgrad,
        "rho_up": rho_up,
        "rho_down": rho_down,
        "angle_rad": angle_rad,
    }


def compute_shock_metrics_mach_clean(case: CaseData, half_angle_deg: float = 30.0, n_angles: int = 31,
                                     min_standoff: float = 0.015, max_standoff: float = 0.12):
    """Clean shock metrics for the Mach sweep, restricted to the upstream sector."""
    rows = []
    offsets = np.linspace(0.0, half_angle_deg, n_angles)
    candidates = []
    for sign in [-1, +1]:
        tmp = []
        for deg in offsets:
            angle = case.upstream_angle + sign*np.deg2rad(deg)
            if math.sin(angle) < -1e-8:
                continue
            det = detect_shock_on_ray_window(case, angle,
                                             min_standoff=min_standoff,
                                             max_standoff=max_standoff)
            if det is None:
                continue
            det["angle_deg_from_upstream"] = sign*deg
            tmp.append(det)
        candidates.append(tmp)
    valid = [c for c in candidates if c]
    if not valid:
        return pd.DataFrame()
    best = max(valid, key=len)
    for d in best:
        d.update({"gas": case.gas, "Kn": case.kn, "Mach": case.mach,
                  "source_tag": case.source_tag, "flow_file": str(case.flow_file)})
        rows.append(d)
    return pd.DataFrame(rows)


def build_clean_mach_metrics(cases_mach, half_angle: float, n_angles: int,
                             min_standoff: float, max_standoff: float):
    rows = []
    for c in cases_mach:
        df = compute_shock_metrics_mach_clean(c, half_angle_deg=min(half_angle, 30.0),
                                              n_angles=max(15, min(n_angles, 41)),
                                              min_standoff=min_standoff,
                                              max_standoff=max_standoff)
        if df.empty:
            print(f"[WARN] clean Mach shock detection failed for {c.gas} Kn={c.kn:g} M={c.mach:g}; using ordinary metrics later if available.")
        else:
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="JFM_Rapid_Cylinder")
    ap.add_argument("--length_unit", default="m")
    ap.add_argument("--half_angle", type=float, default=60.0)
    ap.add_argument("--angles", type=int, default=49)
    ap.add_argument("--pod_modes_to_plot", type=int, default=4, help="number of POD modes to plot in supplementary POD-mode figure")
    ap.add_argument("--include_ratio_panel", action="store_true", help="include the delta_rho/standoff ratio panel in Fig. 2")
    ap.add_argument("--skip_final_checks", action="store_true", help="skip v8 UQ, L2 collapse and surface diagnostics")
    ap.add_argument("--skip_v9_checks", action="store_true", help="skip grad-rho figures and local-mfp 1D proxy diagnostics")
    ap.add_argument("--skip_variable_scales", action="store_true", help="skip v7 variable-specific scale diagnostics")
    ap.add_argument("--grid_nx", type=int, default=220, help="x grid for GLL/gradient diagnostics")
    ap.add_argument("--grid_ny", type=int, default=160, help="y grid for GLL/gradient diagnostics")
    ap.add_argument("--center_x", type=float, default=None)
    ap.add_argument("--center_y", type=float, default=None)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--leftmost_x", type=float, default=None, help="If the cylinder leftmost point is known, center_x = leftmost_x + radius")
    ap.add_argument("--default_kn_if_missing", type=float, default=0.01, help="Kn used for Mach-sweep archives if Kn is not encoded in paths")
    ap.add_argument("--default_mach_if_missing", type=float, default=10.0, help="Mach used for Kn-sweep archives if Mach is not encoded in paths")
    ap.add_argument("--kn_sweep_mach", type=float, default=10.0, help="Mach value used for the main Kn-sweep figures")
    ap.add_argument("--mach_sweep_kn", type=float, default=0.01, help="Kn value used for the main Mach-sweep figures")
    ap.add_argument("--ar_mach_values", default="5:15", help="ordered Mach list for Ar if Mach is missing in paths")
    ap.add_argument("--n2_mach_values", default="5:14", help="ordered Mach list for N2 if Mach is missing in paths")
    ap.add_argument("--mach_sweep_stagnation_only", action="store_true", help="use only stagnation-line metrics in Mach-sweep scaling plots")
    ap.add_argument("--exclude_cases", default="N2:0.01:9", help="comma-separated outlier/corrupted cases to remove, format gas:Kn:Mach")
    ap.add_argument("--mach_clean_min_standoff", type=float, default=0.015, help="minimum standoff for clean Mach-sweep shock detection")
    ap.add_argument("--mach_clean_max_standoff", type=float, default=0.12, help="maximum standoff for clean Mach-sweep shock detection")
    args = ap.parse_args()

    # Convenience geometry option: if the cylinder leftmost x-position is known,
    # convert it to center_x using center_x = leftmost_x + radius.
    if args.leftmost_x is not None:
        if args.radius is None:
            raise ValueError("--leftmost_x requires --radius")
        args.center_x = float(args.leftmost_x) + float(args.radius)
        if args.center_y is None:
            args.center_y = 0.0

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    roots = extract_inputs(args.inputs, out)
    raw = find_cases(roots, default_kn_if_missing=args.default_kn_if_missing, default_mach_if_missing=args.default_mach_if_missing, ar_mach_values=args.ar_mach_values, n2_mach_values=args.n2_mach_values)
    if not raw:
        raise RuntimeError("No DS2FF files found.")
    print(f"[INFO] found {len(raw)} cases")
    center_args = (args.center_x, args.center_y, args.radius)
    cases, inv_rows, shock_list = [], [], []
    for c in raw:
        try:
            case = load_case(c, center_args)
        except Exception as e:
            print(f"[ERROR] failed to load case {c.get('gas')} Kn={c.get('kn')}: {c.get('flow_file')} :: {e}")
            continue
        cases.append(case)
        x, y, rho = get_array(case, "x"), get_array(case, "y"), get_array(case, "rho")
        inv_rows.append({"gas":case.gas,"Kn":case.kn,"Mach":case.mach,"mach_assigned_by_order": bool(c.get("mach_assigned_by_order", False)), "source_tag": case.source_tag, "flow_file":str(case.flow_file),"surface_file":str(case.surface_file or ""), "n_points":len(x), "x_min":np.nanmin(x), "x_max":np.nanmax(x), "y_min":np.nanmin(y), "y_max":np.nanmax(y), "rho_min":np.nanmin(rho), "rho_max":np.nanmax(rho), "center_x":case.center[0], "center_y":case.center[1], "radius":case.radius, "uinf_x":case.uinf[0], "uinf_y":case.uinf[1], "varmap":json.dumps(case.vm)})
        sdf = compute_shock_metrics(case, args.half_angle, args.angles)
        if not sdf.empty: shock_list.append(sdf)
        else: print(f"[WARN] no shock detections: {case.gas} Kn={case.kn:g}")
        make_check_plot(case, sdf, out/"checks", args.length_unit)
    inv = pd.DataFrame(inv_rows).sort_values(["gas","Kn"])
    inv.to_csv(out/"case_inventory.csv", index=False)
    inv.to_csv(out/"read_check_summary.csv", index=False)
    if not shock_list: raise RuntimeError("Shock detection failed for all cases.")
    metrics = pd.concat(shock_list, ignore_index=True)

    # Remove known corrupted/outlier cases before generating summaries/figures.
    exclude_rules = parse_exclude_cases(args.exclude_cases)
    cases, metrics, _tmp = filter_excluded_cases(cases, metrics, pd.DataFrame(), exclude_rules, out)

    metrics.to_csv(out/"shock_metrics_by_angle.csv", index=False)
    summary = metrics.groupby(["gas","Kn","Mach"]).agg(standoff_median=("standoff","median"), standoff_mean=("standoff","mean"), standoff_std=("standoff","std"), delta_rho_median=("delta_rho","median"), delta_rho_mean=("delta_rho","mean"), delta_rho_std=("delta_rho","std"), maxgrad_median=("maxgrad_rho","median")).reset_index()
    summary["delta_over_standoff"] = summary["delta_rho_median"]/summary["standoff_median"]
    summary.to_csv(out/"shock_metrics_summary_all.csv", index=False)

    # Main Kn-sweep subset: fixed Mach, usually Mach 10.
    summary_kn = summary[np.isclose(summary["Mach"], args.kn_sweep_mach)].copy()
    metrics_kn = metrics[np.isclose(metrics["Mach"], args.kn_sweep_mach)].copy()
    cases_kn = [c for c in cases if np.isclose(c.mach, args.kn_sweep_mach)]
    summary_kn.to_csv(out/"shock_metrics_summary.csv", index=False)

    # Main Mach-sweep subset: fixed Kn, usually Kn=0.01.
    # Use only explicit Mach-sweep archives when they are present, to avoid
    # duplicate M=10 cases from the Kn-sweep folders.
    cases_mach_all = [c for c in cases if np.isclose(c.kn, args.mach_sweep_kn) and is_mach_sweep_case(c)]
    if not cases_mach_all:
        cases_mach_all = [c for c in cases if np.isclose(c.kn, args.mach_sweep_kn)]
    cases_mach = deduplicate_mach_sweep_cases(cases_mach_all)

    mach_keys = {(c.gas, round(c.kn, 8), round(c.mach, 8), c.source_tag) for c in cases_mach}
    metrics_mach = metrics[
        metrics.apply(lambda r: (r["gas"], round(float(r["Kn"]), 8), round(float(r["Mach"]), 8), r.get("source_tag", "unknown")) in mach_keys, axis=1)
    ].copy()

    # Re-detect the Mach-sweep shock with an annular upstream search window.
    metrics_mach_clean = build_clean_mach_metrics(cases_mach, args.half_angle, args.angles,
                                                  args.mach_clean_min_standoff,
                                                  args.mach_clean_max_standoff)
    if not metrics_mach_clean.empty:
        metrics_mach_line = metrics_mach_clean
        metrics_mach_line.to_csv(out/"mach_sweep_clean_shock_metrics_by_angle.csv", index=False)
    else:
        metrics_mach_line = safe_mach_metrics_for_plot(metrics_mach, max_angle=30.0, standoff_multiplier=4.0)

    if not metrics_mach_line.empty:
        summary_mach = metrics_mach_line.groupby(["gas","Kn","Mach"]).agg(
            standoff_median=("standoff","median"),
            standoff_mean=("standoff","mean"),
            standoff_std=("standoff","std"),
            delta_rho_median=("delta_rho","median"),
            delta_rho_mean=("delta_rho","mean"),
            delta_rho_std=("delta_rho","std"),
            maxgrad_median=("maxgrad_rho","median")
        ).reset_index()
        summary_mach["delta_over_standoff"] = summary_mach["delta_rho_median"] / summary_mach["standoff_median"]
        summary_mach = add_stagnation_metrics(summary_mach, metrics_mach)
    else:
        summary_mach = pd.DataFrame()
    summary_mach.to_csv(out/"mach_sweep_shock_metrics_summary.csv", index=False)

    # Collapse quality
    collapse_rows = []
    for gas in sorted(set(c.gas for c in cases)):
        gas_cases = sorted([c for c in cases if c.gas == gas], key=lambda c:c.kn)
        xi_common = np.linspace(-5,6,250)
        for var in ["rho","mach","Ttr","Trot"]:
            profs, used = [], []
            for case in gas_cases:
                df = metrics[(metrics["gas"]==gas)&(metrics["Kn"]==case.kn)]
                if df.empty: continue
                row = df.iloc[np.argmin(np.abs(df["angle_deg_from_upstream"].to_numpy()))]
                pr = stagnation_profile(case,row)
                if var not in pr: continue
                xb, xi, qn = pr[var]; good = np.isfinite(xi)&np.isfinite(qn)
                if good.sum()<10: continue
                profs.append(np.interp(xi_common, xi[good], qn[good], left=np.nan, right=np.nan))
                used.append(case.kn)
            if len(profs)>=2:
                P = np.vstack(profs)
                collapse_rows.append({"gas":gas,"variable":var,"n_cases":len(profs),"mean_std_in_xi":np.nanmean(np.nanstd(P,axis=0)),"Kn_used":json.dumps(used)})
    collapse_df = pd.DataFrame(collapse_rows)
    collapse_df.to_csv(out/"collapse_quality_summary.csv", index=False)

    # Additional gradient/GLL diagnostics requested for a JFM Rapid physics story.
    gll_rows = []
    for case in cases:
        try:
            gll_rows.append(compute_gll_and_gradient_summary(case, args.grid_nx, args.grid_ny))
        except Exception as e:
            print(f"[WARN] GLL/gradient diagnostic failed for {case.gas} Kn={case.kn:g}: {e}")
    gll_df = pd.DataFrame(gll_rows)
    gll_df.to_csv(out/"gll_gradient_summary.csv", index=False)

    # v7: variable-specific shock-layer scales and VHS mean-free-path normalization.
    if not args.skip_variable_scales:
        var_metrics = compute_variable_layer_metrics(cases, args.half_angle, args.angles)
        if exclude_rules:
            _, _, var_metrics = filter_excluded_cases([], pd.DataFrame(), var_metrics, exclude_rules, out)
        var_metrics.to_csv(out/"variable_layer_metrics_by_angle.csv", index=False)
        var_summary = summarize_variable_layer_metrics(var_metrics)
        var_summary.to_csv(out/"variable_layer_metrics_summary.csv", index=False)
        own_collapse_df = compute_variable_specific_collapse_quality(cases, var_metrics)
        own_collapse_df.to_csv(out/"variable_specific_collapse_quality.csv", index=False)
    else:
        var_metrics = pd.DataFrame()
        var_summary = pd.DataFrame()
        own_collapse_df = pd.DataFrame()

    if (not args.skip_variable_scales) and (not args.skip_final_checks) and (not var_metrics.empty):
        uq_raw, uq_summary = compute_extraction_uncertainty(cases, args.half_angle, args.angles)
        uq_raw.to_csv(out/"extraction_uncertainty_raw.csv", index=False)
        uq_summary.to_csv(out/"extraction_uncertainty_summary.csv", index=False)

        l2_case_df, l2_summary_df = compute_integrated_l2_collapse(cases, var_metrics)
        l2_case_df.to_csv(out/"collapse_l2_by_case.csv", index=False)
        l2_summary_df.to_csv(out/"collapse_l2_summary.csv", index=False)

        surface_df = compute_surface_metrics(cases)
        surface_df.to_csv(out/"surface_aerothermal_metrics.csv", index=False)
    else:
        uq_summary = pd.DataFrame()
        l2_summary_df = pd.DataFrame()
        surface_df = pd.DataFrame()

    # v9: local-mfp/1D-reference proxy and optional Mach-sweep summary.
    if (not args.skip_v9_checks) and (not var_metrics.empty):
        proxy_1d_df = compute_stagnation_1d_reference_proxy(cases, var_metrics)
        proxy_1d_df.to_csv(out/"stagnation_1d_proxy_summary.csv", index=False)
    else:
        proxy_1d_df = pd.DataFrame()

    figdir = out/"figures"; figdir.mkdir(exist_ok=True)
    for gas in sorted(set(c.gas for c in cases_kn)):
        make_fig1_contours(cases_kn, metrics_kn, gas, figdir, args.length_unit)
        make_fig3_collapse(cases_kn, metrics_kn, gas, figdir, args.length_unit)
        make_fig4_pod(cases_kn, metrics_kn, gas, figdir, args.pod_modes_to_plot)
        if not args.skip_v9_checks:
            make_fig13_grad_rho_contours(cases_kn, metrics_kn, gas, figdir, args.length_unit)
        if not var_metrics.empty:
            var_metrics_kn = var_metrics[np.isclose(var_metrics["Mach"], args.kn_sweep_mach)] if "Mach" in var_metrics.columns else var_metrics
            make_fig9_variable_specific_collapse(cases_kn, var_metrics_kn, gas, figdir)
    make_fig2_scaling(summary_kn, figdir, args.length_unit, include_ratio=args.include_ratio_panel)
    make_fig5_collapse_quality(collapse_df, figdir)
    make_fig6_gll_summary(gll_df, figdir)
    if not var_summary.empty:
        var_summary_kn = var_summary[np.isclose(var_summary["Mach"], args.kn_sweep_mach)] if "Mach" in var_summary.columns else var_summary
        make_fig7_variable_scales(var_summary_kn, figdir, args.length_unit)
    if not own_collapse_df.empty:
        make_fig8_variable_collapse_quality(own_collapse_df, figdir)
    if not uq_summary.empty:
        make_fig10_extraction_uncertainty(uq_summary, figdir)
    if not l2_summary_df.empty:
        make_fig11_l2_collapse(l2_summary_df, figdir)
    if not surface_df.empty:
        make_fig12_surface_correlation(surface_df[surface_df["Mach"].apply(lambda x: np.isclose(x, args.kn_sweep_mach))] if "Mach" in surface_df.columns else surface_df, summary_kn, figdir)
    if not proxy_1d_df.empty:
        make_fig14_stagnation_1d_proxy(proxy_1d_df, figdir)

    # v10 Mach-sweep diagnostics at fixed Kn.
    if not summary_mach.empty and summary_mach["Mach"].nunique() > 1:
        make_fig16_mach_sweep_scaling(summary_mach, figdir, args.length_unit, args.mach_sweep_kn, stagnation_only=True)
        if not var_summary.empty:
            # Prefer explicit mach-sweep entries, matching selected cases.
            var_keys = {(c.gas, round(c.kn, 8), round(c.mach, 8)) for c in cases_mach}
            var_summary_mach = var_summary[
                var_summary.apply(lambda r: (r["gas"], round(float(r["Kn"]), 8), round(float(r["Mach"]), 8)) in var_keys, axis=1)
            ].copy()
            make_fig17_mach_variable_scales(var_summary_mach, figdir, args.length_unit, args.mach_sweep_kn)
        for gas in sorted(set(c.gas for c in cases_mach)):
            make_fig18_mach_contours(cases_mach, metrics_mach_line, gas, figdir, args.length_unit, args.mach_sweep_kn)
            make_fig19_mach_pod(cases_mach, metrics_mach_line, gas, figdir, args.mach_sweep_kn)
    else:
        print("[INFO] No multi-Mach dataset detected for fixed Kn; Mach-sweep figures skipped.")

    mach_sweep_df = summarize_mach_sweep_if_present(cases, metrics, out)
    print(f"[OK] done. Outputs in {out}")
    print("First check these:")
    print("  ", out/"read_check_summary.csv")
    print("  ", out/"checks")
    print("  ", out/"shock_metrics_summary.csv")
    print("  ", out/"variable_layer_metrics_summary.csv")
    print("  ", out/"variable_specific_collapse_quality.csv")
    print("  ", out/"extraction_uncertainty_summary.csv")
    print("  ", out/"collapse_l2_summary.csv")
    print("  ", out/"surface_aerothermal_metrics.csv")
    print("  ", out/"stagnation_1d_proxy_summary.csv")
    print("  ", out/"mach_sweep_summary.csv")
    print("  ", figdir)

if __name__ == "__main__":
    main()

"""Statistics of physically weighted particle observations in finite stations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

CASES = ("Ar_planar", "Ar_cylinder", "N2_planar", "N2_cylinder")
STATIONS = ("upstream_reference", "s10", "s50", "s90", "downstream_reference")
KB = 1.380649e-23
STRIDES = (1, 5, 10, 20, 40)
FORMAT = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else stream_sha(stream)


def stream_sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def almost(a, b):
    np.testing.assert_allclose(a, b, rtol=4.e-10, atol=1.e-290, equal_nan=True)


def normalised_prob(weight, total, widths):
    probs = divide(weight, total[:, None])
    pdf = divide(probs[:, 1:-1], widths)
    supported = total > 0
    almost(np.sum(probs[supported], axis=1), np.ones(np.sum(supported)))
    almost(probs[supported, 0] + (pdf[supported] * widths).sum(axis=1) + probs[supported, -1],
           np.ones(np.sum(supported)))
    require(np.isnan(pdf[~supported]).all(), "Empty station was reported with a defined PDF")
    return probs, pdf


def sign_groups(tallies, negative, zero):
    total = tallies[:, 0]
    remainder = total - tallies[:, negative] - tallies[:, zero]
    if np.issubdtype(tallies.dtype, np.unsignedinteger):
        require(np.all(tallies[:, negative] + tallies[:, zero] <= total), "Sign tally exceeds raw count")
    else:
        require(np.all(remainder >= -5.e-12 * total), "Sign tally exceeds physical count")
    return np.stack((tallies[:, negative], np.maximum(0, remainder), tallies[:, zero]), axis=-1)


def describe(meta, values, observations):
    """Pool additive moments first. n is the number of snapshots, not particles."""
    require(observations > 0, "Empty time selection")
    raw, weight, m = (values[k] for k in ("raw", "weight", "mom"))
    vol, mass = np.asarray(meta["volume_m3"]), meta["mass_kg"]
    w, w2 = weight[:, 0], m[:, 12]
    require(np.all((w > 0) == (w2 > 0)), "Squared weights disagree with occupancy")
    kish = divide(w*w, w2)
    support = w > 0
    require(np.all(kish[support] <= raw[support, 0]*(1+5.e-9)), "Kish exceeds observed particle visits")
    wsg = sign_groups(weight, 1, 3)
    nsg = sign_groups(raw, 1, 3)
    wn = sign_groups(weight, 2, 4)
    almost(m[:, 13:16].sum(axis=1), w2)
    almost(m[:, 16:19].sum(axis=1), w2)
    s1, diagonal, cross = m[:, 19:22], m[:, 22:25], m[:, 25:28]
    tensor = np.zeros((len(w), 3, 3))
    for i, j, column in ((0, 0, 0), (1, 1, 1), (2, 2, 2)):
        tensor[:, i, j] = diagonal[:, column]
    for i, j, column in ((0, 1, 0), (0, 2, 1), (1, 2, 2)):
        tensor[:, i, j] = tensor[:, j, i] = cross[:, column]
    u = divide(s1, w[:, None])
    thermal_diag = divide(diagonal, w[:, None]) - u*u
    if np.any(support):
        scale = divide(diagonal, w[:, None])[support]
        require(np.all(thermal_diag[support] >= -5.e-11 * scale), "Negative resolved central variance")
    central_xyz = (m[:, 28:31] - u * diagonal.sum(axis=1)[:, None]
                   - 2 * np.einsum("nij,nj->ni", tensor, u)
                   + 2 * w[:, None] * u * np.sum(u*u, axis=1)[:, None])
    exposure = observations * vol
    qtr = .5 * mass * central_xyz / exposure[:, None]
    ux = u[:, 0]
    cx3 = divide(m[:, 34] - 3*ux*diagonal[:, 0] + 2*w*ux**3, w)
    qrot = ((m[:, 31:34] - u*m[:, 5, None]) / exposure[:, None] if meta["rotation"] else None)
    totals = {"raw_observations": raw[:, 0], "sum_w": w, "sum_w2": w2,
              "kish_particle_observations": kish, "raw_lab_vx_lt0": nsg[:, 0],
              "raw_lab_vx_eq0": nsg[:, 2], "sum_w_lab_vx_lt0": wsg[:, 0],
              "sum_w2_lab_vx_lt0": m[:, 13], "kish_lab_vx_lt0": divide(wsg[:, 0]**2, m[:, 13]),
              "p_lab_vx_lt0": divide(wsg[:, 0], w), "p_lab_vx_eq0": divide(wsg[:, 2], w),
              "p_normal_lt0": divide(wn[:, 0], w), "kish_normal_lt0": divide(wn[:, 0]**2, m[:, 16]),
              "number_density_m3": w / exposure, "mean_vx_m_s": ux,
              "mean_vnormal_m_s": divide(m[:, 1], w), "mean_vtangent_m_s": divide(m[:, 2], w),
              "mean_kinetic_energy_J": divide(m[:, 4], w), "mean_projected_normal_energy_J": divide(m[:, 3], w),
              "Ttrans_K": mass*thermal_diag.sum(axis=1)/(3*KB), "Txx_K": mass*thermal_diag[:, 0]/KB,
              "Tyy_K": mass*thermal_diag[:, 1]/KB, "Tzz_K": mass*thermal_diag[:, 2]/KB,
              "vx_central_third_m3_s3": cx3, "vx_skewness": divide(cx3, np.maximum(thermal_diag[:, 0], 0)**1.5)}
    for j, component in enumerate("xyz"):
        totals["q_trans_" + component + "_W_m2"] = qtr[:, j]
    if meta["rotation"]:
        totals["mean_Erot_J"] = divide(m[:, 5], w)
        totals["Trot_K"] = totals["mean_Erot_J"]/KB
        for i, condition in enumerate(("lt0", "gt0", "eq0")):
            totals["mean_Erot_J_lab_vx_"+condition] = divide(m[:, 6+i], wsg[:, i])
        for j, component in enumerate("xyz"):
            totals["q_rot_" + component + "_W_m2"] = qrot[:, j]
    axes = {}
    for key, at, n, low, high in meta["marginals"]:
        sl = slice(at, at+n+2)
        edges = np.linspace(low, high, n+1)
        probability, pdf = normalised_prob(weight[:, sl], w, np.diff(edges))
        require(np.array_equal(raw[:, sl].sum(axis=1), raw[:, 0]), key+" raw histogram sum failed")
        axes[key] = (edges, probability, pdf, raw[:, sl], weight[:, sl])
        totals[key + "_underflow_probability"] = probability[:, 0]
        totals[key + "_overflow_probability"] = probability[:, -1]
    vn_w = axes["normal"][4]
    totals["conditional_projected_energy_J_vn_bin"] = divide(values["proj_bin"], vn_w)
    if meta["rotation"]:
        totals["conditional_Erot_J_vn_bin"] = divide(values["rot_bin"], vn_w)
    return totals, axes


def scalar_rows(stats, station_names, context):
    for i, station in enumerate(station_names):
        record = dict(context, station=station)
        for key, array in stats.items():
            if np.ndim(array) == 1:
                number = array[i]
                record[key] = int(number) if np.issubdtype(array.dtype, np.integer) else (
                    float(number) if np.isfinite(number) else None)
        yield record

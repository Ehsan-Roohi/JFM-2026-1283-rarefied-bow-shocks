"""Reproduce the space-resolved nitrogen time-step comparison from four archived ray caches."""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
TEMP = 1
DENSITY = 0
UX = 2
TROT = 3
FIELDS = {"density": DENSITY, "Ttr": TEMP, "ux": UX, "Trot": TROT}


def norm_percent(values, reference):
    return float(100 * np.linalg.norm(values) / np.linalg.norm(reference))


def read_case(control, angle):
    ray = control["rays"][str(angle)]
    path = INPUTS / ray["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != ray["sha256"]:
        raise ValueError(f"Profile checksum mismatch for {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        grid, blocks, mean = (archive[k] for k in ("s", "blocks", "mean"))
    assert blocks.shape == (10, len(grid), 7)
    return grid, blocks, mean, ray["marker_mm"] * 1e-3


def write_csv(name, rows):
    with (HERE / name).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit():
    meta = json.loads((INPUTS / "metadata.json").read_text(encoding="utf-8"))
    lam = meta["lambda_inf_m"]
    controls = meta["controls"]
    summary, profile_rows, block_rows = {}, [], []
    for angle in (170, 180):
        coarse = controls["N2_5ns"]
        fine = controls["N2_2.5ns"]
        s, ba, a, ma = read_case(coarse, angle)
        s2, bb, b, mb = read_case(fine, angle)
        np.testing.assert_array_equal(s, s2)
        assert np.allclose(np.diff(s), 0.0001, rtol=0, atol=1e-15)
        lo, hi = meta["range_m"]
        mask = (s >= lo) & (s <= hi)
        assert mask.sum() == 1451
        for av, blocks in ((a, ba), (b, bb)):
            assert np.isfinite(blocks[:, mask, :4]).all()
            np.testing.assert_allclose(av[mask, :4], blocks[:, mask, :4].mean(axis=0), rtol=1e-12)
        qa, qb = a[:, TEMP], b[:, TEMP]
        residual = qa - qb
        total_ss = float(np.sum(residual[mask] ** 2))
        zones = {
            "wallward": mask & (s < mb - 2 * lam),
            "front": mask & (s >= mb - 2 * lam) & (s <= mb + 2 * lam),
            "upstream": mask & (s > mb + 2 * lam),
        }
        np.testing.assert_array_equal(sum(m.astype(int) for m in zones.values()), mask)
        zones_summary = {}
        for region, selected in zones.items():
            err = residual[selected]
            zones_summary[region] = dict(
                n=int(selected.sum()),
                first_last_mm=(1000 * s[selected][[0, -1]]).tolist(),
                share_of_total_squared_error_percent=float(100 * np.sum(err**2) / total_ss),
                regional_relative_L2_percent=norm_percent(err, qb[selected]),
                mean_signed_5_minus_2p5_K=float(np.mean(err)),
                largest_absolute_difference_K=float(np.max(np.abs(err))),
            )
        checked_shifts = {}
        for name, shift in (("density_marker", ma - mb), ("hypothetical_positive", .0006), ("hypothetical_negative", -.0006)):
            # A feature in the coarse profile originally at ma then appears at mb.
            aligned = np.interp(s + shift, s, qa, left=np.nan, right=np.nan)
            common = mask & np.isfinite(aligned) & np.isfinite(qb)
            assert np.array_equal(common, mask)
            checked_shifts[name] = dict(
                coarse_query_s_plus_mm=1000 * shift,
                same_support_points=int(common.sum()),
                unshifted_relative_L2_percent=norm_percent(residual[common], qb[common]),
                shifted_relative_L2_percent=norm_percent((aligned-qb)[common], qb[common]),
            )
        reduced = {k:norm_percent(a[mask,i]-b[mask,i], b[mask,i]) for k,i in FIELDS.items()}
        np.testing.assert_allclose(reduced["Ttr"], fine["rays"][str(angle)]["old_unshifted_Ttr_percent"], rtol=1e-10)
        peak_a = int(np.flatnonzero(mask)[np.argmax(qa[mask])])
        peak_b = int(np.flatnonzero(mask)[np.argmax(qb[mask])])
        worst = int(np.flatnonzero(mask)[np.argmax(np.abs(residual[mask]))])
        block_halves = {}
        for label, bulk, mean in (("5ns", ba, a), ("2.5ns", bb, b)):
            block_halves[label] = norm_percent(
                bulk[5:,mask,TEMP].mean(axis=0) - bulk[:5,mask,TEMP].mean(axis=0), mean[mask,TEMP])
        same_time_norms = []
        for k, (block5, block_half) in enumerate(zip(ba,bb)):
            d = block5[:,TEMP] - block_half[:,TEMP]
            norm = norm_percent(d[mask], block_half[mask,TEMP])
            same_time_norms.append(norm)
            block_rows.append(dict(angle_deg=angle, block=k+1, start_ms=f"{2+k*.2:.1f}",
                                   end_ms=f"{2+(k+1)*.2:.1f}", norm_percent=norm,
                                   upstream_squared_error_share_percent=
                                   float(100*np.sum(d[zones["upstream"]]**2)/np.sum(d[mask]**2))))
        info = dict(
            physical_unshifted_relative_L2_percent=reduced,
            density_markers_5_and_2p5_mm=[1000*ma,1000*mb],
            zones_about_halfstep_density_marker=zones_summary,
            native_outer_density_onset_2p5_mm=1000*fine['rays'][str(angle)]['outer_density_threshold_onset_m'],
            squared_error_percent_outside_density_onset=float(100*np.sum(
                residual[mask & (s > fine['rays'][str(angle)]['outer_density_threshold_onset_m'])]**2
            ) / total_ss),
            shift_diagnostics=checked_shifts,
            first5_last5_comparison_percent=block_halves,
            equal_time_individual_block_relative_L2_percent_range=[min(same_time_norms),max(same_time_norms)],
            worst_point=dict(s_mm=1000*s[worst], signed_difference_K=residual[worst],
                             coarse_K=qa[worst], fine_K=qb[worst]),
            coarse_peak=dict(s_mm=1000*s[peak_a],K=qa[peak_a]),
            fine_peak=dict(s_mm=1000*s[peak_b],K=qb[peak_b]),
            relative_difference_in_peak_values_percent=float(100*(qa[peak_a]-qb[peak_b])/qb[peak_b]),
            spatial_bounds_of_middle_80pct_squared_residual_mm=(
                1000*np.interp([.1,.9],np.cumsum(residual[mask]**2)/total_ss,s[mask])).tolist(),
        )
        summary[str(angle)] = info
        sda, sdb = np.std(ba[:,mask,TEMP],axis=0,ddof=1), np.std(bb[:,mask,TEMP],axis=0,ddof=1)
        for j, i in enumerate(np.flatnonzero(mask)):
            region = next(name for name, region_mask in zones.items() if region_mask[i])
            profile_rows.append(dict(angle_deg=angle, s_mm=f"{1000*s[i]:.1f}", region=region,
                                     Ttr_5ns_K=f"{qa[i]:.8f}",Ttr_2p5ns_K=f"{qb[i]:.8f}",
                                     signed_5_minus_2p5_K=f"{residual[i]:.8f}",
                                     rho_5ns_over_rhoinf=f"{a[i,DENSITY]:.10f}",rho_2p5ns_over_rhoinf=f"{b[i,DENSITY]:.10f}",
                                     Ttr_5ns_block_SD_K=f"{sda[j]:.8f}",Ttr_2p5ns_block_SD_K=f"{sdb[j]:.8f}"))
    write_csv("ttr_profiles.csv", profile_rows)
    write_csv("ttr_block_comparisons.csv", block_rows)
    (HERE / "ttr_diagnostics.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")


if __name__ == "__main__":
    audit()

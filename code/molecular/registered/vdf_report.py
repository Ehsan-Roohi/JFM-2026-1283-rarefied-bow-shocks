"""Report matched, fully completed four-case distributions with finite-block stability."""
import csv
import gzip
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from vdf_statistics import (CASES, FORMAT, STATIONS, STRIDES, describe, divide, file_sha, load_json,
                            require, save_json, scalar_rows)

SCALARS = ("p_lab_vx_lt0", "p_normal_lt0", "mean_vx_m_s", "Ttrans_K", "Txx_K", "vx_skewness",
           "mean_Erot_J", "mean_Erot_J_lab_vx_lt0", "mean_Erot_J_lab_vx_gt0", "q_trans_x_W_m2", "q_rot_x_W_m2")
WINDOWS = {"all": slice(0, 10), "development_1_to_5": slice(0, 5), "held_out_6_to_10": slice(5, 10)}


def write_csv(path, rows):
    require(rows, "No observations for "+str(path))
    # Argon omits rotational columns. Use fields from all observed gases.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_number(x):
    return float(x) if np.isfinite(x) else None


def case_blocks(compact, name, *, final=True):
    path = Path(compact) / name
    record = load_json(path/"case_status.json")
    require(record["format"] == FORMAT and record["case"] == name and 0<len(record["blocks"])<=10,
            "No valid compact native block prefix for "+name)
    if final:
        require(record["target_receipt_sha256"] and len(record["blocks"])==10,
                "Final report needs all ten native target-verified blocks for "+name)
    blocks = []
    meta = None
    for index, entry in enumerate(record["blocks"], start=1):
        dst = path / ("block_%02d.npz" % index)
        require(entry["file"] == dst.name and load_json(dst.with_suffix(".json")) == entry and
                file_sha(dst) == entry["sha256"] and entry["source"]["block"] == index, "Compact block digest differs")
        with np.load(dst, allow_pickle=False) as data:
            m = json.loads(str(data["metadata_json"]))
            require(json.loads(str(data["source_json"])) == entry["source"] and (not meta or meta == m) and
                    m["station_names"] == list(STATIONS) and m["case"] == name and m["rotation"] == name.startswith("N2_")
                    and data["strides"].tolist() == list(STRIDES), "Case/station/stride/provenance mismatch")
            meta = m
            block = {k: data[k] for k in ("raw", "weight", "mom", "proj_bin", "rot_bin", "strides", "snapshots",
                                           "times_s", "snapshot_series", "snapshot_counts", "sample_sha256",
                                           "joint_raw", "joint_weight", "native_ranks_eligible_listed")}
            require(len(block["times_s"]) == 200 and np.array_equal(block["snapshots"], 200//np.asarray(STRIDES))
                    and int(data["joint_stride"]) == 10, "Missing temporal or joint sampling")
        blocks.append(block)
    t = np.concatenate([b["times_s"] for b in blocks])
    expected = .004 + np.arange(1,len(blocks)*200+1)*1.e-6
    require(np.allclose(t, expected, rtol=0, atol=record["science"]["dt_s"]*.002),
            "Real observations do not cover the registered 4–6 ms continuation")
    hs = np.concatenate([b["sample_sha256"] for b in blocks])
    require(len(set(hs.tolist())) == len(expected), "Repeated native microrecord across windows")
    return meta, blocks, record


def pooled_values(blocks, stride):
    at = STRIDES.index(stride)
    sums = {k: np.sum([b[k][at] for b in blocks], axis=0) for k in ("raw", "weight", "mom", "proj_bin", "rot_bin")}
    return sums, sum(int(b["snapshots"][at]) for b in blocks)


def inspect_case(compact, name, stride=10):
    """Summarise only the contiguous accepted prefix; never generate four-case figures here."""
    m, blocks, state = case_blocks(compact, name, final=False)
    v, n = pooled_values(blocks, stride)
    stats, _ = describe(m, v, n)
    keys = ("raw_observations", "sum_w", "sum_w2", "kish_particle_observations", "raw_lab_vx_lt0",
            "p_lab_vx_lt0", "p_normal_lt0", "incident_underflow_probability", "incident_overflow_probability",
            "Ttrans_K", "Trot_K", "q_trans_x_W_m2", "q_rot_x_W_m2")
    stations = list(scalar_rows({key: stats[key] for key in keys if key in stats}, STATIONS, {}))
    return {"case": name, "accepted_blocks":len(blocks), "target_verified":bool(state["target_receipt_sha256"]),
            "chain_sha256":state["chain_sha256"], "stride_us":stride, "selected_snapshots":n,
            "first_native_time_s":float(blocks[0]["times_s"][0]), "last_native_time_s":float(blocks[-1]["times_s"][-1]),
            "stations":stations}


def pair_correlation(values, lag):
    left, right = values[:-lag], values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    n = int(valid.sum())
    if n < 3 or np.ptp(left[valid]) == 0 or np.ptp(right[valid]) == 0:
        return n, None
    u, v = left[valid] - left[valid].mean(), right[valid] - right[valid].mean()
    norm = np.sqrt(np.sum(u*u)*np.sum(v*v))
    return n, table_number(np.dot(u, v)/norm) if norm > 0 else None


def series_rows(name, meta, blocks, window, subset):
    sequences = np.concatenate([b["snapshot_series"] for b in blocks[subset]])
    names = ("station_molecule_weight", "p_lab_vx_lt0", "p_normal_lt0", "mean_vx_m_s", "mean_Erot_J")
    rows = []
    for s, station in enumerate(meta["station_names"]):
        for j, observable in enumerate(names):
            if observable == "mean_Erot_J" and not meta["rotation"]:
                continue
            for lag in (1, 5, 10, 20, 40, 100):
                pair_count, corr = pair_correlation(sequences[:, s, j], lag)
                rows.append(dict(case=name, window=window, station=station, observable=observable,
                                 lag_us=lag, valid_snapshot_pairs=pair_count, lagged_pearson_r=corr))
    return rows


def distribution_rows(name, window, meta, stats, axes, blocks):
    records = []
    for label, (edges, prob, density, count, weight) in axes.items():
        at = next(axis[1] for axis in meta["marginals"] if axis[0] == label)
        reference = STRIDES.index(10)
        counts_by_block = np.stack([b["raw"][reference, :, at:at+len(edges)+1] for b in blocks], axis=0)
        for j, station in enumerate(meta["station_names"]):
            for i in range(len(edges)+1):
                low, high = (float(edges[i-1]) if i else None), (float(edges[i]) if i < len(edges) else None)
                row = dict(case=name, station=station, window=window, stride_us=10, axis=label,
                           lo_Uinf=low, hi_Uinf=high, support="underflow" if i == 0 else (
                               "overflow" if i == len(edges) else "interior"), raw=int(count[j, i]),
                           contributing_time_blocks=int(np.count_nonzero(counts_by_block[:, j, i])),
                           available_time_blocks=len(blocks), smallest_block_raw=int(counts_by_block[:, j, i].min()),
                           largest_block_raw=int(counts_by_block[:, j, i].max()),
                           physical_weight=float(weight[j, i]), probability=table_number(prob[j, i]),
                           pdf_per_unit_v_over_U=table_number(density[j, i-1]) if 0 < i < len(edges) else None)
                if label == "normal":
                    row["conditional_projected_normal_energy_J"] = table_number(stats["conditional_projected_energy_J_vn_bin"][j, i])
                if label == "normal" and meta["rotation"]:
                    row["conditional_mean_Erot_J"] = table_number(stats["conditional_Erot_J_vn_bin"][j, i])
                records.append(row)
    return records


def block_variation(name, metric, stats_by_block, station, at, stride):
    b = np.asarray([s[metric][at] for s in stats_by_block], dtype=float)
    finite = b[np.isfinite(b)]
    return dict(case=name, station=station, stride_us=stride, variable=metric,
                supported_blocks=len(finite), minimum=table_number(np.min(finite)) if len(finite) else None,
                maximum=table_number(np.max(finite)) if len(finite) else None,
                mean_across_supported_blocks=table_number(finite.mean()) if len(finite) else None,
                block_standard_deviation=table_number(finite.std(ddof=1)) if len(finite)>1 else None)


def distribution_distance(p, q):
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        return None
    return float(.5*np.sum(np.abs(p-q)))


def write_figures(path, summaries, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"STIXGeneral", "mathtext.fontset":"stix", "font.size":9.0,
                         "axes.linewidth":.65, "pdf.fonttype":42, "xtick.direction":"in", "ytick.direction":"in",
                         "lines.solid_capstyle":"butt", "savefig.facecolor":"white"})
    style = {"planar":("#304F79", "-", "Planar"), "cylinder":("#B94C2C", (0, (4, 1.7)), "Cylinder")}
    labels = (r"Upstream reference", r"$s_{10}$", r"$s_{50}$", r"$s_{90}$", r"Wall-side / downstream")
    paths = []
    for gas in ("Ar", "N2"):
        for band, station_ids in (("shock", (1,2,3)), ("references", (0,4))):
            for variable in (("signed_vx",) if gas=="Ar" or band != "shock" else ("signed_vx", "rot_energy")):
                fig, axes = plt.subplots(1, len(station_ids), figsize=(6.95, 2.55), sharey=True,
                                         gridspec_kw={"wspace":.09})
                for ax, s in zip(axes, station_ids):
                    for g, (color, linestyle, legend) in style.items():
                        name = gas+"_"+g
                        stats, marginal = summaries[(name, "all", 10)]
                        if variable == "signed_vx":
                            edges, _, pdf, _, _ = marginal["incident"]
                            ax.stairs(pdf[s], edges, baseline=None, lw=1.25, color=color, linestyle=linestyle, label=legend)
                        else:
                            edges = marginal["normal"][0]
                            # An absent conditional value leaves a real gap.
                            y = stats["conditional_Erot_J_vn_bin"][s, 1:-1]
                            # Display a conditional curve only for bins observed in each registered half.
                            left = summaries[(name, "development_1_to_5", 10)][1]["normal"][3][s,1:-1]
                            right = summaries[(name, "held_out_6_to_10", 10)][1]["normal"][3][s,1:-1]
                            y = np.where((left>0) & (right>0), y, np.nan)
                            y = y/(.5*meta[name]["mass_kg"]*meta[name]["speed_scale_m_s"]**2)
                            ax.plot((edges[:-1]+edges[1:])/2, y, color=color, lw=1.12, linestyle=linestyle, label=legend)
                    velocity = r"$v_x/U_\infty$" if variable == "signed_vx" else r"$v_n/U_\infty$"
                    ax.set_xlabel(velocity+"\n"+labels[s], labelpad=2)
                    ax.set_xlim(edges[0], edges[-1])
                    ax.tick_params(top=True, right=True, direction="in", pad=2)
                ylab = r"$U_\infty f_x(v_x)$" if variable == "signed_vx" else r"$\langle E_{\rm rot}\mid v_n\rangle/(\frac{1}{2}mU_\infty^2)$"
                axes[0].set_ylabel(ylab)
                axes[0].set_ylim(bottom=0)
                handles, names = axes[0].get_legend_handles_labels()
                fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, handlelength=2.8,
                           handletextpad=.5, borderaxespad=.15, fontsize=9.3)
                fig.subplots_adjust(left=.096, bottom=.28, top=.82, right=.985)
                dest = path/f"{gas}_{band}_{variable}.pdf"
                fig.savefig(dest)
                plt.close(fig)
                paths.append(dest)
    return paths


def make_report(compact, output, stations_file):
    output = Path(output)
    registry_sha = file_sha(stations_file)
    registry = load_json(stations_file)
    meta, summaries, status = {}, {}, {}
    rows, stability, correlation, profile, delta, splits, phases, rare, joints = [], [], [], [], [], [], [], [], {}
    for case in CASES:
        m, blocks, state = case_blocks(compact, case)
        meta[case], status[case] = m, state
        for window, sl in WINDOWS.items():
            correlation.extend(series_rows(case, m, blocks, window, sl))
        for index, block in enumerate(blocks):
            correlation.extend(series_rows(case, m, blocks, f"block_{index+1:02d}", slice(index, index+1)))
        for stride in STRIDES:
            block_stats = []
            for index, b in enumerate(blocks):
                arrays, n = pooled_values([b], stride)
                stats, axes = describe(m, arrays, n)
                block_stats.append(stats)
                for s, station in enumerate(STATIONS):
                    common = dict(case=case, block=index+1, station=station, stride_us=stride, snapshots=n,
                                  raw_station_visits=int(arrays["raw"][s,0]), total_station_weight=float(arrays["weight"][s,0]))
                    for label, tally in (("lab_vx_lt0",1),("normal_lt0",2),("lab_vx_eq0",3),("normal_eq0",4)):
                        rare.append(dict(common, event=label, raw=int(arrays["raw"][s,tally]),
                                         raw_weight=float(arrays["weight"][s,tally]), probability=table_number(
                                         divide(arrays["weight"][s,tally],arrays["weight"][s,0])),
                                         kish_if_available=table_number(stats["kish_"+label][s]) if "kish_"+label in stats else None))
                    for axis, (_, probability, _, count, weight) in axes.items():
                        for side, i in (("underflow",0),("overflow",-1)):
                            rare.append(dict(common, event=axis+"_"+side,raw=int(count[s,i]), raw_weight=float(weight[s,i]),
                                             probability=table_number(probability[s,i]),kish_if_available=None))
                rows.extend(scalar_rows(stats, STATIONS, {"case":case, "window":f"block_{index+1:02d}",
                                        "stride_us":stride, "snapshots":n, "start_s_exclusive":.004+index*.0002,
                                        "end_s_inclusive":.004+(index+1)*.0002}))
            for index, station in enumerate(STATIONS):
                for metric in SCALARS:
                    if metric in block_stats[0]:
                        stability.append(block_variation(case, metric, block_stats, station, index, stride))
            for window, sl in WINDOWS.items():
                subset = blocks[sl]
                v, n = pooled_values(subset, stride)
                stats, axes = describe(m, v, n)
                summaries[case, window, stride] = (stats, axes)
                rows.extend(scalar_rows(stats, STATIONS, {"case":case, "window":window, "stride_us":stride,
                                        "snapshots":n, "start_s_exclusive":float(subset[0]["times_s"][0]-1e-6),
                                        "end_s_inclusive":float(subset[-1]["times_s"][-1])}))
                if stride == 10:
                    profile.extend(distribution_rows(case, window, m, stats, axes, subset))
                    count = np.sum([b["joint_raw"] for b in subset], axis=0)
                    weight = np.sum([b["joint_weight"] for b in subset], axis=0)
                    shape = (len(STATIONS), m["joint_normal_axis"][0]+2, m["joint_tangent_axis"][0]+2)
                    prefix = case+"_"+window
                    joints[prefix+"_raw"] = count.reshape(shape)
                    joints[prefix+"_weighted_probability"] = divide(weight,v["weight"][:,0,None]).reshape(shape)
                    joints[prefix+"_normal_edges_Uinf"] = np.linspace(*m["joint_normal_axis"][1:], m["joint_normal_axis"][0]+1)
                    joints[prefix+"_tangent_edges_Uinf"] = np.linspace(*m["joint_tangent_axis"][1:], m["joint_tangent_axis"][0]+1)
            dev, held = (summaries[case, w, stride] for w in ("development_1_to_5", "held_out_6_to_10"))
            for i, station in enumerate(STATIONS):
                result = dict(case=case, station=station, stride_us=stride,
                              binned_lab_vx_total_variation=distribution_distance(dev[1]["incident"][1][i], held[1]["incident"][1][i]))
                for key in SCALARS:
                    if key in dev[0]:
                        result["held_out_minus_development_"+key] = table_number(held[0][key][i]-dev[0][key][i])
                splits.append(result)
        for stride in STRIDES:
            for i, station in enumerate(STATIONS):
                a, ref = summaries[case, "all", stride], summaries[case, "all", 10]
                result = dict(case=case, station=station, stride_us=stride, reference_stride_us=10,
                              binned_lab_vx_total_variation=distribution_distance(a[1]["incident"][1][i], ref[1]["incident"][1][i]))
                for key in SCALARS:
                    if key in a[0]:
                        result["stride_minus_reference_"+key] = table_number(a[0][key][i]-ref[0][key][i])
                phases.append(result)
    require({m["geometry_sha256"] for m in meta.values()} == {registry_sha}, "Station registry or case geometry differs")
    station_density = []
    for row in rows:
        if row["stride_us"] != 10:
            continue
        condition = registry["cases"][row["case"]]
        station = next(st for st in condition["stations"] if st["id"] == row["station"])
        idx = STATIONS.index(row["station"])
        require(np.allclose(station["center_xyz_m"],meta[row["case"]]["station_centers_m"][idx],rtol=0,atol=1e-14),
                "Finite station does not match the registry")
        ratio = row["number_density_m3"] / condition["upstream"]["nrho"]
        before = station["local_profile"]["fields"]["rho_over_rho_inf"]
        threshold = station.get("threshold_rho_over_rho_inf")
        station_density.append({"case":row["case"],"station":row["station"],"window":row["window"],
                                "role":station["role"],"reference_coordinate_m":station["center_m"],
                                "rho_over_rho_inf_measured_patch":ratio,
                                "archived_profile_radial_interval_min":before["min"],
                                "archived_profile_radial_interval_mean":before["mean"],
                                "archived_profile_radial_interval_max":before["max"],
                                "archived_density_level":threshold,
                                "finite_patch_minus_density_level":ratio-threshold if threshold is not None else None})
    for gas in ("Ar", "N2"):
        cp, pp = gas+"_cylinder", gas+"_planar"
        require(meta[cp]["marginals"] == meta[pp]["marginals"], "Unmatched dimensionless bins")
        for window in WINDOWS:
            for stride in STRIDES:
                p, c = summaries[pp,window,stride], summaries[cp,window,stride]
                for i, station in enumerate(STATIONS):
                    result = dict(gas=gas, station=station, window=window, stride_us=stride,
                                  binned_lab_vx_total_variation=distribution_distance(c[1]["incident"][1][i], p[1]["incident"][1][i]))
                    for key in SCALARS:
                        if key in c[0]:
                            result["cylinder_minus_planar_"+key] = table_number(c[0][key][i]-p[0][key][i])
                    delta.append(result)
    output.mkdir(parents=True, exist_ok=True)
    data = {"station_statistics.csv":rows, "measured_station_density.csv":station_density, "finite_block_variation.csv":stability, "rare_event_support.csv":rare,
            "correlation_at_lag.csv":correlation, "marginal_distributions.csv.gz":profile,
            "matched_geometry.csv":delta, "held_out.csv":splits, "thinning_sensitivity.csv":phases}
    written = []
    for name, records in data.items():
        dest = output/name
        write_csv(dest, records)
        written.append(dest)
    joint_path = output/"joint_velocity.npz"
    np.savez_compressed(joint_path, station_names=np.asarray(STATIONS), **joints)
    # Keep the font cache local to this report while rendering.
    with tempfile.TemporaryDirectory(prefix="vdf_fonts_", dir=output) as temp:
        original = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = temp
        try:
            written += [joint_path] + write_figures(output, summaries, meta)
        finally:
            if original is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = original
    report = {"format":FORMAT, "data":"Measured physical particle distributions, four target-verified continuations",
              "source_case_status_sha256":{case:file_sha(Path(compact)/case/"case_status.json") for case in CASES},
              "science_and_final_receipts":{case:{k:v[k] for k in ("chain_sha256","target_receipt_sha256","science")}
                                            for case,v in status.items()}, "metadata":meta, "station_registry_sha256":registry_sha,
              "analysis_sha256":{f:file_sha(Path(__file__).with_name(f)) for f in (
                  "vdf_analysis.py","vdf_statistics.py","vdf_report.py","vdf_reduce.py")},
              "outputs":{p.name:{"bytes":p.stat().st_size,"sha256":file_sha(p)} for p in written}}
    save_json(output/"REPORT.json",report)
    return {"complete_cases":list(CASES), "snapshots_per_case":2000, "files":len(written)+1}

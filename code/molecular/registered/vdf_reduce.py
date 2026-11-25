"""Reduce only published V2 blocks named in the science branch's accepted chain."""
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np

from vdf_statistics import (CASES, FORMAT, STATIONS, STRIDES, almost, describe, divide, file_sha,
                            load_json, require, save_json)


def native_reader(path, pinned=None):
    path = Path(path).resolve(strict=True)
    if pinned:
        require(file_sha(path) == pinned, "Native reader does not match its science SHA256")
    spec = importlib.util.spec_from_file_location("jfm_native_vdf_reader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis_signature():
    here = Path(__file__).resolve()
    return {"reducer_sha256": file_sha(here), "statistics_sha256": file_sha(here.with_name("vdf_statistics.py"))}


def contract_meta(c):
    require(c["ne"] == c["extra"]+23, "Exact squared weights and transport moments require native V2")
    require(np.array_equal(c["incident"], [1, 0, 0]) and np.array_equal(c["frame"], [0, 0, 0]),
            "The primary observable is defined in the stationary laboratory frame with +x incident")
    axes = []
    for name, offset, axis in zip(("incident", "normal", "tangent"), c["offset"], (0, 0, 1)):
        n, lo, hi = c["axes"][axis]
        axes.append([name, offset, n, lo, hi])
    return {"format": FORMAT, "case": c["name"], "rotation": c["rotation"],
            "mass_kg": c["mass"], "speed_scale_m_s": c["scale"],
            "station_names": [s["id"] for s in c["stations"]],
            "volume_m3": [s["volume"] for s in c["stations"]],
            "station_centers_m": [s["center"].tolist() for s in c["stations"]],
            "normal_xyz": [s["normal"].tolist() for s in c["stations"]],
            "dimension": c["dimension"], "geometry_sha256": c["stations_sha256"],
            "manifest_sha256": c["manifest_sha256"], "marginals": axes,
            "joint_normal_axis": c["axes"][2], "joint_tangent_axis": c["axes"][3]}


def validate_against_full(meta, values, n, original):
    stats, axes = describe(meta, values, n)
    require(np.array_equal(values["raw"], original["raw"][:, :values["raw"].shape[1]]), "Full raw marginal counts differ")
    almost(values["weight"], original["physical_number_sum"][:, :values["raw"].shape[1]])
    original_moments = original["moments_SI"]
    nnormal = meta["marginals"][1][2]+2
    extra = 12+nnormal*(1+int(meta["rotation"]))
    almost(values["mom"], np.concatenate([original_moments[:, :12], original_moments[:, extra:]], axis=1))
    almost(values["proj_bin"], original_moments[:, 12:12+nnormal])
    if meta["rotation"]:
        almost(values["rot_bin"], original_moments[:, 12+nnormal:extra])
    for stat, key in (("kish_particle_observations", "kish_particle_observations"),
                      ("p_lab_vx_lt0", "backward_lab_x_fraction"), ("p_normal_lt0", "backward_normal_fraction"),
                      ("number_density_m3", "sample_mean_number_density_m3")):
        almost(stats[stat], original[key])
    for label, data in axes.items():
        almost(data[1], original[label+"_all_bin_probabilities_with_tails"])
        almost(data[2], original[label+"_interior_pdf_s_per_m"]*meta["speed_scale_m_s"])
    for prefix, key in (("q_trans", "translational_heat_flux_W_m2_xyz"), ("q_rot", "rotational_heat_flux_W_m2_xyz")):
        if prefix == "q_rot" and not meta["rotation"]:
            continue
        almost(np.stack([stats[prefix+"_"+c+"_W_m2"] for c in "xyz"], axis=-1), original[key])
    if meta["rotation"]:
        almost(stats["conditional_Erot_J_vn_bin"], original["rot_energy_J_conditional_normal_bin"])


def reduce_block(reader, c, sample_dir, full_path, stride_values, *, run_origin, production, expected_ranks=None):
    """Read each microrecord once and check its hash against its aggregate's ordered list."""
    meta = contract_meta(c)
    end_marg = c["offset"][-1]
    nstations, nnormal = len(c["stations"]), c["axes"][0][0]+2
    strides = np.asarray(stride_values, dtype=int)
    require(np.all(strides > 0) and strides[0] == 1, "Invalid thinning factors")
    with np.load(full_path, allow_pickle=False) as original:
        require(str(original["source_geometry_sha256"]) == c["stations_sha256"]
                and str(original["source_run_sha256"]) == c["run_meta_sha256"]
                and str(original["source_manifest_sha256"]) == c["manifest_sha256"],
                "Run or station provenance differs from validated block")
        steps = original["steps"].astype(np.int64)
        t = original["time_s"]
        hashes = original["source_sample_sha256"].tolist()
        require(len(steps) == c["block"] and len(set(hashes)) == len(steps) and
                np.array_equal(np.diff(steps), np.full(len(steps)-1, c["every"])),
                "Non-contiguous validated steps or duplicate sample SHA256")
        require(np.all((steps - run_origin) % c["every"] == 0), "Global thinning phase is undefined")
        shape = (len(strides), nstations)
        sums = {"raw": np.zeros((*shape, end_marg), dtype=np.uint64),
                "weight": np.zeros((*shape, end_marg)), "mom": np.zeros((*shape, 35)),
                "proj_bin": np.zeros((*shape, nnormal)),
                "rot_bin": np.zeros((*shape, nnormal)) if c["rotation"] else np.empty((*shape, 0))}
        num_sample = np.zeros(len(strides), dtype=np.int64)
        time_series = np.full((len(steps), nstations, 5), np.nan)
        counts = np.zeros((len(steps), nstations), dtype=np.uint64)
        contributors = np.empty((len(steps), 3), dtype=np.int32)
        joint_raw = np.zeros((nstations, c["nhist"]-end_marg), dtype=np.uint64)
        joint_weight = np.zeros(joint_raw.shape)
        full_joint_raw = np.zeros_like(joint_raw)
        full_joint_weight = np.zeros_like(joint_weight)
        joint_stride = 10 if production else int(strides[-1])
        for i, step in enumerate(steps):
            sample = reader.read_sample(sample_dir / ("sample_%015d.vdf" % step), c)
            require(sample["sha256"] == hashes[i], "Native observation hash differs from committed block")
            require(sample["step"] == step and math.isclose(sample["time"], t[i], rel_tol=0, abs_tol=c["dt"]*.002)
                    and math.isclose(t[i], step*c["dt"], rel_tol=0, abs_tol=c["dt"]*.002), "Observation clock differs")
            contributors[i] = sample["ranks"], sample["eligible"], sample["listed"]
            if expected_ranks is not None:
                require(sample["ranks"] == expected_ranks, "Native MPI size differs from admitted run")
            w, raw, moments = sample["weight"], sample["raw"], sample["moments"]
            effective = np.concatenate([moments[:, :12], moments[:, c["extra"]:]], axis=1)
            counts[i] = raw[:, 0]
            time_series[i] = np.stack([w[:, 0], divide(w[:, 1], w[:, 0]), divide(w[:, 2], w[:, 0]),
                                       divide(effective[:, 19], w[:, 0]),
                                       divide(moments[:, 5], w[:, 0]) if c["rotation"] else np.full(nstations, np.nan)], axis=-1)
            full_joint_raw += raw[:, end_marg:]
            full_joint_weight += w[:, end_marg:]
            ordinal = (int(step)-run_origin)//c["every"]
            for j in np.flatnonzero(ordinal % strides == 0):
                sums["raw"][j] += raw[:, :end_marg]
                sums["weight"][j] += w[:, :end_marg]
                sums["mom"][j] += effective
                sums["proj_bin"][j] += moments[:, 12:12+nnormal]
                if c["rotation"]:
                    sums["rot_bin"][j] += moments[:, 12+nnormal:c["extra"]]
                num_sample[j] += 1
            if ordinal % joint_stride == 0:
                joint_raw += raw[:, end_marg:]
                joint_weight += w[:, end_marg:]
        expected = c["block"]//strides
        require(np.all(c["block"] % strides == 0) and np.array_equal(num_sample, expected), "Incomplete thinning phases")
        require(np.array_equal(full_joint_raw, original["raw"][:, end_marg:]), "Complete joint tally differs from native block")
        almost(full_joint_weight, original["physical_number_sum"][:, end_marg:])
        validate_against_full(meta, {k: a[0] for k, a in sums.items()}, len(steps), original)
        js = list(strides).index(joint_stride)
        require(np.array_equal(joint_raw.sum(axis=1), sums["raw"][js, :, 0]), "Selected joint loses raw observations")
        almost(joint_weight.sum(axis=1), sums["weight"][js, :, 0])
    return dict(sums, strides=strides, snapshots=num_sample, steps=steps, times_s=t,
                sample_sha256=np.asarray(hashes), native_ranks_eligible_listed=contributors,
                snapshot_counts=counts, snapshot_series=time_series, joint_stride=joint_stride,
                joint_raw=joint_raw, joint_weight=joint_weight, metadata_json=json.dumps(meta, sort_keys=True))


def case_reduce(campaign, name, output):
    root, destination = Path(campaign).resolve(strict=True), Path(output).resolve()
    require(name in CASES, "Unsupported production case")
    branch = root/"cases"/name
    pin, ledger, target = branch/"SCIENCE.json", branch/"CHAIN.json", branch/"TARGET_COMPLETE.json"
    science = load_json(pin)
    chain_sha = file_sha(ledger)
    chain = load_json(ledger)
    require(science["case"] == name and science["record_protocol"] == 2 and chain, "No admissible committed V2 chain")
    reader = native_reader(root/"control"/"vdf_records.py", science["reader_sha256"])
    plan_path = root/"control"/"sampling_plan.json"
    plan = load_json(plan_path)
    require(file_sha(plan_path) == science["sampling_plan_sha256"] and plan["reference_analysis_stride_s"] == 1.e-5
            and plan["development_blocks"] == list(range(1,6)) and plan["held_out_blocks"] == list(range(6,11)),
            "The registered sampling plan differs")
    micro = plan["native_snapshot_cadence_s"]
    require(math.isclose(micro, 1e-6) and tuple(round(v/micro) for v in plan["sensitivity_analysis_stride_s"]) == STRIDES,
            "Cadence sensitivity schedule differs")
    path = destination/name
    path.mkdir(parents=True, exist_ok=True)
    next_step, parent_sha = science["step0"], science["parent_sha256"]
    every, nblock = science["every_steps"], science["samples_per_block"]
    require(nblock == 200 and math.isclose(science["dt_s"]*every, micro), "Native cadence differs")
    receipts, signature = [], analysis_signature()
    require(len(chain) <= len(science["checkpoint_blocks"]), "Unexpected number of committed anchors")
    for i, entry in enumerate(chain):
        upto = science["step0"] + science["checkpoint_blocks"][i]*nblock*every
        require(entry["start_step"] == next_step and entry["step"] == upto and entry["parent_sha256"] == parent_sha,
                "Discontinuous native checkpoint chain")
        validation = entry["validation"]
        require(validation["verified_samples"] == (upto-next_step)//every and validation["first"] == next_step+every
                and validation["last"] == upto, "Wrong native receipt interval")
        attempt_root = (branch/"attempts").resolve(strict=True)
        attempt = (attempt_root / entry["attempt"]).resolve(strict=True)
        require(attempt.parent == attempt_root, "Incorrect named attempt")
        samples = attempt/"samples"
        c = reader.read_contract(samples)
        require(c["name"] == name and c["manifest_sha256"] == science["stations_sha256"]
                and c["block"] == nblock and c["every"] == every and c["dt"] == science["dt_s"]
                and tuple(s["id"] for s in c["stations"]) == STATIONS, "Station/time contract differs")
        for ref in validation["blocks"]:
            block = Path(ref).resolve(strict=True)
            require((attempt/"validated").resolve() in block.parents and re.fullmatch(r"block_\d{15}_\d{15}\.npz", block.name),
                    "Invalid native block path")
            first, last = [int(n) for n in re.findall(r"\d+", block.stem)]
            require(first == next_step+every and last == first+every*(nblock-1), "Unexpected or repeated block range")
            b = (last-science["step0"])//(nblock*every)
            sha = validation["block_sha256"][block.name]
            require(file_sha(block) == sha, "Native validated block hash differs")
            dst = path/("block_%02d.npz" % b)
            receipt = dst.with_suffix(".json")
            info = dict(case=name, block=b, first_step=first, last_step=last, native_npz_sha256=sha,
                        qualified_attempt=entry["attempt"], committed_anchor=upto, science_sha256=file_sha(pin),
                        generation=signature)
            if receipt.is_file():
                old = load_json(receipt)
                require(old["source"] == info and file_sha(dst) == old["sha256"], "Existing compact receipt/files differ")
            else:
                arrays = reduce_block(reader, c, samples, block, STRIDES, run_origin=science["step0"],
                                      production=True, expected_ranks=science["ranks"])
                reader.publish_npz(dst, dict(arrays, source_json=json.dumps(info, sort_keys=True)))
                old = {"source": info, "sha256": file_sha(dst), "file": dst.name, "bytes": dst.stat().st_size}
                save_json(receipt, old)
            receipts.append(old)
            next_step = last
        require(next_step == upto, "The committed anchor has missing aggregate blocks")
        parent_sha = entry["checkpoint"]["sha256"]
    require(chain_sha == file_sha(ledger), "The run advanced during reduction; rerun to refresh the ledger view")
    final = None
    if target.is_file():
        t = load_json(target)
        require(t["case"] == name and t["chain_sha256"] == chain_sha and t["science_sha256"] == file_sha(pin)
                and t["blocks"] == 10 and t["samples"] == 2000 and t["native_final_readback"]["step"] == next_step
                and len(receipts) == 10 and next_step == science["end_step"], "Target completion receipt disagrees")
        final = file_sha(target)
    status = dict(format=FORMAT, case=name, chain_sha256=chain_sha, target_receipt_sha256=final, science=science,
                  sample_period_s=micro, reference_stride_us=10, end_step=next_step, blocks=receipts)
    save_json(path/"case_status.json", status)
    return {"case": name, "complete_blocks": len(receipts), "target_verified": bool(final),
            "bytes_to_stage": sum(r["bytes"] for r in receipts)}

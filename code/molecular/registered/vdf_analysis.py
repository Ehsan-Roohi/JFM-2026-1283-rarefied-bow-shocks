"""Read committed molecular diagnostics, audit a native fixture, or report four completed cases."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from vdf_reduce import analysis_signature, case_reduce, native_reader, reduce_block
from vdf_statistics import CASES, almost, describe, file_sha, load_json, require, save_json


def oracle_particles(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(("ITEM: ATOMS", "ITEM: PARTICLES")))
    keys = lines[start].split()[2:]
    rows = [dict(zip(keys, line.split())) for line in lines[start+1:] if line.strip()]
    require(len(rows) == 6 and len({r["id"] for r in rows}) == 6, "Native gate is not the identifiable six-particle fixture")
    return sorted(rows, key=lambda r:int(r["id"]))


def admitted_fixture(root, geometry, kind, expected_hash):
    choices = []
    for path in (root/f"{geometry}_4_{kind}").glob("attempt_*"):
        pin = path/"RUN_OK.json"
        if pin.is_file():
            r = load_json(pin)
            if r["sha256"] == expected_hash and r["ranks"] == 4 and file_sha(path/"in.fixture") == r["deck_sha"]:
                choices.append(path)
    require(len(choices) == 1, f"Expected exactly one qualified {geometry} four-rank {kind} fixture")
    return choices[0]


def fixture(gate, decoder, receipt):
    root, decoder = Path(gate).resolve(strict=True), Path(decoder).resolve(strict=True)
    validation = root/"fixture_report.json"
    gate_report = load_json(validation)
    require(gate_report["passed"] is True and set(gate_report["cases"]) == {"planar","cylinder"},
            "Native V2 gate must already have completed its independent original/instrumented checks")
    reader = native_reader(decoder)
    result = {"evidence":"Only the named native six-particle MPI gate. No production observations or figures.",
              "fixture_gate_sha256":file_sha(validation), "reader_sha256":file_sha(decoder),
              "analysis":dict(analysis_signature(), fixture_driver_sha256=file_sha(Path(__file__)),
                              report_helpers_sha256=file_sha(Path(__file__).with_name("vdf_report.py"))),
              "native_geometry_builds":{}}
    for geom in ("planar", "cylinder"):
        gate_case = gate_report["cases"][geom]
        measured = admitted_fixture(root,geom,"full",gate_case["instrumented_sha256"])
        baseline = admitted_fixture(root,geom,"base",gate_case["original_sha256"])
        q = oracle_particles(baseline/"oracle.4")
        require(q == oracle_particles(measured/"oracle.4"), "Native independent final microstates differ")
        observed = [r for r in q if float(r["x"]) < 2]
        ww = np.array([1 if float(r["x"]) < 1 else 4 for r in observed],float)
        vv = np.array([[float(r[i]) for i in ("vx","vy","vz")] for r in observed])
        er = np.array([float(r["erot"]) for r in observed])
        c = reader.read_contract(measured/"data")
        require(c["name"]=="fixture" and c["block"]==2 and c["stations"][1]["id"]=="empty",
                "Unexpected native test dimensions")
        final_interval = measured/"blocks"/"block_000000000000003_000000000000004.npz"
        compact = reduce_block(reader,c,measured/"data",final_interval,(1,2),run_origin=0,production=False,expected_ranks=4)
        require(compact["snapshots"].tolist()==[2,1] and compact["times_s"].shape==(2,), "Fixture thinning failed")
        meta = json.loads(compact["metadata_json"])
        # Exercise the transfer format and numerical table path on the measured native fixture.
        # All temporary products here are test-only and are removed on leaving the context.
        from vdf_report import distribution_distance, pair_correlation, write_csv
        from vdf_statistics import scalar_rows
        cache_root = Path(receipt).resolve().parent if receipt else Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix="native_vdf_gate_", dir=cache_root) as cache:
            export = Path(cache)/"measured_fixture_compact.npz"
            np.savez_compressed(export,**compact)
            with np.load(export,allow_pickle=False) as reopened:
                require(np.array_equal(reopened["sample_sha256"],compact["sample_sha256"]), "Fixture provenance changed on transfer")
                source = {k:reopened[k][1] for k in ("raw","weight","mom","proj_bin","rot_bin")}
                full_sums = {k:reopened[k][0] for k in source}
                full_stats, full_axes = describe(meta,full_sums,2)
                single_stats, one_axes = describe(meta,source,1)
                candidates = list(scalar_rows(single_stats,meta["station_names"],{"dataset":"native fixture","snapshot_count":1}))
                table = Path(cache)/"station_observations.csv"
                write_csv(table,candidates)
                require(table.is_file() and candidates[0]["raw_observations"]==5 and
                        candidates[1]["p_lab_vx_lt0"] is None and full_stats["raw_observations"][0]==10,
                        "Measured fixture table did not preserve empty support or thinning")
                require(distribution_distance(full_axes["incident"][1][1],one_axes["incident"][1][1]) is None,
                        "A distance was assigned to a wholly unsupported station")
                require(distribution_distance(full_axes["incident"][1][0],one_axes["incident"][1][0])<1.e-12,
                        "Identical static populations disagree after thinning")
                valid_pairs, coef = pair_correlation(reopened["snapshot_series"][:,0,1],1)
                require(valid_pairs==1 and coef is None,"Insufficient fixture lag pairs produced a correlation")
        stats, axes = describe(meta,source,1)
        require(int(stats["raw_observations"][0]) == len(ww) and stats["raw_observations"][1] == 0,
                "Fixture occupancy differs from independently dumped particles")
        almost(stats["sum_w"][0],ww.sum())
        almost(stats["sum_w2"][0],np.square(ww).sum())
        almost(stats["kish_particle_observations"][0],ww.sum()**2/np.square(ww).sum())
        for station_frame, v in (("lab_vx",vv[:,0]),("normal",vv[:,1])):
            almost(stats["p_"+station_frame+"_lt0"][0],ww[v<0].sum()/ww.sum())
            almost(stats["kish_"+station_frame+"_lt0"][0],ww[v<0].sum()**2/np.square(ww[v<0]).sum())
        require(np.isnan(stats["p_lab_vx_lt0"][1]) and np.isnan(stats["Trot_K"][1]), "Empty station has a statistic")
        for label, projected in (("incident",vv[:,0]),("normal",vv[:,1]),("tangent",-vv[:,0])):
            edges, probabilities, *_ = axes[label]
            direct = np.bincount(np.searchsorted(edges,projected,side="right"),weights=ww,minlength=len(edges)+1)
            almost(probabilities[0],direct/ww.sum())
        u = (vv*ww[:,None]).sum(axis=0)/ww.sum()
        centered = vv-u
        volume = meta["volume_m3"][0]
        qtr = .5*c["mass"]*((ww*(centered**2).sum(axis=1))[:,None]*centered).sum(axis=0)/volume
        qr = ((ww*er)[:,None]*centered).sum(axis=0)/volume
        # Central-vector cancellation can leave round-off at 1e-44 W/m2 in this deliberately tiny fixture.
        np.testing.assert_allclose([stats["q_trans_"+v+"_W_m2"][0] for v in "xyz"],qtr,rtol=4e-10,atol=1e-44)
        np.testing.assert_allclose([stats["q_rot_"+v+"_W_m2"][0] for v in "xyz"],qr,rtol=4e-10,atol=1e-40)
        almost(stats["mean_Erot_J_lab_vx_lt0"][0],np.average(er[vv[:,0]<0],weights=ww[vv[:,0]<0]))
        require(np.array_equal(compact["joint_raw"].sum(axis=1),source["raw"][:,0]), "Fixture joint mass mismatch")
        result["native_geometry_builds"][geom] = {"baseline_sha256":gate_case["original_sha256"],
               "instrumented_sha256":gate_case["instrumented_sha256"], "observed_block_sha256":file_sha(final_interval),
               "independent_native_oracle_sha256":file_sha(baseline/"oracle.4"),
               "time_steps":compact["steps"].tolist(), "full_and_thinned_snapshot_counts":compact["snapshots"].tolist(),
               "compact_npz_reread_and_statistic_csv_pass":True,
               "physical_weight":float(stats["sum_w"][0]), "squared_weight_sum":float(stats["sum_w2"][0]),
               "Kish_observation_count":float(stats["kish_particle_observations"][0]),
               "p_lab_vx_lt0":float(stats["p_lab_vx_lt0"][0]), "p_normal_lt0":float(stats["p_normal_lt0"][0]),
               "native_conditions_compared":"unequal cell weights, two outer velocity tails, exact zeros, both flow axes, empty station, conditional rotational energy, joint normalization, direct heat flux"}
    result["passed"] = True
    if receipt:
        save_json(receipt,result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("reduce", help="Reduce the immutable microrecords of officially committed blocks")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--case", choices=CASES, required=True)
    r.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("inspect", help="Summarise the accepted contiguous prefix of a single case, no figures")
    r.add_argument("--compact", type=Path, required=True)
    r.add_argument("--case", choices=CASES, required=True)
    r.add_argument("--stride-us", type=int, choices=(1,5,10,20,40), default=10)
    r = sub.add_parser("report", help="Verify compact transfer, then compare four complete ten-block cases")
    r.add_argument("--compact", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--stations", type=Path, default=Path(__file__).resolve().parent.parent/"stations"/"stations.json",
                   help="Original station registry, checked against the geometry hash of all four cases")
    r = sub.add_parser("fixture", help="Audit an already executed native six-particle gate; writes no figures")
    r.add_argument("--gate", type=Path, required=True)
    r.add_argument("--decoder", type=Path, required=True)
    r.add_argument("--receipt", type=Path)
    a = p.parse_args()
    if a.action == "reduce":
        result = case_reduce(a.campaign,a.case,a.out)
    elif a.action == "inspect":
        from vdf_report import inspect_case
        result = inspect_case(a.compact, a.case, a.stride_us)
    elif a.action == "report":
        from vdf_report import make_report
        result = make_report(a.compact,a.out,a.stations)
    else:
        result = fixture(a.gate,a.decoder,a.receipt)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

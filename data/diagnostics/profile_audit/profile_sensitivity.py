"""Audit exported-ray root counts and sensitivity without reading the full fields again."""
from pathlib import Path
import csv
import importlib.util
import json
import hashlib
import sys

import numpy as np

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("portable", root / "density_crossings_170.py")
portable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable)
archive = np.load(root / "existing_density_rays.npz")
meta = json.loads(str(archive["meta"]))
records = []
for case in meta["cases"]:
    if case["bank"] != "Kn":
        continue
    key = case["prefix"]
    base = archive[key + "_s"] * 1000
    ratio = archive[key + "_rho"] / case["rho_ref"]
    gamma = 5/3 if case["gas"] == "Ar" else 7/5
    rh = (gamma+1)*100 / ((gamma-1)*100+2)
    for stride in (8,4,2,1):
        ss, q = base[::stride], ratio[::stride]
        valid = np.flatnonzero(np.isfinite(q))
        islands = np.split(valid, np.flatnonzero(np.diff(valid)>1)+1)
        ind = islands[-1]
        samples = list(zip(ss[ind],q[ind]))
        ds = float((ss[1]-ss[0]))
        for pct in (10,50,90):
            val = 1+pct/100*(rh-1)
            positive, negative = portable.roots(samples,val)
            assert q[ind[-1]] < val and positive
            loc = positive[-1]
            assert loc > ss[ind[0]] and loc < ss[ind[-1]]
            records.append(dict(gas=case["gas"],Kn=case["kn"],grid_points=len(ss),kernel="unsmoothed",
                                spacing_mm=ds,level_pct=pct,s_mm=loc,inward_count=len(positive),outward_count=len(negative),
                                inward_roots_mm="|".join(f"{z:.6f}" for z in reversed(positive)),
                                outward_roots_mm="|".join(f"{z:.6f}" for z in reversed(negative))))
with (root / "raw_ray_sensitivity.csv").open("w", newline="") as stream:
    wr=csv.DictWriter(stream,records[0],lineterminator="\n");wr.writeheader();wr.writerows(records)
lookup={(r["gas"],r["Kn"],r["grid_points"],r["level_pct"]):r for r in records}
def cmp(grid):
    subset=[(abs(r["s_mm"]-lookup[r["gas"], r["Kn"], 1601,r["level_pct"]]["s_mm"]),r["gas"],r["Kn"],r["level_pct"])
            for r in records if r["grid_points"]==grid]
    return max(subset)
def getcsv(name):
    with (root / name).open(newline="") as f:
        return list(csv.DictReader(f))
cross=getcsv("fixed_RH_crossings.csv")
raw=[r for r in cross if float(r["factor"])==0]
all_norm=getcsv("density_crossings_170.csv")
for percent in (10,50,90):
    assert all(int(v[f"s{percent}_inward_count"])>=1 for v in all_norm)
assert all(r["crossing_order_pass"]=="True" and r["upstream_plateau_pass"]=="True" and r["finite_components"]=="1" for r in cross)
all_at_1=[r for r in cross if float(r["factor"])==1]
lowfac={}
for f in (0,.5,1,2):
    for g in ("Ar","N2"):
        rows=sorted([r for r in cross if float(r["factor"])==f and r["gas"]==g],key=lambda r:float(r["Kn"]))
        assert all(float(a["s10_mm"])<float(b["s10_mm"]) for a,b in zip(rows, rows[1:]))
        a,b=rows[:2]
        lowfac[f"{g}_{f:g}"]={k:float(b[k])-float(a[k]) for k in ["s10_mm","s50_mm","s90_mm","W_mm"]}
def spread(table, column):
    return {g:[min(float(r[column]) for r in table if r["gas"]==g),
               max(float(r[column]) for r in table if r["gas"]==g)] for g in ("Ar","N2")}
def argmax_gap(table_a,table_b,key):
    aa={(r["gas"],r["Kn"]):r for r in table_a};bb={(r["gas"],r["Kn"]):r for r in table_b}
    return max((abs(float(a[key])-float(bb[(g,k)][key])),g,k) for (g,k),a in aa.items())
summary={"kernel":"unsmoothed piecewise-linear DS2V density-ray samples; all distance units mm",
    "bank_reference":"imposed m*n from the Kn input deck",
    "n_cases":len(raw),"gas_case_counts":{g:sum(v["gas"]==g for v in raw) for g in ('Ar','N2')},
    "all_raw_and_all_smoothed_have_finite_ordered_brackets_on_one_component_and_pass_tail_test":True,
    "raw_upstream_tail_median_range":spread(raw,"tail_median"), "raw_upstream_tail_relative_sd_range":spread(raw,"tail_rel_sd"),
    "closest_valid_extent_mm":{"outboard_beyond_first_s10":min(float(r["outermost_finite_s_max_mm"])-float(r["s10_mm"]) for r in raw),
                               "inboard_below_first_s90":min(float(r["s90_mm"])-float(r["outermost_finite_s_min_mm"]) for r in raw)},
    "raw_from_0p01_to_0p025_at_each_smoothing_factor":lowfac,
    "s10_monotonically_outward_in_both_gases_at_all_four_smoothing_factors":True,
    "max_abs_raw_vs_primary_smoothed_position_difference_mm_gas_Kn":{s:argmax_gap(raw,all_at_1,s) for s in ["s10_mm","s50_mm","s90_mm","W_mm"]},
    "max_abs_raw_sampling_grid_position_difference_vs_1601_mm_gas_Kn_level":{str(grid):cmp(grid) for grid in (801,3201,6401)},
    "multiple_raw_inward_roots_by_ray_grid":[{k:v for k,v in r.items() if k!="kernel"} for r in records if r["inward_count"]>1],
    "cautions":["Interpolated ray spacing differs from independent DSMC export or collision grid resolution.",
                "The 10,50,90% levels are fractions of an imposed ideal normal-shock density jump, not a measured local downstream state.",
                "A bracketed overlap with these levels does not establish a separable front, recovered limiting RH state, or converged solution."]}
(root/"PROFILE_AUDIT_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
for key in ["closest_valid_extent_mm","max_abs_raw_vs_primary_smoothed_position_difference_mm_gas_Kn","max_abs_raw_sampling_grid_position_difference_vs_1601_mm_gas_Kn_level","multiple_raw_inward_roots_by_ray_grid"]:print(key,summary[key])

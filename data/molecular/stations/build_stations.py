"""Build molecular-sampling stations on existing ten-block Mach-10 SPARTA mean profiles."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


WORKSPACE = "campaign"
R = 0.1524
THETA = math.radians(170.0)
HALF_NORMAL = 0.0004
HALF_TANGENT = 0.002
BLOCK_S = 0.0002
GAS = {
    "Ar": {"gamma": 5/3, "nrho": 3.856e20, "mass": 6.63e-26, "u": 2634.1},
    "N2": {"gamma": 1.4, "nrho": 3.941e20, "mass": 4.65e-26, "u": 2883.3},
}
# Fingerprints are of the small verified profile arrays, not of a reconstructed VDF.
INPUTS = {
    "Ar_planar": ("dcfce1cde01d3aa6e63964b107bf5ed2d4b49ee7bf2e9c53992ff5a65ebffd22",
                  "analysis_planar_v22_20261031/Ar_planar_long.npz"),
    "N2_planar": ("f6b884ce2b99da3d15e6325321e65949f9486abc6f865fa969fe28590b798dfd",
                  "analysis_planar_v22_20261031/N2_planar_long.npz"),
    "Ar_cylinder": ("6b8e782157f0fe5d0663f3b527ac41d04bc2118d51bb3e05a5338ee2aa50035c",
                    "convergence_controls_v20/result_C4_Ar_L4_20261027T224552Z/C4_Ar_L4_ray170.npz"),
    "N2_cylinder": ("419ca393abf9bc09c8f8c8b11abe05bfa29007f8e240aeb51e01df4ef21edb45",
                    "convergence_controls_v20/result_C4_N2_L4_20261030T071906Z/C4_N2_L4_ray170.npz"),
}
EXPECTED_ROOTS_MM = {
    "Ar_planar": [-11.46314229, 3.59223296, 15.77422780],
    "N2_planar": [-7.40938804, 0.88793780, 7.73934829],
    "Ar_cylinder": [109.29534941, 94.78875478, 83.32901361],
    "N2_cylinder": [68.02053544, 60.01769402, 53.38221028],
}


def profile(case_id, input_dir):
    gas, geometry = case_id.split("_")
    expected, rel = INPUTS[case_id]
    name = Path(rel).name
    candidates = [input_dir / (expected[:12] + "_" + name), input_dir / name, input_dir / rel]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing {name} in {input_dir}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Unexpected input hash for {case_id}")
    gas_data = GAS[gas]
    dt = 2.e-8 if geometry == "planar" else 5.e-9
    with np.load(path, allow_pickle=False) as z:
        if geometry == "planar":
            coord = z["x"]
            means = z["mean"]
            blocks = z["B"]
            columns = list(z["variables"].astype(str))
            rho = means[:, columns.index("rho")] / (gas_data["nrho"] * gas_data["mass"])
            temperature = means[:, columns.index("Ttr")]
            velocity = means[:, columns.index("u")]
            trot = means[:, columns.index("Trot")] if gas == "N2" else None
            tau = z["tau"][None, :]
            if not np.allclose(z["times"], np.arange(11, 21) * BLOCK_S):
                raise ValueError("Unexpected planar block times")
        else:
            coord = z["s"]
            means = z["mean"]
            blocks = z["blocks"]
            rho, temperature, velocity = means[:, 0], means[:, 1], means[:, 2]
            trot = means[:, 3] if gas == "N2" else None
            ratios = blocks[:, :, 5]  # documented C4 dt/tau from each block's sampled grid
            tau = np.divide(dt, ratios, out=np.full(ratios.shape, np.nan), where=ratios > 0)
        keep = np.flatnonzero(np.isfinite(rho))
        if np.any(np.diff(keep) != 1) or np.any(np.diff(coord) <= 0) or blocks.shape[0] != 10:
            raise ValueError(f"Unexpected coordinate support or block count for {case_id}")
        if not np.allclose(means[keep], blocks[:, keep].mean(axis=0), rtol=1.e-9, atol=1.e-10, equal_nan=True):
            raise ValueError(f"Stored mean differs from the ten blocks for {case_id}")
        data = {"rho_over_rho_inf": rho[keep], "Ttr_K": temperature[keep], "ux_m_s": velocity[keep]}
        if trot is not None:
            data["Trot_K"] = trot[keep]
        return coord[keep], data, tau[:, keep], dt, {"workspace_path": WORKSPACE + "/" + rel,
               "sha256": digest, "filename": name, "ten_block_mean_verified": True}


def crossing(coord, density, threshold, geometry):
    # Walk from upstream towards downstream in both geometries.
    x, y = (coord, density) if geometry == "planar" else (coord[::-1], density[::-1])
    hits = np.flatnonzero((y[:-1] < threshold) & (y[1:] >= threshold))
    if not len(hits):
        raise ValueError("Freestream-to-downstream crossing is absent")
    i = hits[0]
    pos = x[i] + (threshold-y[i]) * (x[i+1]-x[i]) / (y[i+1]-y[i])
    return float(pos), int(len(hits))


def station_shape(geometry, s):
    if geometry == "planar":
        bounds = [[s-HALF_NORMAL, s+HALF_NORMAL], [0., .001], [0., .001]]
        return {"type": "planar_slab", "center_xyz_m": [s, .0005, .0005],
                "normal_unit": [1., 0., 0.], "tangent_unit": [0., 1., 0.],
                "half_normal_m": HALF_NORMAL, "aabb_xyz_m": bounds,
                "x_bounds_m": bounds[0], "y_bounds_m": bounds[1], "z_bounds_m": bounds[2],
                "volume_m3": 2*HALF_NORMAL*1.e-6, "dx_m": .0002}
    c, sn = math.cos(THETA), math.sin(THETA)
    normal, tangent = np.array([-c, -sn]), np.array([sn, -c])
    center = np.array([R + (R+s)*c, (R+s)*sn])
    vertices = [center + i*HALF_NORMAL*normal+j*HALF_TANGENT*tangent for i in (-1,1) for j in (-1,1)]
    extents = np.array(vertices)
    if (extents[:,0].min() < -.2 or extents[:,0].max() > .65 or extents[:,1].min() < 0
        or extents[:,1].max() > .4 or s-HALF_NORMAL <= 0):
        raise ValueError("Cylinder sampling rectangle intersects the solid or lies outside the half-domain")
    # Station sites are clear of all refinement boundaries and of physical cut cells.
    lower_radius = R+s-HALF_NORMAL
    upper_radius = math.hypot(R+s+HALF_NORMAL,HALF_TANGENT)
    radii = np.array([R+.150, R+.120, R+.032, R+.012])
    if np.any((radii >= lower_radius)&(radii <= upper_radius)):
        raise ValueError("This case needs explicit heterogeneous refinement weights")
    level = int(np.count_nonzero(radii > upper_radius))
    area = 4*HALF_NORMAL*HALF_TANGENT
    bounds = [[float(extents[:,0].min()),float(extents[:,0].max())],
              [float(extents[:,1].min()),float(extents[:,1].max())],[-.5,.5]]
    return {"type":"oriented_rectangle_2d","center_xyz_m":[*center.tolist(),0.0],
            "normal_unit":[*normal.tolist(),0.0],"tangent_unit":[*tangent.tolist(),0.0],
            "half_normal_m":HALF_NORMAL,"half_tangent_m":HALF_TANGENT,"aabb_xyz_m":bounds,
            "gas_area_m2":area,"unit_span_m":1.0,"volume_m3":area,
            "base_relative_refinement_level_from_deck":level,
            "nominal_cell_xy_m":[(.85/776)/(2**level),(.4/400)/(2**level)]}


def summarize(x, fields, tau, bounds):
    mask=(x >= bounds[0]-1.e-12)&(x <= bounds[1]+1.e-12)
    if mask.sum()<2 or not np.all(np.isfinite(tau[:,mask])):
        raise ValueError(f"Missing support within {bounds}")
    values={}
    for key,field in fields.items():
        v=field[mask]
        if not np.all(np.isfinite(v)):
            raise ValueError(f"Missing {key} over {bounds}")
        values[key]={"min":float(v.min()),"mean":float(v.mean()),"max":float(v.max())}
    return {"coordinate_bounds_m":list(bounds),"sampled_coordinate_count":int(mask.sum()),
            "sampled_min_m":float(x[mask][0]),"sampled_max_m":float(x[mask][-1]),
            "fields":values,"tau_min_s":float(tau[:,mask].min()),"tau_max_s":float(tau[:,mask].max())}


def build_case(case_id, input_dir):
    gas, geom = case_id.split("_")
    coord, fields, tau, dt, source = profile(case_id, input_dir)
    gamma=GAS[gas]["gamma"]
    compression=(gamma+1)*100/((gamma-1)*100+2)
    stations=[]
    if geom=="planar":
        refs={"upstream_reference":(-.14,[-.145,-.135],"freestream-side planar plateau"),
              "downstream_reference":(.14,[.135,.145],"downstream planar plateau")}
    else:
        refs={"upstream_reference":(.195,[.190,.200],"freestream-side reference near the inlet"),
              "downstream_reference":(.035,[.030,.040],"post-shock wall-side gas still decelerating")}
    for station_id in ["upstream_reference","s10","s50","s90","downstream_reference"]:
        if station_id in refs:
            pos, window, role = refs[station_id]
            extra={}
        else:
            frac=int(station_id[1:])/100
            target=1+frac*(compression-1)
            pos,count=crossing(coord,fields["rho_over_rho_inf"],target,geom)
            role="inside the compression layer"
            window=(pos-HALF_NORMAL,pos+HALF_NORMAL)
            extra={"normal_shock_jump_fraction":frac,"threshold_rho_over_rho_inf":target,
                   "first_inward_crossing_count":count}
        shape=station_shape(geom,pos)
        station={"id":station_id,"center_m":pos,"center_xyz_m":shape["center_xyz_m"],
                 "role":role,"shape":shape,
                 "local_profile":summarize(coord,fields,tau,(pos-HALF_NORMAL,pos+HALF_NORMAL)),
                 "context_window":summarize(coord,fields,tau,window),**extra}
        stations.append(station)
    roots=np.array([s["center_m"] for s in stations if s["id"] in ("s10","s50","s90")])
    if not np.allclose(1000*roots,EXPECTED_ROOTS_MM[case_id],rtol=0,atol=2.e-5):
        raise ValueError(f"{case_id} differs from the independent published crossings: {roots*1000}")
    source["tau_basis"]="ten-block mean of the native tau field" if geom=="planar" else (
        "dt divided by each of the ten grid-block values of dt/tau, sampled on the ray")
    source["sampled_coordinate_extent_m"]=[float(coord[0]),float(coord[-1])]
    source["block_end_times_s"]=[i*BLOCK_S for i in range(11,21)]
    conf={"gas":gas,"geometry":geom,"freestream_Mach":10.0,"nominal_Kn":.01,"upstream":GAS[gas],
          "inflow_temperature_K":200.0,"native_dt_s":dt,"shape_membership":"half-open on each upper edge",
          "global_streamwise_unit":[1,0,0],"stations":stations,"provenance":source}
    if geom=="cylinder":
        conf.update({"coordinate":"wall distance s outward along theta=170 degrees from +x",
                     "cylinder_center_xyz_m":[R,0,0],"R_m":R,"theta_deg":170,
                     "count_weights":"global fnum times the solver bounding-cell-area weight; use actual grid and flow area"})
    else:
        conf.update({"coordinate":"native x increasing downstream",
                     "count_weights":"constant fnum per species; both homogeneous periodic cross-stream directions included"})
    return conf


def generate(input_dir):
    cases={key:build_case(key,input_dir) for key in INPUTS}
    tau_limit=max(s["local_profile"]["tau_max_s"] for c in cases.values() for s in c["stations"])
    possible=[5,8,10,12.5,20,25,40,50,100]
    interval=next(u*1.e-6 for u in possible if u*1.e-6 >= tau_limit
                  and abs(BLOCK_S/(u*1.e-6)-round(BLOCK_S/(u*1.e-6)))<1.e-8)
    for case in cases.values():
        ticks=interval/case["native_dt_s"]
        if abs(ticks-round(ticks))>1.e-8:
            raise ValueError("Sampling cadence must land on native solver steps")
        case["suggested_stride_steps"]=round(ticks)
        for station in case["stations"]:
            local=station["local_profile"]
            station["suggested_snapshot_spacing_over_local_tau"]=[
                interval/local["tau_max_s"],interval/local["tau_min_s"]]
    return {"schema_version":1,"cases":cases,
            "sampling":{"primary_snapshot_interval_s":interval,"pilot_observed_max_local_tau_s":tau_limit,
                        "decimation_factors":[1,2,4],"block_duration_s":BLOCK_S,
                        "snapshots_per_block":round(BLOCK_S/interval),"comparison_block_count":10,
                        "development_blocks":[1,2,3,4,5],"held_out_blocks":[6,7,8,9,10],
                        "bin_edges":"fix prior to examining held-out molecular data",
                        "weighting":"report raw counts, sum(w), sum(w*w), Kish count and bin masses per station and block",
                        "temporal_dependence":"report lagged moment/bin-mass correlations and block-to-block differences; collision-time spacing is not independence"},
            "provenance":{"prior_mean_window_s":[.002,.004],"phase":"source matched Mach-10 nonreacting controls",
                          "definition":"first freestream-to-wall (cylinder) or freestream-to-downstream (planar) crossing of 1+p*(rho_RH/rho_infty-1), without smoothing, on each own full SPARTA mean profile",
                          "geometry":"cylinder sampling basis is the inward ray and its counterclockwise transverse axis, not a measured local shock normal; laboratory vx is available in all four cases"}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    result=generate(args.inputs)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"stations.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf8")
    with (args.out/"station_summary.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["case","station","coordinate_mm","center_x_mm","center_y_mm",
                    "rho_over_rho_inf","Ttr_K","ux_m_s","tau_min_us","tau_max_us","role"])
        for key,case in result["cases"].items():
            for v in case["stations"]:
                local=v["local_profile"]; m=local["fields"];x,y,_=v["center_xyz_m"]
                w.writerow([key,v["id"],f"{1000*v['center_m']:.8f}",f"{1000*x:.8f}",f"{1000*y:.8f}",
                            f"{m['rho_over_rho_inf']['mean']:.8f}",f"{m['Ttr_K']['mean']:.5f}",
                            f"{m['ux_m_s']['mean']:.5f}",f"{1e6*local['tau_min_s']:.6f}",
                            f"{1e6*local['tau_max_s']:.6f}",v["role"]])
    print("Station groups",len(result["cases"]),"interval in microseconds",
          result["sampling"]["primary_snapshot_interval_s"]*1e6)


if __name__=="__main__":
    main()
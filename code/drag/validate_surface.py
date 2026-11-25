"""Independently test the signed drag decomposition, temporal statistics and inputs.

The verification path projects the native force along the endpoint normal.
It reconstructs the surface's normal load a second way from scalar pressure.
It does not call the production integrator when validating real data.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path


def check(condition, description):
    if not condition:
        raise ValueError(description)


def equal(actual, expected, label, rel=2e-10, atol=3e-13):
    check(math.isfinite(float(actual)) and math.isfinite(float(expected))
          and math.isclose(float(actual), float(expected), rel_tol=rel, abs_tol=atol),
          f"{label}: {actual} versus {expected}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def table(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def independently_integrate(raw, end, samples, half_count=None, body=None):
    lines = raw.decode("ascii").splitlines()
    check(lines[0] == "ITEM: TIMESTEP" and int(lines[1]) == end, "Wrong frame time")
    header = lines[8].split()[2:]
    check(header == "id v1x v1y v2x v2y area".split() + [f"f_wall[{k}]" for k in range(1, 9)], "Unexpected native header")
    ids = set()
    totals = [[] for _ in range(9)]
    for row in lines[9:]:
        index, x1, y1, x2, y2, a, n, fx, fy, px, py, sx, sy, pressure = map(float, row.split())
        check(index.is_integer() and index > 0 and index not in ids, "ID duplicated or nonintegral")
        ids.add(index)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        equal(a, length, "Native area vs line length", rel=2e-10, atol=1e-14)
        if body == "cylinder":
            for x, y in ((x1, y1), (x2, y2)):
                equal(math.hypot(x-.1524, y), .1524, "Analytic cylinder radius")
                check(y >= -2e-12, "Cylinder below symmetry line")
        elif body == "plate":
            check(min(y1,y2) >= -2e-12 and max(y1,y2) <= .1524 + 2e-12 and
                  ((abs(x1)+abs(x2) < 2e-12) or (abs(x1-.02)+abs(x2-.02) < 2e-12) or
                   (abs(y1-.1524)+abs(y2-.1524) < 2e-12)), "Element outside analytical half plate")
        # Independent of the velocity distribution, its signed momentum vector
        # projects onto an orthonormal tangent/normal basis for each straight line.
        tx, ty = dx / length, dy / length
        normal_work = fx * ty - fy * tx
        tangential_work = fx * tx + fy * ty
        nxforce, nyforce = normal_work * ty, -normal_work * tx
        txforce, tyforce = tangential_work * tx, tangential_work * ty
        for predicted, native in ((nxforce,px*a),(nyforce,py*a),(txforce,sx*a),(tyforce,sy*a),
                                  (pressure*dy,px*a),(-pressure*dx,py*a)):
            equal(predicted, native, "Element projected load", rel=2e-8, atol=3e-13)
        count = round(n*samples)
        equal(n*samples, count, "Simulated particle count", rel=1e-12, atol=1e-6)
        check(count >= 0, "Negative simulated impacts")
        for acc, value in zip(totals, (fx,fy,nxforce,nyforce,txforce,tyforce,pressure*dy,length,count)):
            acc.append(value)
    check(len(ids) == int(lines[3]) == len(lines) - 9, "Incomplete native frame")
    if half_count is not None:
        check(len(ids) == half_count, "Incorrect number of upper-body elements")
    vals = list(map(math.fsum, totals))
    netx, nety, pressx, pressy, shearx, sheary, from_scalar, _area, impacts = vals
    equal(netx, pressx+shearx, "Total streamwise force balance")
    equal(nety, pressy+sheary, "Total transverse force balance")
    equal(pressx, from_scalar, "Independent scalar-pressure surface integral")
    check(netx > 0 and impacts > 0, "Unresolved positive drag or impacts")
    return vals


def verify(root, parent, output):
    prov = json.loads((output / "drag_provenance.json").read_bytes())
    for name, value in prov["result_sha256"].items():
        check(sha((output / name).read_bytes()) == value, f"Changed result table {name}")
    sub = table(output/"drag_subwindows.csv")
    blocks = table(output/"drag_blocks.csv")
    cases = table(output/"drag_summary.csv")
    pairs = table(output/"drag_shape_comparison.csv")
    check(len(sub) == len(cases)*50 and len(blocks) == len(cases)*10 and
          len(cases) == len(prov["case_order"]) == len({c["case"] for c in cases}), "Record cardinality mismatch")
    names = set(prov["case_order"])
    check(names == {x["case"] for x in cases} == {x["case"] for x in blocks} == {x["case"] for x in sub},
          "Extra or missing case")
    rerun, diagnostics = {}, {}
    for c in cases:
        name, body = c["case"], c["body"]
        parent_dir = parent/"cases"/name
        original = (parent_dir/"in.case").read_text()
        gas = re.search(r"^mixture gas (\S+)",original,re.M)[1]
        flow = re.search(r"^mixture gas .*? nrho (\S+) vstream (\S+) ",original,re.M)
        spec = re.search(r"^species (\S+) ",original,re.M)[1]
        species_file = parent/Path(spec).name
        mass = float(next(l.split()[2] for l in species_file.read_text().splitlines() if l.startswith(gas+" ")))
        config = json.loads((parent_dir/"case.json").read_text())
        den = .5 * mass * float(flow[1]) * float(flow[2])**2 * .3048
        equal(float(c["dynamic_pressure_Pa"]) * .3048, den, "Freestream drag denominator")
        base, width, dt = config["total_steps"], config["block_steps"], config["dt"]
        frame_rows = {(int(r["block"]), int(r["subwindow"])):r for r in sub if r["case"] == name}
        block_rows = {int(r["block"]):r for r in blocks if r["case"] == name}
        check(len(frame_rows) == 50 and len(block_rows) == 10, "Repeated or missing temporal windows")
        commitments = {}
        for f in (root/"cases"/name/"attempts").glob("attempt-*/log.drag"):
            for match in re.finditer(r"^DRAG_BLOCK_COMMITTED (\d+) (\d+) (\d+) state[01]\.restart$", f.read_text(errors="replace"), re.M):
                k, end, particles = (int(match[i]) for i in (1,2,3))
                check(k not in commitments and particles>0 and end==base+(k+1)*width, "Ambiguous or invalid native commit")
                commitments[k] = (f.parent, end, particles)
        check(set(commitments) == set(range(1,11)), "Native committed sequence is incomplete")
        coef = {x:[] for x in ("Cd","Cd_pressure","Cd_shear")}
        total_hits = 0
        max_frame_coef_difference = {key:0.0 for key in coef}
        for k in range(1,11):
            vals = []
            committed, end, particles = commitments[k]
            b = block_rows[k]
            check(b["winning_attempt"] == committed.name and int(b["solver_particles_at_commit"]) == particles and
                  int(b["end_step_inclusive"]) == end and int(b["start_step_exclusive"]) == end-width,
                  "Block owner, particle count or interval mismatch")
            for j in range(1,6):
                f = frame_rows[k,j]
                t1 = base+width+k*width-width+j*(width//5)
                t0 = t1-width//5
                check(int(f["end_step_inclusive"])==t1 and int(f["start_step_exclusive"])==t0, "Bad subwindow interval")
                equal(f["start_s"], t0*dt, "Physical start time")
                equal(f["end_s"], t1*dt, "Physical end time")
                native = committed/f"wall.{t1}.surf"
                check(native.relative_to(root).as_posix() == f["native_dump"], "Wrong native dump selected")
                raw = native.read_bytes()
                check(sha(raw) == f["dump_sha256"], "Native dump SHA mismatch")
                elems = 2048 if body=="cylinder" else 326
                a = independently_integrate(raw,t1,width//5,elems,body)
                length = .3048*2048*math.sin(math.pi/4096) if body=="cylinder" else .3248
                equal(a[7],length,"Analytic upper-body perimeter")
                equal(f["simulated_impacts_half"], a[8], "Native impacts")
                force_names = ("Fx_half","Fy_half","Fx_pressure_half","Fy_pressure_half","Fx_shear_half","Fy_shear_half")
                for key, x in zip(force_names,a[:6]):
                    equal(f[key+"_N_per_m"],x,"Subwindow "+key)
                for key, idx in zip(coef,(0,2,4)):
                    independent = 2*a[idx]/den
                    equal(f[key],independent,"Subwindow "+key)
                    max_frame_coef_difference[key] = max(max_frame_coef_difference[key], abs(float(f[key])-independent))
                vals.append(a)
            hits = sum(int(x[8]) for x in vals)
            total_hits += hits
            check(hits == int(b["simulated_impacts_half"]), "Block hits")
            for component, idx in zip(("Fx", "Fx_pressure", "Fx_shear"),(0,2,4)):
                equal(b[component+"_full_N_per_m"],2*math.fsum(x[idx] for x in vals)/5,"Full-body "+component)
            for key, idx in zip(coef,(0,2,4)):
                value = 2*math.fsum(x[idx] for x in vals)/(5*den)
                equal(b[key],value,"Block "+key)
                coef[key].append(value)
        check(int(c["simulated_impacts_half"]) == total_hits, "Case impact total")
        for key, series in coef.items():
            mean = math.fsum(series)/10
            sd = math.sqrt(math.fsum((x-mean)**2 for x in series)/9)
            for field, val in (("mean",mean),("block_sd",sd),("first_block",series[0]),("last_block",series[-1]),
                               ("first5_mean",math.fsum(series[:5])/5),("last5_mean",math.fsum(series[5:])/5)):
                equal(c[key+"_"+field],val,field+" "+key)
        rerun[(gas,float(c["Kn"]),body)] = {k:math.fsum(v)/10 for k,v in coef.items()}
        for component, key in zip(("Fx", "Fx_pressure", "Fx_shear"),rerun[(gas,float(c["Kn"]),body)]):
            equal(c[component+"_full_N_per_m_mean"],rerun[(gas,float(c["Kn"]),body)][key]*den,"Full-body mean force")
        diagnostics[name] = dict(simulated_impacts_half=total_hits, committed_blocks=10, matched_native_frames=50,
                independently_projected_mean=rerun[(gas,float(c["Kn"]),body)],
                max_absolute_subwindow_coefficient_difference=max_frame_coef_difference)
    expected_pairs = {(g,kn) for g,kn,b in rerun if (g,kn,"plate") in rerun and (g,kn,"cylinder") in rerun}
    check(expected_pairs == {(r["gas"],float(r["Kn"])) for r in pairs} and len(pairs)==len(expected_pairs), "Missing matched body pair")
    for r in pairs:
        gas,kn = r["gas"],float(r["Kn"])
        c,p = (rerun[(gas,kn,b)] for b in ("cylinder","plate"))
        for key in c:
            equal(r["plate_minus_cylinder_"+key],p[key]-c[key],"Shape difference "+key)
        equal(r["plate_over_cylinder_Cd"],p["Cd"]/c["Cd"],"Shape Cd ratio")
    report = dict(status="Verified",case_count=len(cases),subwindows=len(sub),committed_blocks=len(blocks),
                  shape_pairs=len(pairs),cases=diagnostics,validator_sha256=sha(Path(__file__).read_bytes()),
                  result_sha256=prov["result_sha256"],
                  method="Force-only projections on each native line and a scalar-pressure line integral independently verify the signed normal and shear forces. A separate reduction checks every reported temporal coefficient and every plate-cylinder difference.")
    fd,tmp = tempfile.mkstemp(dir=output,prefix="drag-validation-",suffix=".tmp")
    with os.fdopen(fd,"w",encoding="utf-8") as f:
        json.dump(report,f,indent=2)
        f.write("\n")
    os.replace(tmp,output/"drag_independent_validation.json")
    return report


def self_test():
    # These invented forces and counters test signs, units and failure paths.
    # Synthetic data never appear in the scientific results.
    from analyze_surface import EvidenceError, FIELDS, NATIVE, digest, integrate_dump, point_key, resolve_attempts, physics_signature

    def rejected(fn):
        try:
            fn()
        except (ValueError, EvidenceError):
            return
        raise RuntimeError("A deliberately corrupt synthetic fixture was accepted")

    with tempfile.TemporaryDirectory(prefix="synthetic-drag-",dir=Path(__file__).resolve().parent) as directory:
        tree=Path(directory)
        head = "ITEM: TIMESTEP\n10\nITEM: NUMBER OF SURFS\n2\nITEM: BOX BOUNDS oo ro pp\n-1 1\n0 1\n-0.5 0.5\nITEM: SURFS "+" ".join(FIELDS)+"\n"
        row1="1 0 0 0 0.05 .05 .7 .3 .1 6 0 0 2 6\n"
        row2="2 0 .05 .1 .05 .1 1 .3 -.4 0 -4 3 0 4\n"
        p=tree/"wall.10.surf"
        meta={"width":50,"box":[-1,1,0,1,-.5,.5],"q":10,
              "geometry":{point_key((0,0,0,.05)):.05,point_key((0,.05,.1,.05)):.1}}
        def raw_results(content):
            p.write_text(content,encoding="ascii")
            return integrate_dump(p,10,meta,None)
        val, _, _ = raw_results(head+row1+row2)
        equal(val["Cd"],1.2/(10*.3048),"Synthetic whole-body Cd")
        equal(val["Cd_pressure"],.6/(10*.3048),"Synthetic pressure Cd")
        equal(val["Cd_shear"],.6/(10*.3048),"Synthetic shear Cd")
        equal(val["Fy_half_N_per_m"],-.3,"Synthetic nonzero half-body transverse load")
        check(val["simulated_impacts_half"]==17,"Synthetic impacts")
        independent=independently_integrate(p.read_bytes(),10,10,2)
        equal(independent[0],.6,"Independent synthetic drag")
        rejected(lambda:raw_results(head+row1+row2.replace("2 0","1 0",1)))
        rejected(lambda:raw_results(head+row1+row2.replace(" .3 -.4"," .35 -.4")))
        rejected(lambda:raw_results(head+row1+row2.replace("-4 3 0 4","-4 3 0 7")))
        rejected(lambda:raw_results(head+row1.replace(".05 .7",".05 .71")+row2))
        rejected(lambda:raw_results(head.replace("f_wall[1]","f_wall[0]")+row1+row2))
        name="Ar_plate_Kn0p1"
        parent=tree/"cases"/name/"attempts"
        start=40000
        width=2000
        pin=dict(original_job="321",restart_stamp=[3,77,1234,5678])
        physical=["mixture gas Ar nrho 1e19 vstream 2600 0 0 temp 200","collide vss gas ../../argon.vss relax constant",
                  "collide_modify nearcp yes 10 rotate smooth vibrate no","surf_collide wall diffuse 500.0 1.0",
                  "surf_modify all collide wall","fix inflow emit/face gas xlo xhi yhi twopass","timestep 1.5e-07",
                  "fix lb balance 2000 1.1 rcb part"]
        cm=dict(case=name,base=start,width=width,np=100,dt=1.5e-7,ntasks=32,pin=pin,binary="c"*64,
                parent_physics=physics_signature(chr(10).join(physical)))
        def attempt(i,k0,ks):
            d=parent/f"attempt-{i:04}"
            d.mkdir(parents=True)
            source=(f"/original/cases/{name}/checkpoint.final" if not k0 else
                    f"/scenario/cases/{name}/attempts/attempt-0001/state{k0%2}.restart")
            step=start if not k0 else start+(k0+1)*width
            count=100+k0
            su=width//5
            decklines=["read_restart "+source+" balance rcb cell"]+physical+["run 0"]
            if not k0:decklines.append("run "+str(width))
            decklines.extend(["compute hits surf all gas "+NATIVE,
                              f"fix wall ave/surf all 1 {su} {su} c_hits[*]","run 1",
                              f"dump wallDump surf all {su} wall.*.surf "+" ".join(FIELDS),
                              "dump_modify wallDump format float %.16g",f"run {width-1}"]+
                             [f"run {width}"]*(10-k0-1))
            dr=("\n".join(decklines)+"\n").encode()
            (d/"in.drag").write_bytes(dr)
            startup=dict(case=name,attempt=i,source_closed_blocks=k0,sample_blocks=10,subblocks_per_block=5,
                         sample_every_solver_step=True,MPI_ranks=32,binary_sha256="c"*64,original_parent_job="321",
                         source_step=step,source_particles=count,source=source,source_sha256="a"*64,
                         source_stamp=pin["restart_stamp"],deck_sha256=digest(dr),job_id="322")
            (d/"startup.json").write_text(json.dumps(startup))
            log=f"DRAG_ADMISSION {step} {count} {step*cm['dt']:.9g} 1.5e-07\n"
            for k in ks:log+=f"DRAG_BLOCK_COMMITTED {k} {start+(k+1)*width} {100+k} state{k%2}.restart\n"
            (d/"log.drag").write_text(log)
            return d
        a=attempt(1,0,range(1,5))
        b=attempt(2,4,range(5,11))
        (a/"wall.51600.surf").write_text("uncommitted synthetic partial")
        ignored=parent/"attempt-0003"
        ignored.mkdir()
        (ignored/"log.drag").write_text("native preemption before a commit")
        win,audit=resolve_attempts(tree,cm)
        check(len(win)==10 and win[4][0]==a and win[5][0]==b and audit["ignored_uncommitted_attempts"]==[ignored.name],
              "Requeue fixture chose the wrong winning block")
        log=(a/"log.drag").read_text()
        (a/"log.drag").write_text(log+"DRAG_BLOCK_COMMITTED 5 52000 105 state1.restart\n")
        rejected(lambda:resolve_attempts(tree,cm))
        (a/"log.drag").write_text(log)
        bad=(b/"log.drag").read_text().replace("DRAG_ADMISSION 50000 104","DRAG_ADMISSION 50000 103")
        (b/"log.drag").write_text(bad)
        rejected(lambda:resolve_attempts(tree,cm))
    print("Synthetic tests passed: signs, normalization, counts, corrupt rows, geometry identity, preemption and duplicate committed blocks.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--parent-root",type=Path)
    parser.add_argument("--results",type=Path)
    parser.add_argument("--self-test",action="store_true",help="Run synthetic regression fixtures")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    if args.results is not None or not args.self_test:
        if args.parent_root is None or args.results is None:
            parser.error("Real validation requires --parent-root and --results.")
        report=verify(args.run_root.resolve(),args.parent_root.resolve(),args.results.resolve())
        print(f"Independently verified {report['case_count']} cases, {report['committed_blocks']} blocks, {report['subwindows']} dumps and {report['shape_pairs']} matched pairs.")


if __name__=="__main__":
    main()

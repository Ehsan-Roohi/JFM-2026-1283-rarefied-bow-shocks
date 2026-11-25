"""Integrate committed native SPARTA wall tallies for the matched Mach-10 bodies.

Run with Python 3.8+ and no external dependencies.  Every file selector is
relative to --run-root or --parent-root.  See drag_provenance.json for units,
time windows, reproducibility information and the native source definitions.
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

D = 0.3048
BLOCKS, SUBS = 10, 5
CASES = tuple(f"{gas}_{body}_{kn}" for gas in ("Ar", "N2")
              for kn in ("Kn0p01", "Kn0p1") for body in ("cylinder", "plate"))
FIELDS = ["id", "v1x", "v1y", "v2x", "v2y", "area"] + [f"f_wall[{i}]" for i in range(1, 9)]
NATIVE = "n fx fy px py shx shy press"
COMPONENTS = ("Cd", "Cd_pressure", "Cd_shear")
FORCES = ("Fx_half_N_per_m", "Fy_half_N_per_m", "Fx_pressure_half_N_per_m",
          "Fy_pressure_half_N_per_m", "Fx_shear_half_N_per_m", "Fy_shear_half_N_per_m")
COMMIT = re.compile(r"^DRAG_BLOCK_COMMITTED (\d+) (\d+) (\d+) (state[01]\.restart)$", re.M)
ADMISSION = re.compile(r"^DRAG_ADMISSION (\d+) (\d+) ([\d.eE+-]+) ([\d.eE+-]+)$", re.M)
RHO_DT_TOL = 5e-7


class EvidenceError(ValueError):
    """A required native scientific or provenance check failed."""


def require(condition, message):
    if not condition:
        raise EvidenceError(str(message))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def checked_bytes(path, expected):
    data = path.read_bytes()
    require(digest(data) == expected, f"SHA-256 mismatch for {path}")
    return data


def instruction(text, prefix):
    rows = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    matched = [row for row in rows if row.startswith(prefix)]
    require(len(matched) == 1, f"Expected one {prefix!r}, found {matched}")
    return matched[0].split()


def near(a, b, rel=1e-11, absolute=1e-13):
    return math.isclose(a, b, rel_tol=rel, abs_tol=absolute)


def point_key(coords):
    # The generator uses %.16g. Rounding is solely for identifying
    # segments across clipping/restarts and does not change force quadrature.
    return tuple(round(x, 10) for x in coords)


def half_geometry(raw, body):
    rows = [line.strip() for line in raw.decode().splitlines() if line.strip()]
    pstart, lstart = rows.index("Points"), rows.index("Lines")
    point_rows = [row.split() for row in rows[pstart + 1:lstart]]
    points = {int(row[0]): tuple(map(float, row[1:3])) for row in point_rows}
    require(len(points) == len(point_rows), "Duplicate source point")
    full_ids, segments = set(), {}
    for row in rows[lstart + 1:]:
        label, a, b = map(int, row.split())
        require(label not in full_ids, f"Repeated source segment {label}")
        full_ids.add(label)
        x1, y1 = points[a]
        x2, y2 = points[b]
        y1, y2 = [0.0 if abs(y) < 1e-12 else y for y in (y1, y2)]
        if max(y1, y2) <= 0.0:
            continue
        if min(y1, y2) < 0.0:
            cross = x1 + (x2 - x1) * (-y1 / (y2 - y1))
            if y1 < 0.0:
                x1, y1 = cross, 0.0
            else:
                x2, y2 = cross, 0.0
        key = point_key((x1, y1, x2, y2))
        length = math.hypot(x2 - x1, y2 - y1)
        require(key not in segments and length > 0, "Repeated or degenerate clipped segment")
        segments[key] = length
    expected_n = 2048 if body == "cylinder" else 326
    expected_length = (expected_n * D * math.sin(math.pi / (2 * expected_n))
                       if body == "cylinder" else D + 0.020)
    total = math.fsum(segments.values())
    require(len(segments) == expected_n and near(total, expected_length),
            f"{body}: unexpected half geometry, {len(segments)} segments, {total} m")
    return segments


def physics_signature(deck):
    fields = ("mixture gas ", "collide ", "collide_modify ", "surf_collide ",
              "surf_modify ", "fix inflow ", "timestep ", "fix lb ")
    result = {}
    for field in fields:
        tokens = instruction(deck, field)
        if field == "collide ":
            tokens[3] = tokens[3].replace(chr(92), "/").rsplit("/", 1)[-1]
        result[field.strip()] = tokens
    return result


def parent_evidence(parent, name, pins):
    case = parent / "cases" / name
    pin = pins["cases"][name]
    cfg = json.loads(checked_bytes(case / "case.json", pin["case.json"]))
    deck = checked_bytes(case / "in.case", pin["in.case"]).decode()
    log = checked_bytes(case / "log.case", pin["log_sha256"]).decode(errors="replace")
    gas, body, kn_label = name.split("_")
    require((gas, body, float(kn_label[2:].replace("p", "."))) ==
            (cfg["gas"], cfg["body"], cfg["Kn"]), "Case label and metadata disagree")
    require(instruction(deck, "dimension ") == ["dimension", "2"] and
            instruction(deck, "boundary ") == ["boundary", "o", "ro", "p"], "Expected 2D reflecting y-low boundary")
    surf = instruction(deck, "read_surf ")
    require(Path(surf[1]).name == body + ".surf" and surf[2:] == ["clip", "1e-12"], "Surface clip mismatch")
    require(instruction(deck, "surf_collide ")[2:] == ["diffuse", "500.0", "1.0"], "Wall model changed")
    config_domain = cfg["domain"]
    require(config_domain[2] == 0 and config_domain[3] > D / 2, "Half-body box mismatch")
    bounds = list(map(float, instruction(deck, "create_box ")[1:]))
    require(bounds == list(config_domain) + [-.5, .5], "Input box differs from the expected unit span")
    species = instruction(deck, "species ")
    require(gas in species[2:], "Case species is absent")
    mix = instruction(deck, "mixture gas ")
    require(mix[2] == gas, "Incorrect incident species")
    n_inf = float(mix[mix.index("nrho") + 1])
    vind = mix.index("vstream") + 1
    u, v, w = map(float, mix[vind:vind + 3])
    require(n_inf > 0 and u > 0 and (v, w) == (0.0, 0.0) and
            near(n_inf, cfg["nrho"], RHO_DT_TOL) and near(u, cfg["U"], RHO_DT_TOL), "Freestream mismatch")
    sfile = Path(species[1]).name
    spec = checked_bytes(parent / sfile, pins["shared"][sfile]).decode()
    mass = float(instruction(spec, gas + " ")[2])
    require(mass > 0, "Invalid molecular mass")
    q = 0.5 * mass * n_inf * u * u
    dt = float(instruction(deck, "timestep ")[1])
    width, base = cfg["block_steps"], cfg["total_steps"]
    require(cfg["nblocks"] == BLOCKS and width % SUBS == 0 and
            cfg["warm_steps"] + BLOCKS * width == base and near(width * dt, .0003) and
            near(dt, cfg["dt"]) and pin["parent_source_step"] == base,
            "Parent time scales or sampling windows changed")
    require(re.search(r"^CURVATURE_RUN_COMPLETE " + str(base) + r"$", log, re.M), "Missing parent completion")
    end_rows = re.findall(r"^\s*" + str(base) + r"\s+[\d.eE+-]+\s+(\d+)\s+\d+\s+\d+\s+\d+\s*$", log, re.M)
    require(end_rows and int(end_rows[-1]) > 0, "Missing original final particle count")
    geometry_raw = checked_bytes(parent / surf[1].split("/")[-1], pins["shared"][body + ".surf"])
    return dict(case=name, gas=gas, body=body, Kn=cfg["Kn"], dt=dt, width=width,
                base=base, np=int(end_rows[-1]), ntasks=cfg["ntasks"], box=bounds, geometry=half_geometry(geometry_raw, body),
                parent_physics=physics_signature(deck), n_inf=n_inf, mass=mass, u=u, q=q, pin=pin, binary=pins["binary_sha256"])


def resolve_attempts(run, meta):
    name, base, width = meta["case"], meta["base"], meta["width"]
    winners, admissions, ignored = {}, {}, []
    for folder in sorted((run / "cases" / name / "attempts").glob("attempt-*")):
        log_path = folder / "log.drag"
        if not log_path.is_file():
            ignored.append(folder.name)
            continue
        log_raw = log_path.read_bytes()
        log = log_raw.decode(errors="replace").replace("\r\n", "\n")
        markers = COMMIT.findall(log)
        if not markers:
            ignored.append(folder.name)
            continue
        start_path, deck_path = folder / "startup.json", folder / "in.drag"
        start_raw = start_path.read_bytes()
        startup = json.loads(start_raw)
        deck_raw = checked_bytes(deck_path, startup["deck_sha256"])
        deck = deck_raw.decode()
        require(physics_signature(deck) == meta["parent_physics"], "Continuation physics differ from the pinned parent")
        k0 = startup["source_closed_blocks"]
        require(startup["case"] == name and int(folder.name.split("-")[1]) == startup["attempt"]
                and 0 <= k0 < BLOCKS and startup["sample_blocks"] == BLOCKS
                and startup["subblocks_per_block"] == SUBS and startup["sample_every_solver_step"] is True
                and startup["MPI_ranks"] == meta["ntasks"] and startup["binary_sha256"] == meta["binary"]
                and str(startup["original_parent_job"]) == str(meta["pin"]["original_job"]), f"{folder.name}: inconsistent startup protocol")
        expected_first = base if k0 == 0 else base + (k0 + 1) * width
        adm = ADMISSION.findall(log)
        require(len(adm) == 1, f"{folder.name}: missing or repeated native admission")
        st, npart = map(int, adm[0][:2])
        t, dt = map(float, adm[0][2:])
        require(st == startup["source_step"] == expected_first and npart == startup["source_particles"] > 0
                and near(dt, meta["dt"], RHO_DT_TOL) and near(t, st * dt, RHO_DT_TOL), "Native restart admission mismatch")
        src = Path(startup["source"].replace(chr(92), "/"))
        require(instruction(deck, "read_restart ")[1].replace(chr(92), "/") == src.as_posix()
                and re.fullmatch("[0-9a-f]{64}", startup["source_sha256"]), "Incorrect restart provenance")
        if k0 == 0:
            require(src.name == "checkpoint.final" and src.parent.name == name and npart == meta["np"],
                    "Cold admission does not match the original final state")
            # Device IDs may differ between mounts. Inode, byte count and mtime are frozen on Unity.
            require(startup["source_stamp"][1:] == meta["pin"]["restart_stamp"][1:], "Original restart stamp mismatch")
        else:
            require(src.name == f"state{k0 % 2}.restart", "Wrong alternating restart for continuation")
        sub = width // SUBS
        require(instruction(deck, "compute hits ") == ("compute hits surf all gas " + NATIVE).split()
                and instruction(deck, "fix wall ") == f"fix wall ave/surf all 1 {sub} {sub} c_hits[*]".split()
                and instruction(deck, "dump wallDump ") == f"dump wallDump surf all {sub} wall.*.surf".split() + FIELDS
                and instruction(deck, "dump_modify wallDump ") == "dump_modify wallDump format float %.16g".split(),
                "Force compute, averaging cadence, dump columns or precision changed")
        runs = [int(line.split()[1]) for line in deck.splitlines() if line.startswith("run ")]
        require(runs == [0] + ([width] if k0 == 0 else []) + [1, width - 1] + [width] * (BLOCKS - k0 - 1),
                "Unexpected settling or production run lengths")
        require(list(map(lambda m: int(m[0]), markers)) == list(range(k0 + 1, k0 + 1 + len(markers))),
                "Noncontiguous native commitment sequence")
        for k, step, n, restart in markers:
            k, step, n = map(int, (k, step, n))
            require(1 <= k <= BLOCKS and step == base + (k + 1) * width and n > 0 and restart == f"state{k % 2}.restart",
                    f"Unexpected native commitment in {folder.name}")
            require(k not in winners, f"Ambiguous duplicate winning attempts for {name}, block {k}")
            winners[k] = (folder, step, n)
        completion = bool(re.search(r"^JFM_DRAG_SAMPLING_COMPLETE$", log, re.M))
        exit_path = folder / "exit.json"
        exit_data = exit_path.read_bytes() if exit_path.is_file() else None
        exit_code = None
        if exit_data is not None:
            termination = json.loads(exit_data)
            exit_code = termination["returncode"]
            require(termination["closure_markers"] == [list(m) for m in markers] and
                    termination["complete"] == (exit_code == 0 and completion and len(markers) == BLOCKS-k0),
                    "Attempt exit report differs from native commitment/completion markers")
        native_banner = next((line for line in log.splitlines() if line.startswith("SPARTA (")), None)
        committed_end_steps = {int(m[1]) - width + j*sub for m in markers for j in range(1, SUBS+1)}
        outside = sorted(f.name for f in folder.glob("wall.*.surf")
                         if f.name not in {f"wall.{s}.surf" for s in committed_end_steps})
        admissions[folder.name] = dict(attempt=folder.name, job_id=str(startup["job_id"]), source_closed_blocks=k0,
                native_solver_banner=native_banner, excluded_wall_dumps_outside_committed_intervals=outside,
                source_attempt=src.parent.name if k0 else None, source_step=st, source_particles=npart,
                source_sha256=startup["source_sha256"], log_sha256=digest(log_raw), deck_sha256=digest(deck_raw),
                startup_sha256=digest(start_raw), native_completion=completion, exit_code=exit_code,
                exit_sha256=digest(exit_data) if exit_data is not None else None)
    require(set(winners) == set(range(1, BLOCKS + 1)), f"{name}: {len(winners)} of 10 native committed blocks")
    for item in admissions.values():
        k = item["source_closed_blocks"]
        if k:
            prev, step, count = winners[k]
            require(item["source_attempt"] == prev.name and item["source_step"] == step and
                    item["source_particles"] == count, "Broken admitted restart lineage")
    terminal = admissions[winners[BLOCKS][0].name]
    audit = dict(winning_attempts=list(admissions.values()), ignored_uncommitted_attempts=ignored,
                 final_native_and_zero_exit_confirmed=terminal["native_completion"] and terminal["exit_code"] == 0)
    return winners, audit


def integrate_dump(path, step, meta, known_ids):
    raw = path.read_bytes()
    lines = raw.splitlines()
    require(len(lines) >= 10 and lines[0] == b"ITEM: TIMESTEP" and int(lines[1]) == step and
            lines[2] == b"ITEM: NUMBER OF SURFS" and lines[4].startswith(b"ITEM: BOX BOUNDS ") and
            lines[8].decode().split() == ["ITEM:", "SURFS"] + FIELDS, f"Native dump header mismatch in {path.name}")
    bounds = [float(x) for line in lines[5:8] for x in line.split()]
    require(len(bounds) == 6 and all(near(x, y, 1e-6) for x, y in zip(bounds, meta["box"])),
            "Native unit-span half-domain bounds mismatch")
    declared = int(lines[3])
    require(declared == len(meta["geometry"]) == len(lines) - 9, "Incomplete surface dump or wrong body")
    seen, ids = set(), {}
    all_forces = [[] for _ in FORCES]
    hits, hit_elements, step_count = 0, 0, meta["width"] // SUBS
    max_row_error = max_orientation_error = 0.0
    for line in lines[9:]:
        cells = line.split()
        require(len(cells) == len(FIELDS), f"Malformed surface row at {path.name}")
        sid = int(cells[0])
        values = tuple(map(float, cells[1:]))
        require(all(map(math.isfinite, values)), "Non-finite wall tally")
        x1, y1, x2, y2, area, n, fx, fy, px, py, sx, sy, press = values
        key = point_key(values[:4])
        require(key in meta["geometry"] and key not in seen and sid not in ids and sid > 0,
                f"Unexpected or repeated wall segment in {path.name}, ID {sid}")
        seen.add(key)
        ids[sid] = key
        length = math.hypot(x2 - x1, y2 - y1)
        require(near(area, length, 5e-10, 2e-14) and near(length, meta["geometry"][key], 5e-10, 2e-14),
                "Native area disagrees with the clipped reference line length")
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        pressure_x, pressure_y = area * px, area * py
        shear_x, shear_y = area * sx, area * sy
        scale = max(abs(fx), abs(fy), abs(pressure_x), abs(pressure_y), abs(shear_x), abs(shear_y), area * abs(press), 1e-200)
        residual = max(abs(fx - pressure_x - shear_x), abs(fy - pressure_y - shear_y))
        orient = area * max(abs(px + press * nx), abs(py + press * ny), abs(sx * nx + sy * ny))
        require(residual <= 2e-9 * scale + 2e-16 and orient <= 2e-9 * scale + 2e-16,
                f"Signed force or normal/shear projection fails at {path.name}, ID {sid}")
        max_row_error = max(max_row_error, residual)
        max_orientation_error = max(max_orientation_error, orient)
        hit_count = n * step_count
        rounded = round(hit_count)
        require(n >= 0 and near(hit_count, rounded, 1e-12, 1e-6), "Mean simulated impact count is not integral over the sample")
        hits += rounded
        hit_elements += (rounded > 0)
        for store, val in zip(all_forces, (fx, fy, pressure_x, pressure_y, shear_x, shear_y)):
            store.append(val)
    if known_ids is not None:
        require(ids == known_ids, "Native element ID/geometry assignments changed within an attempt")
    fx, fy, pfx, pfy, sfx, sfy = sums = list(map(math.fsum, all_forces))
    close_x, close_y = fx - pfx - sfx, fy - pfy - sfy
    denom = meta["q"] * D
    require(fx > 0 and hits > 0 and abs(close_x) <= 1e-9 * max(abs(fx), abs(pfx), abs(sfx), 1e-14)
            and abs(close_y) <= 1e-9 * max(abs(fy), abs(pfy), abs(sfy), 1e-14) + 1e-13,
            "Net signed momentum balance or positive integrated drag/count failed")
    vals = dict(zip(FORCES, sums))
    vals.update(Cd=2 * fx / denom, Cd_pressure=2 * pfx / denom, Cd_shear=2 * sfx / denom,
                simulated_impacts_half=hits, hit_segments=hit_elements, wall_segments=declared,
                closure_x_N_per_m=close_x, closure_y_N_per_m=close_y,
                max_element_closure_N_per_m=max_row_error, max_projection_residual_N_per_m=max_orientation_error)
    return vals, ids, digest(raw)


def analyze_case(run, parent, name, pins):
    meta = parent_evidence(parent, name, pins)
    win, audit = resolve_attempts(run, meta)
    force_sub, force_blocks, attempt_ids = [], [], {}
    w, dt = meta["width"], meta["dt"]
    for block in range(1, BLOCKS + 1):
        folder, end, npart = win[block]
        begin = end - w
        items = []
        for j in range(1, SUBS + 1):
            step = begin + j * (w // SUBS)
            path = folder / f"wall.{step}.surf"
            require(path.is_file(), f"Missing committed native dump {path}")
            vals, ids, sha = integrate_dump(path, step, meta, attempt_ids.get(folder.name))
            attempt_ids[folder.name] = ids
            rec = dict(case=name, gas=meta["gas"], body=meta["body"], Kn=meta["Kn"], block=block,
                       subwindow=j, start_step_exclusive=step-w//SUBS, end_step_inclusive=step,
                       start_s=(step-w//SUBS)*dt, end_s=step*dt, attempt=folder.name,
                       native_dump=path.relative_to(run).as_posix(), dump_sha256=sha, **vals)
            force_sub.append(rec)
            items.append(rec)
        block_avgs = {c:statistics.mean(x[c] for x in items) for c in FORCES + COMPONENTS}
        block_row = dict(case=name, gas=meta["gas"], body=meta["body"], Kn=meta["Kn"], block=block,
                         start_step_exclusive=begin, end_step_inclusive=end, start_s=begin*dt, end_s=end*dt,
                         winning_attempt=folder.name, solver_particles_at_commit=npart,
                         simulated_impacts_half=sum(x["simulated_impacts_half"] for x in items), **block_avgs)
        for component in ("Fx", "Fx_pressure", "Fx_shear"):
            block_row[component+"_full_N_per_m"] = 2 * block_row[component+"_half_N_per_m"]
        force_blocks.append(block_row)
    stats = {}
    for c in COMPONENTS:
        vals = [r[c] for r in force_blocks]
        stats.update({c+"_mean":statistics.mean(vals), c+"_block_sd":statistics.stdev(vals),
                      c+"_first_block":vals[0], c+"_last_block":vals[-1],
                      c+"_first5_mean":statistics.mean(vals[:5]), c+"_last5_mean":statistics.mean(vals[5:])})
    for c in FORCES:
        stats[c+"_mean"] = statistics.mean(r[c] for r in force_blocks)
    audit.update(case_sha256=meta["pin"]["case.json"], original_input_sha256=meta["pin"]["in.case"],
                 original_log_sha256=meta["pin"]["log_sha256"], parent_job=str(meta["pin"]["original_job"]),
                 parent_final_step=meta["base"], parent_final_simulated_particles=meta["np"],
                 verified_native_dump_header=FIELDS, step_seconds=dt, steps_per_subwindow=w//SUBS,
                 max_element_force_closure_N_per_m=max(r["max_element_closure_N_per_m"] for r in force_sub),
                 max_element_projection_residual_N_per_m=max(r["max_projection_residual_N_per_m"] for r in force_sub))
    summary = dict(case=name, gas=meta["gas"], body=meta["body"], Kn=meta["Kn"], blocks=BLOCKS,
                   native_subwindows=len(force_sub), sample_start_s=force_sub[0]["start_s"], sample_end_s=force_sub[-1]["end_s"],
                   half_surface_elements=len(meta["geometry"]), half_surface_length_m=math.fsum(meta["geometry"].values()),
                   simulated_impacts_half=sum(x["simulated_impacts_half"] for x in force_sub),
                   mass_kg=meta["mass"], n_inf_m3=meta["n_inf"], U_inf_m_s=meta["u"],
                   dynamic_pressure_Pa=meta["q"], projected_height_m=D, **stats)
    for component in ("Fx", "Fx_pressure", "Fx_shear"):
        summary[component+"_full_N_per_m_mean"] = 2 * summary[component+"_half_N_per_m_mean"]
    return force_sub, force_blocks, summary, audit


def shape_comparisons(cases):
    pairs = []
    lookup = {(row["gas"], row["Kn"], row["body"]): row for row in cases}
    for gas, kn in sorted({k[:2] for k in lookup}):
        a, b = (lookup.get((gas, kn, body)) for body in ("cylinder", "plate"))
        if a is None or b is None:
            continue
        require(near(a["dynamic_pressure_Pa"], b["dynamic_pressure_Pa"]), "Matched freestream pressures differ")
        row = dict(gas=gas, Kn=kn, cylinder_case=a["case"], plate_case=b["case"])
        for c in COMPONENTS:
            key = c + "_mean"
            row["cylinder_"+c] = a[key]
            row["plate_"+c] = b[key]
            row["plate_minus_cylinder_"+c] = b[key] - a[key]
        row["plate_over_cylinder_Cd"] = b["Cd_mean"] / a["Cd_mean"]
        row["plate_minus_cylinder_Cd_percent_of_cylinder"] = 100 * (b["Cd_mean"] / a["Cd_mean"] - 1)
        pairs.append(row)
    return pairs


METHOD = {
    "native_compute": "compute hits surf all gas n fx fy px py shx shy press, using the default flux normalization.",
    "native_averaging": "Each fix ave/surf output averages every solver step over one non-overlapping 0.06 ms window. Five equally long windows give one 0.30 ms block. Ten consecutive blocks follow 0.30 ms of restart settling.",
    "coordinates": "The reflected upper half uses the original oriented surface clipped at y=0. Ordinal surface IDs may be renumbered by clipping; the endpoint pairs define the physical segments.",
    "normalization": "Cd = 2 sum(Fx_half)/(0.5 m n_inf U_inf^2 D), D = 0.3048 m with unit span. fx and fy are forces per unit span. Pressure and shear components are signed stresses multiplied by the native line area. The reported force on the body is positive downstream. Fy_half is retained with its sign; full-body y force cancels by symmetry.",
    "impact_counts": "The n column is simulated impacts per solver step on each half-body element. Multiplication by the number of sampled solver steps recovers the integer number of simulated wall impacts.",
    "variability": "Block SD is the sample temporal standard deviation of ten consecutive block means. First/last denote the first or last complete block. First5/last5 are means over 1.5 ms. These summaries do not assume independent blocks.",
    "force_tolerance": "Elementwise x/y closure and normal/tangent direction checks use 2e-9 times the largest local component plus 2e-16 N/m to allow floating-point cancellation.",
    "source_documentation": ["https://sparta.github.io/doc/compute_surf.html","https://sparta.github.io/doc/fix_ave_surf.html","https://github.com/sparta/sparta/blob/20Jan2025/src/compute_surf.cpp","https://github.com/sparta/sparta/blob/20Jan2025/src/dump_surf.cpp","https://github.com/sparta/sparta/blob/20Jan2025/src/fix_ave_surf.cpp"],
}


def analyze(run, parent, output, names):
    run, parent, output = (Path(p).resolve() for p in (run, parent, output))
    pin_raw = (run / "pins.json").read_bytes()
    pins = json.loads(pin_raw)
    for shared, expected in pins["shared"].items():
        checked_bytes(parent / shared, expected)
    require(bool(names) and set(names) <= set(CASES) and len(names) == len(set(names)), "Invalid or duplicate case selection")
    sub, blocks, summary, audit = [], [], [], {}
    for name in names:
        events, native_blocks, totals, evidence = analyze_case(run, parent, name, pins)
        sub.extend(events)
        blocks.extend(native_blocks)
        summary.append(totals)
        audit[name] = evidence
    pairs = shape_comparisons(summary)
    rows_by_file = {"drag_subwindows.csv":sub, "drag_blocks.csv":blocks, "drag_summary.csv":summary,
                    "drag_shape_comparison.csv":pairs}
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output)) as temp:
        temp = Path(temp)
        hashes = {}
        for file, rows in rows_by_file.items():
            with (temp / file).open("w", newline="", encoding="utf-8") as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            hashes[file] = digest((temp / file).read_bytes())
        prov = dict(method=METHOD, case_order=names, all_eight_cases=set(names) == set(CASES), case_count=len(names),
                    block_count=len(blocks), native_dump_count=len(sub), pins_sha256=digest(pin_raw),
                    binary_sha256=pins["binary_sha256"], analysis_sha256=digest(Path(__file__).read_bytes()),
                    original_shared_file_sha256=pins["shared"], cases=audit, result_sha256=hashes)
        (temp / "drag_provenance.json").write_text(json.dumps(prov, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for file in list(rows_by_file) + ["drag_provenance.json"]:
            os.replace(str(temp / file), str(output / file))
    return prov


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=CASES, help="Repeat to audit a particular set of fully sampled cases")
    args = parser.parse_args()
    report = analyze(args.run_root, args.parent_root, args.out, args.case or list(CASES))
    print(f"{report['case_count']} cases, {report['block_count']} committed blocks, {report['native_dump_count']} native wall dumps")


if __name__ == "__main__":
    main()

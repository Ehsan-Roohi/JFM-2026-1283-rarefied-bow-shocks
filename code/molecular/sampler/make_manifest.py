"""Build the native read-only VDF fix contract from the checked station JSON."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def make_manifest(station_file, case_name, samples_per_block=100, cap=4096, salt=None):
    raw = Path(station_file).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    case = json.loads(raw.decode("utf-8-sig"))["cases"][case_name]
    if not (1 <= samples_per_block <= 100000 and 0 <= cap <= 16384):
        raise ValueError("Invalid block size or molecular retention cap")
    if salt is None:
        salt = int.from_bytes(hashlib.sha256((sha + case_name + ":vdf-v2").encode()).digest()[:8], "little")
    if not (0 <= salt < 2**64):
        raise ValueError("Salt must be an unsigned 64-bit integer")
    geometry, species = case["geometry"], case["gas"]
    if geometry not in {"cylinder","planar"} or species not in {"Ar","N2"}:
        raise ValueError("Unexpected native geometry or species")
    scale = float(case["upstream"]["u"])
    if scale <= 0 or list(case["global_streamwise_unit"]) != [1.0, 0.0, 0.0]:
        raise ValueError("Expected a finite positive upstream velocity along lab +x")
    def number(x):
        if not math.isfinite(float(x)):
            raise ValueError("Nonfinite station parameter")
        return format(float(x), ".17g")
    def row(values):
        return " ".join(number(x) for x in values)
    lines = [
        "JFM_VDF_2",
        f"case {case_name} {species} {2 if geometry=='cylinder' else 3} {int(species=='N2')} {sha}",
        f"sampling {samples_per_block} {cap} {salt}",
        f"frame {number(scale)} 0 0 0 1 0 0",
        "axes 256 -2 2 192 -1.5 1.5 96 -2 2 96 -1.5 1.5",
        f"stations {len(case['stations'])}",
    ]
    for station in case["stations"]:
        shape=station["shape"]
        bounds=shape["aabb_xyz_m"]
        kind={"oriented_rectangle_2d":"rect","planar_slab":"box"}[shape["type"]]
        if not station["id"].replace("_","").isalnum():
            raise ValueError("Invalid station identifier")
        vals = [*shape["center_xyz_m"],*shape["normal_unit"],*shape["tangent_unit"],
                shape["half_normal_m"],shape.get("half_tangent_m",0.0),
                *[v[0] for v in bounds],*[v[1] for v in bounds],shape["volume_m3"]]
        lines.append(f"station {station['id']} {kind} {row(vals)}")
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stations",type=Path,required=True)
    p.add_argument("--case",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples-per-block",type=int,default=100)
    p.add_argument("--molecular-cap",type=int,default=4096)
    p.add_argument("--salt",type=int)
    a=p.parse_args()
    result=make_manifest(a.stations,a.case,a.samples_per_block,a.molecular_cap,a.salt).encode()
    if a.output.exists():
        if a.output.read_bytes()!=result:
            raise FileExistsError(f"Different VDF manifest already exists at {a.output}")
    else:
        with a.output.open("xb") as f:
            f.write(result)
    print(a.output,hashlib.sha256(result).hexdigest())

if __name__=="__main__":
    main()

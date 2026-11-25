"""Export the measured laboratory velocity marginals of the two verified planar baselines."""
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "analysis"))
from vdf_report import WINDOWS, case_blocks, distribution_rows, pooled_values
from vdf_statistics import describe, file_sha


def main():
    compact = root / "analysis_compact"
    cases = ("Ar_planar", "N2_planar")
    rows, provenance = [], {}
    for case in cases:
        meta, blocks, status = case_blocks(compact, case, final=True)
        if len(blocks) != 10 or not status["target_receipt_sha256"]:
            raise ValueError(f"Incomplete case: {case}")
        np.testing.assert_allclose(meta["normal_xyz"], [[1.,0.,0.]] * len(meta["station_names"]))
        for window, indices in WINDOWS.items():
            selected = blocks[indices]
            data, snapshots = pooled_values(selected, 10)
            stats, axes = describe(meta, data, snapshots)
            for ix in (0,1):
                np.testing.assert_allclose(axes["incident"][ix], axes["normal"][ix], rtol=1e-12, atol=1e-15)
            result = distribution_rows(case, window, meta, stats, axes, selected)
            station_mass = {}
            for row in result:
                if row["axis"] != "normal":
                    continue
                row["axis"] = "laboratory_vx"
                row["Mach"] = 10
                row["nominal_Kn"] = 0.01
                row["time_start_s_exclusive"] = round(float(selected[0]["times_s"][0]) - 1.e-6, 9)
                row["time_end_s_inclusive"] = round(float(selected[-1]["times_s"][-1]), 9)
                row["snapshots_at_10us_stride"] = snapshots
                row["Uinf_m_s"] = meta["speed_scale_m_s"]
                station = row["station"]
                station_mass[station] = station_mass.get(station, 0.) + row["probability"]
                rows.append(row)
            if any(not math.isclose(s,1.,abs_tol=1e-12) for s in station_mass.values()) or len(station_mass)!=5:
                raise ValueError(f"Probability mass mismatch: {case}, {window}")
        provenance[case] = {"compact_status_sha256":file_sha(compact/case/"case_status.json"),
                            "native_target_sha256":status["target_receipt_sha256"],
                            "chain_sha256":status["chain_sha256"]}
    output = compact / "planar_velocity_distributions.csv"
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with tempfile.NamedTemporaryFile(dir=compact, mode="w", newline="",encoding="utf-8", delete=False) as f:
        wr=csv.DictWriter(f,fieldnames=columns);wr.writeheader();wr.writerows(rows);pending=Path(f.name)
    with pending.open(newline="",encoding="utf-8") as f:
        reread=list(csv.DictReader(f))
    if len(reread)!=len(rows) or {row["case"] for row in reread}!=set(cases):
        raise ValueError("CSV failed readback")
    os.replace(pending, output)
    record={"csv_sha256":file_sha(output),"rows":len(rows),"source":provenance,
            "case_scope":"Planar baselines only at Mach 10, nominal Kn=0.01, laboratory +x along inflow.",
            "time_scope":"Original native snapshots every 1 microsecond from 4.001 through 6.000 ms. Here every tenth snapshot is retained on the registered phase.",
            "fields":"Bin edges in vx/Uinf. Interval density per unit vx/Uinf; probability includes separately retained underflow and overflow. Conditional rotational energy (J) is reported only for nitrogen where a bin has support.",
            "analysis_scripts_sha256":{n:file_sha(root/"analysis"/n) for n in ("vdf_statistics.py","vdf_reduce.py","vdf_report.py")},
            "exporter_sha256":file_sha(Path(__file__))}
    receipt=output.with_suffix(".json")
    with tempfile.NamedTemporaryFile(dir=compact,mode="w",delete=False) as f:
        f.write(json.dumps(record,indent=2)+"\n");p=Path(f.name)
    os.replace(p,receipt)
    print(output, record["rows"], record["csv_sha256"])


if __name__ == "__main__":
    main()

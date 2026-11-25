"""Run on Unity in cases/: pack the mol.*.dat dumps of a film case into one npz
(per frame: id, x, y, vx[, erot]; float32/int64, frame offsets).
usage: python3 pack_molecules.py <case> [...]"""
import glob, json, pathlib, sys
import numpy as np

for case in sys.argv[1:]:
    d = pathlib.Path(case); m = json.loads((d / 'case.json').read_text())
    fs = sorted(glob.glob(str(d / 'mol.*.dat')), key=lambda f: int(f.split('.')[-2]))
    cols = {k: [] for k in ('id', 'x', 'y', 'vx', 'erot')}; off = [0]; steps = []
    for f in fs:
        q = np.loadtxt(f, skiprows=9, ndmin=2)
        steps.append(int(f.split('.')[-2]))
        cols['id'].append(q[:, 0].astype(np.int64)); cols['x'].append(q[:, 1].astype(np.float32))
        cols['y'].append(q[:, 2].astype(np.float32)); cols['vx'].append(q[:, 3].astype(np.float32))
        if q.shape[1] > 5: cols['erot'].append(q[:, 5].astype(np.float32))
        off.append(off[-1] + len(q))
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    np.savez(d.name + '_mol.npz', offsets=np.array(off), steps=np.array(steps), meta=json.dumps(m), **out)
    print(case, len(fs), 'frames', off[-1], 'molecules', flush=True)

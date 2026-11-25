"""Run on Unity in cases/: steady mean (last 40 outputs) of u, v on the full grid of a movie run -> <case>_uv.npz"""
import json, pathlib, sys
import numpy as np
for case in sys.argv[1:]:
    d = pathlib.Path(case); m = json.loads((d / 'case.json').read_text())
    nx, ny, x0 = m['nx'], m['ny'], m['domain'][0]
    U = np.zeros((ny, nx)); V = np.zeros((ny, nx)); N = np.zeros((ny, nx))
    for k in range(m['nframes'] - 39, m['nframes'] + 1):
        a = np.loadtxt(d / f"frame.{k * m['frame_steps']}.grid", skiprows=9)
        ix = np.clip(((a[:, 1] - x0) / m['dx']).astype(int), 0, nx - 1); iy = np.clip((a[:, 2] / m['dy']).astype(int), 0, ny - 1)
        ok = a[:, 4] > 0
        np.add.at(U, (iy[ok], ix[ok]), a[ok, 6]); np.add.at(V, (iy[ok], ix[ok]), a[ok, 7]); np.add.at(N, (iy[ok], ix[ok]), 1)
    with np.errstate(invalid='ignore'):
        np.savez_compressed(d.name + '_uv.npz', u=(U / N).astype(np.float32), v=(V / N).astype(np.float32), meta=json.dumps(m))
    print(case, 'ok', flush=True)

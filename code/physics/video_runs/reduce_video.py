"""Run on Unity inside cases/: reduce each movie case to one small npz.
Fields per frame on the uniform grid (float16): rho/rho_inf, T/T_inf, Trot/T_inf, (Pxx-Pyy)/p with p = n k T.
Particles in the stagnation strip: per frame, histograms along x of all and of upstream-running (vx<0) molecules.
usage: python3 reduce_video.py <case> [...]"""
import json, pathlib, sys
import numpy as np

KB = 1.380649e-23


def grid(path, nx, ny, x0, dx, dy):
    a = np.loadtxt(path, skiprows=9)
    ix = np.clip(((a[:, 1] - x0) / dx).astype(int), 0, nx - 1)
    iy = np.clip((a[:, 2] / dy).astype(int), 0, ny - 1)
    out = np.full((4, ny, nx), np.nan, np.float32)
    n, T = a[:, 4], a[:, 9]
    p = n * KB * np.where(T > 0, T, np.nan)
    out[0, iy, ix] = n
    out[1, iy, ix] = T
    out[2, iy, ix] = a[:, 8]
    out[3, iy, ix] = (a[:, 10] - a[:, 11]) / p
    return out


for case in sys.argv[1:]:
    d = pathlib.Path(case); m = json.loads((d / 'case.json').read_text())
    nx, ny, x0 = m['nx'], m['ny'], m['domain'][0]
    steps = [m['frame_steps'] * k for k in range(1, m['nframes'] + 1)]
    edges = np.linspace(x0, 0.0, 201)
    F, H, A, V = [], [], [], []
    for s in steps:
        g = grid(d / f'frame.{s}.grid', nx, ny, x0, m['dx'], m['dy'])
        g[0] /= m['nrho']; g[1] /= 200.0; g[2] /= 200.0
        k = int(np.ceil(nx / 480))
        if k > 1:          # block-average to at most 480 px wide
            gy, gx = ny // k * k, nx // k * k
            g = np.nanmean(g[:, :gy, :gx].reshape(4, gy // k, k, gx // k, k), axis=(2, 4))
        F.append(g.astype(np.float16))
        pf = d / f'part.{s}.dat'
        try:
            q = np.loadtxt(pf, skiprows=9, ndmin=2)
        except Exception:
            q = np.zeros((0, 5))
        x, vx = (q[:, 1], q[:, 3]) if len(q) else (np.zeros(0), np.zeros(0))
        A.append(np.histogram(x, edges)[0]); H.append(np.histogram(x[vx < 0], edges)[0])
        V.append(np.histogram(x[vx < 0], edges, weights=vx[vx < 0])[0])
    np.savez_compressed(d.name + '.npz', fields=np.array(F), all=np.array(A), hot=np.array(H), hot_vx=np.array(V),
                        edges=edges, t=np.array(steps) * m['dt'], meta=json.dumps(m))
    print(case, 'done', flush=True)

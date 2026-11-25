"""Run on Unity in cases/: v_x distributions on the stagnation strip of a movie run.
Stations: 10/50/90% of the ideal Mach-10 normal-shock density rise (as the planar/cylinder
samplers), located on the frame-averaged stagnation-row density; 0.8 mm slabs (or one cell).
Also a 'wall side' station 2 slab widths... no: at the last cell before the body.
usage: python3 strip_vdf.py <case> <first_frame_index> -> <case>_vdf.npz"""
import json, pathlib, sys
import numpy as np

RH = {'Ar': 3.884, 'N2': 5.714}
d = pathlib.Path(sys.argv[1]); k0 = int(sys.argv[2])
m = json.loads((d / 'case.json').read_text())
steps = [m['frame_steps'] * k for k in range(1, m['nframes'] + 1)][k0:]
# stagnation-row density, frame-averaged
acc = None
for s in steps:
    a = np.loadtxt(d / f'frame.{s}.grid', skiprows=9)
    a = a[a[:, 2] < m['dy']]
    o = np.argsort(a[:, 1]); a = a[o]
    acc = a[:, 4] if acc is None else acc + a[:, 4]
x = a[:, 1]; rho = acc / len(steps) / m['nrho']
x, rho = x[x < 0], rho[x < 0]
st = {}
for f in (0.1, 0.5, 0.9):
    lev = 1 + f * (RH[m['gas']] - 1)
    st[f'{int(f*100)}%'] = x[np.argmax(rho >= lev)]
st['wall side'] = -0.035      # same wall distance as the cylinder sampler
half = max(0.4e-3, 0.5 * m['dx'])
edges = np.linspace(-1.5, 1.6, 311)
H = {k: np.zeros(310) for k in st}; N = {k: 0 for k in st}
for s in steps:
    q = np.loadtxt(d / f'part.{s}.dat', skiprows=9, ndmin=2)
    for k, xs in st.items():
        sel = np.abs(q[:, 1] - xs) <= half
        H[k] += np.histogram(q[sel, 3] / m['U'], edges)[0]; N[k] += int(sel.sum())
np.savez(d.name + '_vdf.npz', edges=edges, x_rho=x, rho=rho, meta=json.dumps(m),
         stations=json.dumps({k: float(v) for k, v in st.items()}), counts=json.dumps(N),
         **{'H_' + k: H[k] for k in st})
print(d.name, {k: round(float(v), 4) for k, v in st.items()}, N)

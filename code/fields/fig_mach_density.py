"""Figure 5: density field rho/rho_inf of the Mach-10 members of the Mach sweep
(Kn ~ 0.01), (a) argon and (b) nitrogen.  Two identical frames, the colour scale of
the Knudsen-sweep figures.

The Mach sweep has no DS2VD.DAT next to its field files.  Its freestream number density
is the one stated in the paper (section 2): lambda_inf/D = 0.01 with the hard-sphere mean
free path 1/(sqrt(2) pi d_ref^2 n_inf), d_ref = 4.17e-10 m, i.e. n_inf = 4.2467e20 m^-3
for both gases.  The script checks this against the undisturbed upstream cells.

usage: python -I fig_mach_density.py
"""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import *

ML = ROOT / 'mach'
# argon: Triangulatedgrid/TriangulateGridM10.dat extracted (unchanged) from ML/'Argon-Data-Mach5-15.rar'
SRC = {'Ar': ML / 'Ar' / 'TriangulateGridM10.dat',
       'N2': ML / 'N2' / 'Ma=10' / 'DS2FF.DAT'}
N_INF = 1.0 / (np.sqrt(2.0) * np.pi * 4.17e-10 ** 2 * 0.01 * D)
WIN, XT, YT = (-0.575, 2.05, 1.25), 0.5, 0.5


def read_tri(path):
    """Tecplot FE-triangle export of a DS2V field: node table only."""
    txt = open(path, encoding='latin-1').read()
    head, rest = txt.split('ZONE', 1)
    names = [k for k in re.findall(r'"([^"]*)"', head) if k]
    n = int(re.search(r'Nodes=\s*(\d+)', rest).group(1))
    body = rest[rest.index('\n', rest.index('DT=(')):].split(None, n * len(names) + 1)[: n * len(names)]
    vals = np.array([float(t) for t in body]).reshape(n, len(names))
    return {k: vals[:, i] for i, k in enumerate(names)}


fig, axs, cax = make_figure(1)
for i, (ax, gas, lab) in enumerate(zip(axs, ('Ar', 'N2'), ('Argon', 'Nitrogen'))):
    d = read_tri(SRC[gas]) if gas == 'Ar' else read_ds2ff(SRC[gas])
    up = d['X'] / D < -0.5
    n_up = d['ND'][up].mean()
    assert abs(n_up / N_INF - 1) < 0.005, (gas, n_up, N_INF)       # field agrees with the stated n_inf
    cf, Z = density_panel(ax, d, N_INF, WIN)
    style_panel(ax, WIN, XT, YT, 'ab'[i], lab, xlabel=True, ylabel=i == 0)
    print(gas, SRC[gas], f'n_inf={N_INF:.5g}  upstream mean ND={n_up:.5g} ({100 * (n_up / N_INF - 1):+.2f} %)',
          f'U_up={np.hypot(d["U"], d["V"])[up].mean():.1f} m/s', 'window', WIN, f'min {Z.min():.3f} max {Z.max():.2f}')
density_colorbar(fig, cf, cax, label_every=2)
save(fig, 'Mach_density_Ar_N2')

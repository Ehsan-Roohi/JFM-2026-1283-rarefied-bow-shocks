"""OPTIONAL companion for the supplement (section S3): the Mach-5 fields and the highest
Mach number of each sweep (argon Mach 15, nitrogen Mach 14) in the same style and on the
same colour levels as the Mach-10 figure, because the supplement says these fields are
'on the colour scale of' that figure.

usage: python -I fig_mach_density_supp.py
"""
import sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import *

ML = ROOT / 'mach'
TRI = ML / 'Ar'          # extracted from ML/'Argon-Data-Mach5-15.rar'
N2 = ML / 'N2'
PANELS = [('Ar', 5, TRI / 'TriangulateGridM5.dat'), ('N2', 5, N2 / 'Ma=5' / 'DS2FF.DAT'),
          ('Ar', 15, TRI / 'TriangulatedGridM15.dat'), ('N2', 14, N2 / 'Ma=14' / 'DS2FF.DAT')]
N_INF = 1.0 / (np.sqrt(2.0) * np.pi * 4.17e-10 ** 2 * 0.01 * D)
A_INF = {'Ar': np.sqrt(5 / 3 * 1.380649e-23 * 200 / 6.63e-26), 'N2': np.sqrt(1.4 * 1.380649e-23 * 200 / 4.65e-26)}
WIN, XT, YT = (-0.575, 2.05, 1.25), 0.5, 0.5


def read_tri(path):
    txt = open(path, encoding='latin-1').read()
    head, rest = txt.split('ZONE', 1)
    names = [k for k in re.findall(r'"([^"]*)"', head) if k]
    n = int(re.search(r'Nodes=\s*(\d+)', rest).group(1))
    body = rest[rest.index('\n', rest.index('DT=(')):].split(None, n * len(names) + 1)[: n * len(names)]
    vals = np.array([float(t) for t in body]).reshape(n, len(names))
    return {k: vals[:, i] for i, k in enumerate(names)}


fig, axs, cax = make_figure(2)
for i, (ax, (gas, mach, src)) in enumerate(zip(axs, PANELS)):
    d = read_tri(src) if src.suffix == '.dat' else read_ds2ff(src)
    up = d['X'] / D < -0.5
    n_up, u_up = d['ND'][up].mean(), np.hypot(d['U'], d['V'])[up].mean()
    assert abs(n_up / N_INF - 1) < 0.005, (gas, mach, n_up)
    assert abs(u_up / A_INF[gas] / mach - 1) < 0.01, (gas, mach, u_up)          # the file is the stated Mach number
    cf, Z = density_panel(ax, d, N_INF, WIN, allow_over=True)
    lab = ('Ar' if gas == 'Ar' else r'N$_2$') + rf', $M_\infty={mach}$'
    style_panel(ax, WIN, XT, YT, 'abcd'[i], lab, xlabel=i >= 2, ylabel=i % 2 == 0)
    print(gas, mach, src, f'upstream ND {n_up:.5g} ({100 * (n_up / N_INF - 1):+.2f} %)  U {u_up:.1f} (M = {u_up / A_INF[gas]:.2f})',
          f'min {Z.min():.3f} max {Z.max():.2f}')
density_colorbar(fig, cf, cax)
save(fig, 'Supp_Mach_density_M5_Mmax')

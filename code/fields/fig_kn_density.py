"""Figures 3 and 4: density field rho/rho_inf of the Mach-10 Knudsen sweep, argon and nitrogen.
One composite PDF per gas: 2 x 2 identical frames, one shared colour bar.

usage: python -I fig_kn_density.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import *

# window (x0, x1, y1) in units of D, x tick step, y tick step; width/height = 2.1 in every panel.
# Each window lies inside the DS2V output domain of its case (no blank regions).
WIN = {0.01:  ((-0.575, 2.05, 1.25), 0.5, 0.5),
       0.025: ((-0.90, 2.25, 1.50), 0.5, 0.5),
       0.1:   ((-1.60, 3.02, 2.20), 1.0, 1.0),
       1.0:   ((-3.90, 4.92, 4.20), 2.0, 2.0)}
KNS = [0.01, 0.025, 0.1, 1.0]

for gas, name in (('Ar', 'Kn_density_Ar'), ('N2', 'Kn_density_N2')):
    fig, axs, cax = make_figure(2)
    for i, (ax, kn) in enumerate(zip(axs, KNS)):
        d, ninf, p = load_case(gas, kn)
        win, xt, yt = WIN[kn]
        cf, Z = density_panel(ax, d, ninf, win)
        style_panel(ax, win, xt, yt, 'abcd'[i], rf'$Kn_\infty={kn:g}$', xlabel=i >= 2, ylabel=i % 2 == 0)
        print(gas, kn, p, f'n_inf={ninf:.4g}', 'window', win, f'min {np.nanmin(Z):.3f} max {np.nanmax(Z):.2f}')
    density_colorbar(fig, cf, cax)
    save(fig, name)
    plt.close(fig)

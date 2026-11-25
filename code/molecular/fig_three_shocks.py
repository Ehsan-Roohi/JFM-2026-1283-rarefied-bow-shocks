"""Planar shock vs cylinder vs flat plate: v_n distributions at 10/50/90% of the density rise
and on the wall side, Mach 10, Kn 0.01, argon and nitrogen.
The planar sampler uses streamwise velocity and the cylinder sampler uses inward radial
velocity on the 170-degree ray over 4-6 ms. The publication takes the plate histograms
from the independent steady high-particle stagnation-line run in strip_final.
usage: python fig_three_shocks.py <K_dir> <strip_npz_dir> <out_dir>"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hot_cold_vdf as H

K, S, out = map(pathlib.Path, sys.argv[1:4])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.linewidth': 0.7})
planar = H.load_planar(str(K / 'analysis_compact/planar_velocity_distributions.csv'))
cyl = {}
for g in ('Ar', 'N2'):
    cyl.update(H.load_cylinder(str(K / f'cylinder_distributions/{g}_cylinder_final/velocity_distributions.csv'), g, 10)[0])
STN = [('s10', '10%'), ('s50', '50%'), ('s90', '90%'), ('downstream_reference', 'wall side')]
fig, ax = plt.subplots(2, 4, figsize=(10, 4.8))
hot = []
for i, g in enumerate(('Ar', 'N2')):
    z = np.load(S / f'{g}_plate_Kn0p01_vdf.npz')
    e = z['edges']; xc = 0.5 * (e[1:] + e[:-1]); w = e[1] - e[0]
    for j, (st, key) in enumerate(STN):
        a = ax[i, j]
        for src, ls, col, lab in ((planar, '--', '0.45', 'planar shock'), (cyl, '-', 'k', 'cylinder')):
            lo, hi, p = src[(g, st)]
            f = np.isfinite(lo) & np.isfinite(hi)
            a.plot(0.5 * (lo[f] + hi[f]), H.smooth(p[f] / (hi[f] - lo[f])), ls, color=col, lw=1.0, label=lab)
            hot.append((g, key, lab, H.stats(g, lo, hi, p)['hot_fraction']))
        h = z['H_' + key].astype(float)
        from scipy.ndimage import gaussian_filter1d
        h = gaussian_filter1d(h, float(__import__('os').environ.get('PLATE_SIGMA', 3.5)), mode='constant')
        pdf = h / (h.sum() * w)
        a.plot(xc, pdf, '-.', color='tab:blue', lw=1.2, label='flat plate')
        hot.append((g, key, 'flat plate', h[xc < 0].sum() / h.sum()))
        a.axvline(0, color='0.6', lw=0.5); a.set_xlim(-1.5, 1.6)
        a.text(0.97, 0.95, f'({"abcdefgh"[4*i+j]})', transform=a.transAxes, ha='right', va='top')
        if i == 0: a.set_title(f'{key} of density rise' if '%' in key else key)
        if i == 1: a.set_xlabel(r'$v_n/U_\infty$')
        if j == 0: a.set_ylabel(('Argon' if g == 'Ar' else 'Nitrogen') + '\n' + r'PDF of $v_n/U_\infty$')
ax[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(out / 'vdf_three_shocks.pdf'); fig.savefig(out / 'vdf_three_shocks.png', dpi=200)
with open(out / 'vdf_three_shocks_hot.csv', 'w') as fh:
    fh.write('gas,station,geometry,hot_fraction\n')
    for r in hot:
        fh.write(f'{r[0]},{r[1]},{r[2]},{r[3]:.4f}\n'); print(*r)

"""Contours of the normal-stress anisotropy (P_xx - P_yy)/p around the cylinder and the flat plate,
argon and nitrogen, Kn 0.01 and 0.1; steady average of the last 40 outputs of the high-particle runs.
-> analysis/aniso_contours.png/.pdf"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.linewidth': 0.7})
D = 0.3
CASES = [('Ar', 'cylinder'), ('Ar', 'plate'), ('N2', 'cylinder'), ('N2', 'plate')]
fig, ax = plt.subplots(2, 4, figsize=(11, 4.3), constrained_layout=True)
for i, kn in enumerate(('0p01', '0p1')):
    win = 1.0 if kn == '0p01' else 2.5
    for j, (g, b) in enumerate(CASES):
        z = np.load(f'npz_hi/hi_{g}_{b}_Kn{kn}.npz'); m = json.loads(str(z['meta']))
        A = np.nanmean(z['fields'][-40:, 3].astype(np.float32), axis=0)
        w = np.isfinite(A).astype(float); A0 = np.nan_to_num(A)
        A = np.where(w > 0, gaussian_filter(A0 * w, 1.0) / np.maximum(gaussian_filter(w, 1.0), 1e-6), np.nan)
        k = int(np.ceil(m['nx'] / 480)); dx, dy = m['dx'] * k, m['dy'] * k
        ext = [m['domain'][0] / D, (m['domain'][0] + A.shape[1] * dx) / D, 0, A.shape[0] * dy / D]
        a = ax[i, j]; a.set_facecolor('0.35')
        im = a.imshow(A, origin='lower', extent=ext, cmap='RdBu_r', vmin=-1.5, vmax=1.5, aspect='equal')
        X = np.linspace(ext[0], ext[1], A.shape[1]); Y = np.linspace(ext[2], ext[3], A.shape[0])
        a.contour(X, Y, np.nan_to_num(A), levels=[0.5, 1.0], colors='k', linewidths=0.5)
        uvp = pathlib.Path(f'uv_hi/hi_{g}_{b}_Kn{kn}_uv.npz')
        if uvp.exists():                                   # streamlines of the steady mean flow
            q = np.load(uvp); u, v = q['u'], q['v']; s = max(1, u.shape[1] // 300)
            u, v = u[::s, ::s], v[::s, ::s]; solid = ~np.isfinite(u)
            u, v = np.where(solid, 0, u), np.where(solid, 0, v)
            xs = (m['domain'][0] + (np.arange(u.shape[1]) + 0.5) * m['dx'] * s) / D
            ys = ((np.arange(u.shape[0]) + 0.5) * m['dy'] * s) / D
            a.streamplot(xs, ys, np.ma.array(u, mask=solid), np.ma.array(v, mask=solid), density=1.2,
                         color='0.15', linewidth=0.45, arrowsize=0.5)
        a.set_xlim(-win, ext[1]); a.set_ylim(0, min(win + 0.3, ext[3]))
        if i == 0: a.set_title(f"{'argon' if g == 'Ar' else 'nitrogen'}, {'cylinder' if b == 'cylinder' else 'flat plate'}")
        if j == 0: a.set_ylabel(f"$Kn_\\infty$ = {kn.replace('p', '.')}\n$y/D$")
        if i == 1: a.set_xlabel('$x/D$')
        a.text(0.03, 0.93, f'({"abcdefgh"[4*i+j]})', transform=a.transAxes, va='top', color='k')
fig.colorbar(im, ax=ax, shrink=0.8, label=r'$(P_{xx}-P_{yy})/p$')
fig.savefig('analysis/aniso_contours.png', dpi=200); fig.savefig('analysis/aniso_contours.pdf')
print('ok')

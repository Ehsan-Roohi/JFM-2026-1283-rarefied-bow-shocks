"""Stagnation-line T, rho and u for argon/nitrogen x cylinder/plate on one graph (Kn 0.01 and 0.1),
steady part (last 40 frames) of the high-particle time-resolved runs.  -> analysis/four_cases_stagnation.png"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.linewidth': 0.7})
D = 0.3
STY = {('Ar', 'cylinder'): ('k', '-'), ('Ar', 'plate'): ('k', '--'), ('N2', 'cylinder'): ('tab:red', '-'),
       ('N2', 'plate'): ('tab:red', '--')}
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex='row')
for i, kn in enumerate(('0p01', '0p1')):
    for (g, b), (c, ls) in STY.items():
        z = np.load(f'npz_hi/hi_{g}_{b}_Kn{kn}.npz'); m = json.loads(str(z['meta']))
        F = z['fields'][-40:].astype(np.float32); k = int(np.ceil(m['nx'] / 480))
        x = m['domain'][0] + (np.arange(F.shape[-1]) + 0.5) * m['dx'] * k
        row = np.nanmean(F[:, :, :2, :], axis=(0, 2)); up = x < 0
        lab = f"{'argon' if g == 'Ar' else 'nitrogen'}, {'cylinder' if b == 'cylinder' else 'flat plate'}"
        ax[i, 0].plot(x[up] / D, row[1, up], ls, color=c, lw=1.2, label=lab)
        ax[i, 1].plot(x[up] / D, row[0, up], ls, color=c, lw=1.2)
        # u/U from the momentum: not stored; use the mass-flux-free proxy rho*T? -> use particle hot fraction instead
        A, H = z['all'][-40:].sum(0).astype(float), z['hot'][-40:].sum(0).astype(float)
        e = z['edges']; xm = 0.5 * (e[1:] + e[:-1]); nb = 4; n = len(xm) // nb * nb
        A, H, xm = A[-n:].reshape(-1, nb).sum(1), H[-n:].reshape(-1, nb).sum(1), xm[-n:].reshape(-1, nb).mean(1)
        ok = A > 200
        ax[i, 2].plot(xm[ok] / D, 100 * H[ok] / A[ok], ls, color=c, lw=1.2)
    ax[i, 0].set_ylabel(f"$Kn_\\infty$ = {kn.replace('p', '.')}\n$T/T_\\infty$")
    ax[i, 1].set_ylabel(r'$\rho/\rho_\infty$'); ax[i, 1].set_ylim(0, 12)
    ax[i, 2].set_ylabel(r'molecules with $v_x<0$ (%)'); ax[i, 2].set_ylim(0, 60)
    for a in ax[i]: a.set_xlim(-1.0 if kn == '0p01' else -2.5, 0); a.axvline(0, color='0.6', lw=0.6)
for a in ax[1]: a.set_xlabel('distance from the body, $x/D$ (stagnation line)')
ax[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
for n_, a in enumerate(ax.flat): a.text(0.97, 0.94, f'({"abcdef"[n_]})', transform=a.transAxes, ha='right', va='top')
fig.tight_layout(); fig.savefig('analysis/four_cases_stagnation.png', dpi=200); fig.savefig('analysis/four_cases_stagnation.pdf')
print('ok')

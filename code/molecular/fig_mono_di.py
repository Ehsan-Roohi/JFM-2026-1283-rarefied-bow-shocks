"""Monatomic vs diatomic: argon and nitrogen distributions on the cylinder stagnation line on
common axes (log scale shows the upstream-running tail), plus T_rot of hot and cold N2.
usage: python fig_mono_di.py <K_dir> <out_dir>"""
import csv, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hot_cold_vdf as H

K, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.linewidth': 0.7})
cyl = {}
for g in ('Ar', 'N2'):
    cyl.update(H.load_cylinder(str(K / f'cylinder_distributions/{g}_cylinder_final/velocity_distributions.csv'), g, 10)[0])
fig, ax = plt.subplots(1, 4, figsize=(10, 3.0), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1]))
C = {'Ar': ('k', '-', 'argon'), 'N2': ('tab:red', '--', 'nitrogen')}
for j, st in enumerate(['s10', 's50', 's90']):
    a = ax[j]
    for g in ('Ar', 'N2'):
        lo, hi, p = cyl[(g, st)]
        f = np.isfinite(lo) & np.isfinite(hi)
        x = 0.5 * (lo[f] + hi[f]); pdf = H.smooth(p[f] / (hi[f] - lo[f]), 2.5)
        a.semilogy(x, pdf, C[g][1], color=C[g][0], lw=1.1, label=C[g][2])
        s = H.stats(g, *cyl[(g, st)])
        a.text(0.03, 0.95 - 0.09 * (g == 'N2'), f"{C[g][2]}: {100*s['hot_fraction']:.1f}% hot",
               transform=a.transAxes, color=C[g][0], fontsize=8, va='top')
    a.axvspan(-1.6, 0, color='tab:red', alpha=0.06, lw=0)
    a.axvline(0, color='0.6', lw=0.5)
    a.set(xlim=(-1.5, 1.6), ylim=(1e-3, 5), xlabel=r'$v_n/U_\infty$', title=f'{H.LABEL[st]} of density rise')
    a.text(0.97, 0.95, f'({"abc"[j]})', transform=a.transAxes, ha='right', va='top')
ax[0].set_ylabel(r'PDF of $v_n/U_\infty$')
ax[2].legend(frameon=False, fontsize=8, loc='lower center')
rows = list(csv.DictReader(open(out / 'blocks_N2_cylinder.csv')))
lab = {'upstream_reference': 'up', 's10': '10%', 's50': '50%', 's90': '90%', 'downstream_reference': 'wall'}
n = range(len(rows))
a = ax[3]
a.plot(n, [float(r['Trot_hot_K']) if r['Trot_hot_K'] != 'nan' else np.nan for r in rows], 'o-', color='tab:red', ms=4, label=r'$T_{\rm rot}$, hot ($v_n<0$)')
a.plot(n, [float(r['Trot_cold_K']) for r in rows], 's--', color='tab:blue', ms=4, mfc='w', label=r'$T_{\rm rot}$, cold ($v_n\geq0$)')
a.set_xticks(list(n), [lab[r['station']] for r in rows])
a.set(xlabel='station', ylabel=r'$T_{\rm rot}$ (K)', title='nitrogen rotation')
a.legend(frameon=False, fontsize=8, loc='lower right')
a.text(0.03, 0.95, '(d)', transform=a.transAxes, va='top')
fig.tight_layout()
fig.savefig(out / 'vdf_mono_di_cylinder.pdf'); fig.savefig(out / 'vdf_mono_di_cylinder.png', dpi=200)
print('ok')

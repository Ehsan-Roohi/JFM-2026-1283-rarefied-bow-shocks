"""Paper figure: cylinder vs flat plate on the stagnation line at equal Kn and at
equal room, in units of lambda_inf, aligned at each profile's 10 % point.
Rows: argon density, argon temperature, nitrogen density, nitrogen temperature.
Columns: both bodies at Kn = 0.05 | cylinder Kn = 0.05 with plate Kn = 0.1.

usage: python fig_equal_room_paper.py <campaign_dir> <planar_dir> <out_pdf>
"""
import os, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_curvature import PLANAR, case_profile, crossing  # noqa: E402

camp, pdir, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 10})
CYL, PLA = '#1f4e9c', '#c0392b'
LAM001 = {'Ar': 3.0472e-3, 'N2': 3.0488e-3}


def load(gas, body, kn):
    tag = f'{kn:g}'.replace('.', 'p')
    meta, p = case_profile(str(camp / 'cases' / f'{gas}_{body}_Kn{tag}'))
    pl = PLANAR[gas]
    lam = meta['lambda_m']
    s10 = crossing(p['s'], p['rho'], 1 + 0.1 * (pl['ratio'] - 1))
    return dict(xi=(s10 - p['s']) / lam, rho=p['rho'], T=p['T'], wall=s10 / lam, room=s10 / lam / pl['w10_90'])


def planar(gas):
    z = np.load(pdir / f'{gas}_planar_long.npz')
    x, m = z['x'], z['mean']
    rho, T = m[:, 0] / m[0, 0], m[:, 2] / 200.0
    lv = 1 + 0.1 * (PLANAR[gas]['ratio'] - 1)
    x10 = next(x[i - 1] + (lv - rho[i - 1]) / (rho[i] - rho[i - 1]) * (x[i] - x[i - 1])
               for i in range(1, len(rho)) if rho[i - 1] < lv <= rho[i])
    return (x - x10) / LAM001[gas], rho, T


fig = plt.figure(figsize=(7.2, 8.0))
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 0.42, 1, 1], hspace=0.07, wspace=0.17,
                      left=0.085, right=0.985, top=0.965, bottom=0.06)
axs = np.empty((4, 2), dtype=object)
for c in range(2):
    for g, r0 in enumerate((0, 3)):
        axs[2 * g + 1, c] = fig.add_subplot(gs[r0 + 1, c])
        axs[2 * g, c] = fig.add_subplot(gs[r0, c], sharex=axs[2 * g + 1, c])
        axs[2 * g, c].tick_params(labelbottom=False)
letters = iter('abcdefgh')
for g, gas in enumerate(('Ar', 'N2')):
    xp, rp, Tp = planar(gas)
    for c, pair in enumerate(((('cylinder', 0.05), ('plate', 0.05)), (('cylinder', 0.05), ('plate', 0.1)))):
        axr, axT = axs[2 * g, c], axs[2 * g + 1, c]
        xmax = 0
        for (body, kn), col, ls in zip(pair, (CYL, PLA), ('-', '--')):
            d = load(gas, body, kn)
            m = d['xi'] <= d['wall'] + 1e-9
            lab = (('cylinder' if body == 'cylinder' else 'flat plate')
                   + rf', $Kn_\infty={kn:g}$, $s_{{10}}/w_p={d["room"]:.2f}$')
            axr.plot(d['xi'][m], d['rho'][m], ls, color=col, lw=1.4, label=lab)
            axT.plot(d['xi'][m], d['T'][m], ls, color=col, lw=1.4)
            for a in (axr, axT):
                a.axvline(d['wall'], color=col, ls=':', lw=0.9)
            xmax = max(xmax, d['wall'])
        for a, q in ((axr, rp), (axT, Tp)):
            a.plot(xp, q, color='k', ls=(0, (4, 2)), lw=1.0, zorder=1, label='planar shock' if a is axr else None)
            a.set_xlim(-12, xmax * 1.06 + 1.5)
        axr.set_ylim(0.5, 14)
        axr.legend(fontsize=7.2, loc='upper left', bbox_to_anchor=(0.0, 0.90), frameon=False, handlelength=2.2)
        gname = 'argon' if gas == 'Ar' else 'nitrogen'
        axr.set_title(f'{gname}, ' + (r'equal $Kn_\infty=0.05$' if c == 0 else 'equal room'), fontsize=10)
        axT.set_xlabel(r'distance downstream of the 10% point, $/\lambda_\infty$', fontsize=9)
        if c == 0:
            axr.set_ylabel(r'$\rho/\rho_\infty$')
            axT.set_ylabel(r'$T_{\rm tr}/T_\infty$')
for a in axs.flat:
    a.text(0.02, 0.95, f'({next(letters)})', transform=a.transAxes, va='top', ha='left', fontsize=10)
    a.tick_params(labelsize=8.5)
fig.savefig(out)
fig.savefig(out.with_suffix('.png'), dpi=150)
print('wrote', out)

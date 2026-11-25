"""Paper figure: molecular velocity distributions through the planar Mach-10 shock,
argon (monatomic) against nitrogen (diatomic), at five stations from upstream to
downstream.  Molecules with v_x < 0 ("hot", running upstream) are shaded.
Panel (f): hot fraction and hot share of the fourth central moment.

usage: python fig_vdf_planar_paper.py planar_velocity_distributions.csv OUT_PDF
"""
import pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hot_cold_vdf import load_planar, stats, STATIONS  # noqa: E402

src, out = sys.argv[1], pathlib.Path(sys.argv[2])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 10, 'axes.linewidth': 0.7,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
BLUE, RED = '#1f4e9c', '#c0392b'
GAS = {'Ar': dict(color=BLUE, ls='-', name='argon'), 'N2': dict(color=RED, ls=(0, (5, 2)), name='nitrogen')}
TITLE = {'upstream_reference': 'upstream', 's10': '10% of the density rise', 's50': '50%', 's90': '90%',
         'downstream_reference': 'downstream'}
data = load_planar(src)
st = {(g, s): stats(g, *data[(g, s)]) for g in GAS for s in STATIONS}

fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.5))
for k, s in enumerate(STATIONS):
    ax = axs.flat[k]
    ax.axvspan(-1.3, 0, color='0.9', lw=0, zorder=0)
    for g, sty in GAS.items():
        lo, hi, p = data[(g, s)]
        fin = np.isfinite(lo) & np.isfinite(hi)
        v, pdf = 0.5 * (lo[fin] + hi[fin]), p[fin] / (hi[fin] - lo[fin])
        pdf = np.where(pdf >= 1e-3, pdf, np.nan)       # below this the bins hold only a few visits
        ax.semilogy(v, pdf, color=sty['color'], ls=sty['ls'], lw=1.4, label=sty['name'])
    ax.set_xlim(-1.3, 1.9); ax.set_ylim(1e-3, 20)
    ax.set_xticks([-1, 0, 1])
    ax.text(0.03, 0.95, f'({"abcde"[k]}) {TITLE[s]}', transform=ax.transAxes, va='top', fontsize=9.5)
    if s != 'upstream_reference':
        ax.text(0.03, 0.80, 'hot:', transform=ax.transAxes, va='top', fontsize=8.5, color='0.25')
        for j, g in enumerate(GAS):
            ax.text(0.03, 0.70 - 0.1 * j, f"{100 * st[(g, s)]['hot_fraction']:.1f}%", transform=ax.transAxes,
                    va='top', fontsize=8.5, color=GAS[g]['color'])
    if k >= 3:
        ax.set_xlabel(r'$v_x/U_\infty$')
    if k % 3 == 0:
        ax.set_ylabel(r'$g\,U_\infty$')
    else:
        ax.tick_params(labelleft=False)
axs[0, 0].legend(frameon=False, fontsize=9, loc='center left', bbox_to_anchor=(0.0, 0.55), handlelength=2.2)
axs[0, 0].text(0.30, 0.10, r'$v_x<0$', transform=axs[0, 0].transAxes, ha='right', va='bottom', fontsize=8.5, color='0.3')

ax = axs[1, 2]
x = np.arange(len(STATIONS))
for g, sty in GAS.items():
    ax.plot(x, [100 * st[(g, s)]['hot_fraction'] for s in STATIONS], '-o', color=sty['color'], ms=5, lw=1.2,
            label=f"{sty['name']}, hot fraction")
    share = [100 * st[(g, s)]['hot_share_c4'] if st[(g, s)]['hot_fraction'] > 1e-3 else np.nan for s in STATIONS]
    ax.plot(x, share, '--s', color=sty['color'], mfc='white', ms=5, lw=1.2, label=f"{sty['name']}, share of $\\int c^4 g$")
ax.set_xticks(x); ax.set_xticklabels(['up', '10%', '50%', '90%', 'down'], fontsize=8.5)
ax.set_ylim(0, 108); ax.set_ylabel('per cent', labelpad=1)
ax.yaxis.set_label_position('right'); ax.yaxis.tick_right(); ax.tick_params(labelleft=False, labelright=True)
ax.set_xlabel('station')
ax.text(0.03, 0.95, '(f)', transform=ax.transAxes, va='top', fontsize=9.5)
ax.text(2.25, 84, 'hot share of\n' + r'$\int c^4 g\,{\rm d}v_x$', fontsize=8.5, color='0.25', ha='left', va='center', linespacing=1.1)
ax.text(2.2, 10, 'hot fraction', fontsize=8.5, color='0.25', ha='left', va='center')
fig.tight_layout(w_pad=0.5, h_pad=0.6)
fig.savefig(out); fig.savefig(out.with_suffix('.png'), dpi=170)
for g in GAS:
    for s in STATIONS:
        print(g, s, {k: round(v, 4) for k, v in st[(g, s)].items()})

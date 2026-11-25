"""Paper figure: front width relative to the planar shock against the room s10/w_p,
all four series (Ar/N2 x cylinder/plate).  Large fonts, one legend above the panels.
usage: python fig_truncation_paper.py <campaign_dir> <out_pdf>"""
import json, pathlib, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d, out = pathlib.Path(sys.argv[1]) / 'results', pathlib.Path(sys.argv[2])
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 11, 'axes.linewidth': 0.8,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
BLUE, RED = '#1f4e9c', '#c0392b'
style = {('Ar', 'cylinder'): dict(marker='o', color=BLUE, mfc=BLUE, label='argon, cylinder'),
         ('Ar', 'plate'): dict(marker='s', color=BLUE, mfc='white', label='argon, flat plate'),
         ('N2', 'cylinder'): dict(marker='o', color=RED, mfc=RED, label='nitrogen, cylinder'),
         ('N2', 'plate'): dict(marker='s', color=RED, mfc='white', label='nitrogen, flat plate')}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.95), sharey=True)
panels = (('full', '(a) full width, 10–90%'), ('up', '(b) upstream half, 10–50%'), ('down', '(c) downstream half, 50–90%'))
for gas in ('Ar', 'N2'):
    rows = json.load(open(d / gas / 'truncation.json'))
    for body in ('cylinder', 'plate'):
        rr = sorted((r for r in rows if r['body'] == body), key=lambda r: r['room'])
        for k, (key, _) in enumerate(panels):
            ax[k].plot([r['room'] for r in rr], [r[key] for r in rr], lw=1.1, ms=6, mew=1.1, **style[(gas, body)])
for a, (_, title) in zip(ax, panels):
    a.axhline(1, color='0.55', lw=0.7, ls=(0, (4, 3)))
    a.set_xscale('log'); a.set_xlim(0.35, 13); a.set_ylim(0.08, 1.06)
    a.set_xticks([0.5, 1, 2, 5, 10]); a.set_xticklabels(['0.5', '1', '2', '5', '10'])
    a.minorticks_off()
    a.set_title(title, fontsize=10.5, pad=5)
    a.set_xlabel(r'room, $s_{10}/w_p$')
ax[0].set_ylabel('width / planar width')
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=10.5, handlelength=2.2, columnspacing=1.6,
           bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=0.6)
fig.savefig(out); fig.savefig(out.with_suffix('.png'), dpi=160)
print('wrote', out)

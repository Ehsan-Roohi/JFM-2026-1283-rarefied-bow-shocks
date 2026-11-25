"""Width of the cylinder front relative to the planar shock versus the room
between the front and the wall, both in units of the planar 10-90 % width.
usage: python truncation_plot.py <dir with planar_vs_cylinder_mfp.json>"""
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = json.load(open(d / 'planar_vs_cylinder_mfp.json'))
pl = {r['gas']: r for r in rows if r['case'] == 'planar'}
table = []
for r in rows:
    if r['case'] == 'planar':
        continue
    p = pl[r['gas']]
    w, w1 = r['w10_90_mfp'], r['w10_50_mfp']
    table.append(dict(gas=r['gas'], Kn=float(r['case']), room=r['room_mfp'] / p['w10_90_mfp'],
                      full=w / p['w10_90_mfp'], up=w1 / p['w10_50_mfp'], down=(w - w1) / p['w50_90_mfp'],
                      Tpeak=r['T_peak'] / p['T_peak']))
print(f"{'gas':3s} {'Kn':>6s} {'room/w_pl':>9s} {'full':>6s} {'up(10-50)':>9s} {'down(50-90)':>11s} {'Tpeak/planar':>12s}")
for t in table:
    print(f"{t['gas']:3s} {t['Kn']:6.3f} {t['room']:9.2f} {t['full']:6.3f} {t['up']:9.3f} {t['down']:11.3f} {t['Tpeak']:12.3f}")
json.dump(table, open(d / 'truncation_table.json', 'w'), indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 10})
fig, ax = plt.subplots(figsize=(4.6, 3.4))
mk = {'Ar': 'o', 'N2': 's'}
for gas in ('Ar', 'N2'):
    tt = sorted([t for t in table if t['gas'] == gas], key=lambda t: t['room'])
    x = [t['room'] for t in tt]
    for key, col, lab in (('up', '#1f4e9c', 'upstream half (10-50%)'), ('full', 'k', 'full width (10-90%)'),
                          ('down', '#c0392b', 'downstream half (50-90%)')):
        ax.plot(x, [t[key] for t in tt], marker=mk[gas], color=col, lw=0.9, ms=5, mfc='white' if gas == 'N2' else col,
                label=f"{lab}" if gas == 'Ar' else None)
ax.axhline(1, color='0.6', lw=0.6)
ax.set_xlabel(r'room between front and wall / planar shock width')
ax.set_ylabel('cylinder width / planar width')
ax.set_xlim(0, 5)
ax.set_ylim(0, 1.15)
ax.legend(frameon=False, fontsize=7.5, loc='lower right')
ax.text(0.03, 0.95, 'filled: argon   open: nitrogen', transform=ax.transAxes, fontsize=7.5, va='top')
fig.tight_layout()
fig.savefig(d / 'truncation.pdf')
fig.savefig(d / 'truncation.png', dpi=200)

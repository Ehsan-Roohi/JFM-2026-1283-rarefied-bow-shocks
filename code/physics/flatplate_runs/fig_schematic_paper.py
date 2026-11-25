"""Paper figure: the two matched configurations, drawn to scale for Kn = 0.05:
(a) cylinder, (b) flat plate.  Shows the computational domain, boundary conditions,
the body, the stagnation line and the definition of s10 (10 % density-rise point to
the body face).  The 10 % and 90 % density contours are the computed argon ones.

usage: python fig_schematic_paper.py <campaign_dir> <out_pdf>
"""
import csv, gzip, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from scipy.ndimage import gaussian_filter

camp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
R, TP = 0.1524, 0.02
D = 2 * R
KN, GAS, RATIO = 0.05, 'Ar', 3.8835
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9.5, 'axes.linewidth': 0.7})
BLUE, RED, GREY = '#1f4e9c', '#c0392b', '#555555'


def load(body):
    d = camp / 'cases' / f'{GAS}_{body}_Kn{KN:g}'.replace('.', 'p')
    meta = json.load(open(d / 'case.json'))
    with gzip.open(d / 'avg_fields.csv.gz', 'rt') as f:
        head = f.readline(); rows = list(csv.DictReader(f))
    c = int(head.split('coarsen=')[1].split()[0])
    dx, dy, xmin = meta['dx'] * c, meta['dy'] * c, meta['domain'][0]
    nx, ny = int(round((meta['domain'][1] - xmin) / dx)) + 1, int(round(meta['domain'][3] / dy)) + 1
    rho = np.full((ny, nx), np.nan)
    for r in rows:
        if not r['nrho'] or float(r['flow_fraction']) < 0.5:
            continue
        i, j = int(round((float(r['xc']) - xmin) / dx - 0.5)), int(round(float(r['yc']) / dy - 0.5))
        if 0 <= i < nx and 0 <= j < ny:
            rho[j, i] = float(r['nrho']) / meta['nrho']
    ok = np.isfinite(rho)
    s = gaussian_filter(np.where(ok, rho, 0.0), 1.5) / np.maximum(gaussian_filter(ok.astype(float), 1.5), 1e-6)
    return meta, (xmin + (np.arange(nx) + 0.5) * dx) / D, ((np.arange(ny) + 0.5) * dy) / D, np.where(ok, s, np.nan)


trunc = {(r['body'], r['Kn']): r for r in json.load(open(camp / 'results' / GAS / 'truncation.json'))}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.85))
fig.subplots_adjust(left=0.065, right=0.99, top=0.9, bottom=0.15, wspace=0.14)
for ax, body, lab in zip(axs, ('cylinder', 'plate'), ('(a) cylinder', '(b) flat plate')):
    meta, x, y, rho = load(body)
    x0, x1, _, y1 = [v / D for v in meta['domain']]
    ax.add_patch(Rectangle((x0, 0), x1 - x0, y1, fc='#f4f7fb', ec='k', lw=1.0, zorder=0))
    X, Y = np.meshgrid(x, y)
    for frac, col, ls in ((0.1, BLUE, '-'), (0.9, RED, '-')):
        ax.contour(X, Y, rho, levels=[1 + frac * (RATIO - 1)], colors=[col], linewidths=1.1, linestyles=ls, zorder=2)
    if body == 'cylinder':
        ax.add_patch(Circle((0.5, 0), 0.5, fc='0.78', ec='k', lw=0.9, zorder=4))
        ax.text(0.5, 0.17, 'diffuse wall\n500 K', ha='center', va='center', fontsize=7.5, zorder=5, linespacing=1.0)
    else:
        ax.add_patch(Rectangle((0, 0), TP / D, 0.5, fc='0.78', ec='k', lw=0.9, zorder=4))
        ax.annotate('diffuse wall, 500 K', xy=(TP / D, 0.3), xytext=(0.45, 0.42), fontsize=8, va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=GREY), zorder=5)
    # stagnation line and s10
    s10 = trunc[(body, KN)]['s10_mm'] / 1e3 / D
    ax.plot([x0, 0], [0, 0], color='k', lw=2.2, solid_capstyle='butt', zorder=6)
    ax.plot([-s10], [0], 'o', ms=4.5, color=BLUE, zorder=7, clip_on=False)
    ya = 0.22
    ax.add_patch(FancyArrowPatch((-s10, ya), (0, ya), arrowstyle='<->', mutation_scale=9, lw=0.9, color='k', zorder=7))
    ax.plot([-s10, -s10], [0, ya + 0.04], color=BLUE, lw=0.7, ls=':', zorder=6)
    ax.plot([0, 0], [0, ya + 0.04], color='k', lw=0.7, ls=':', zorder=6)
    ax.text(-s10 / 2, ya + 0.045, r'$s_{10}$', ha='center', va='bottom', fontsize=10, zorder=7,
            bbox=dict(fc='#f4f7fb', ec='none', pad=0.6))
    ax.annotate('stagnation line', xy=(x0 + 0.25, 0.0), xytext=(x0 + 0.06, 0.45), fontsize=8, va='center',
                arrowprops=dict(arrowstyle='-', lw=0.6, color=GREY), zorder=7)
    # contour labels
    ax.text(-s10 - 0.05, 1.15, '10%', color=BLUE, fontsize=8.5, ha='right', va='center', zorder=7)
    ax.text(0.32 if body == 'cylinder' else 0.3, 0.8, '90%', color=RED, fontsize=8.5, ha='center', va='center', zorder=7)
    # height of the body
    xb = 1.25 if body == 'cylinder' else 0.3
    if body == 'cylinder':
        ax.add_patch(FancyArrowPatch((0.0, -0.12), (1.0, -0.12), arrowstyle='<->', mutation_scale=8, lw=0.8,
                                     color='k', clip_on=False, zorder=7))
        ax.text(0.5, -0.2, r'$D=2R$', ha='center', va='top', fontsize=8.5, clip_on=False)
    else:
        ax.add_patch(FancyArrowPatch((0.2, 0.0), (0.2, 0.5), arrowstyle='<->', mutation_scale=8, lw=0.8, color='k', zorder=7))
        ax.text(0.23, 0.17, r'$R$', ha='left', va='center', fontsize=8.5, zorder=7)
    # freestream arrows and boundary labels
    for yy in (0.95, 1.3, 1.65):
        ax.add_patch(FancyArrowPatch((x0 + 0.06, yy), (x0 + 0.42, yy), arrowstyle='-|>', mutation_scale=9, lw=1.0,
                                     color='k', zorder=7))
    ax.text(x0 + 0.06, 1.8, r'$M_\infty=10$', ha='left', va='bottom', fontsize=8.5)
    ax.text((x0 + x1) / 2, y1 + 0.03, 'freestream boundary', ha='center', va='bottom', fontsize=8)
    ax.text(x1 - 0.03, y1 / 2, 'freestream boundary', ha='right', va='center', fontsize=8, rotation=90)
    ax.text((0.5 + x1) / 2 + 0.25, 0.04, 'symmetry plane', ha='center', va='bottom', fontsize=8)
    ax.set_title(lab, fontsize=10, loc='left', pad=14)
    ax.set_xlim(x0 - 0.02, x1 + 0.02); ax.set_ylim(-0.02, y1 + 0.02); ax.set_aspect('equal')
    ax.set_xlabel(r'$x/D$', fontsize=9, labelpad=9 if body == 'cylinder' else 2)
    ax.tick_params(labelsize=8, length=2.5)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([-1, 0, 1, 2]); ax.set_yticks([0, 1, 2])
axs[0].set_ylabel(r'$y/D$', fontsize=9)
fig.savefig(out); fig.savefig(out.with_suffix('.png'), dpi=170)
print('wrote', out)

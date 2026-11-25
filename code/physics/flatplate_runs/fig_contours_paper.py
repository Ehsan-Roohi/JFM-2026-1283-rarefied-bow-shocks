"""Paper figure, Tecplot-style RGB contours: argon, cylinder (top row) and flat
plate (bottom row) at Kn = 0.01, 0.05, 0.2.  In every panel the upper half
shows density, the lower half translational temperature (the flow is symmetric).

usage: python fig_contours_paper.py <campaign_dir> <out_pdf> [gas]
"""
import csv, gzip, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.patches import Circle, Rectangle
from scipy.ndimage import gaussian_filter

camp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
GAS = sys.argv[3] if len(sys.argv) > 3 else 'Ar'
R, TP = 0.1524, 0.02
D = 2 * R
PLANAR_W = {'Ar': 8.94, 'N2': 4.97}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9.5, 'axes.linewidth': 0.7})
KNS = [0.01, 0.05, 0.2]
WIN = {0.01: (-0.30, 0.40, 0.42), 0.05: (-0.45, 0.40, 0.51), 0.2: (-0.80, 0.40, 0.72)}
CMAP = plt.get_cmap('jet')          # blue-cyan-green-yellow-red, as in Tecplot


def load(body, kn):
    d = camp / 'cases' / f'{GAS}_{body}_Kn{kn:g}'.replace('.', 'p')
    meta = json.load(open(d / 'case.json'))
    with gzip.open(d / 'avg_fields.csv.gz', 'rt') as f:
        head = f.readline()
        rows = list(csv.DictReader(f))
    c = int(head.split('coarsen=')[1].split()[0])
    dx, dy, xmin = meta['dx'] * c, meta['dy'] * c, meta['domain'][0]
    nx, ny = int(round((meta['domain'][1] - xmin) / dx)) + 1, int(round(meta['domain'][3] / dy)) + 1
    rho, T = np.full((ny, nx), np.nan), np.full((ny, nx), np.nan)
    for r in rows:
        if not r['nrho'] or float(r['flow_fraction']) < 0.5:
            continue
        i, j = int(round((float(r['xc']) - xmin) / dx - 0.5)), int(round(float(r['yc']) / dy - 0.5))
        if 0 <= i < nx and 0 <= j < ny:
            rho[j, i] = float(r['nrho']) / meta['nrho']
            T[j, i] = float(r['temp']) / 200.0
    x, y = xmin + (np.arange(nx) + 0.5) * dx, (np.arange(ny) + 0.5) * dy
    # display smoothing of the sampling scatter: NaN-aware Gaussian (about 0.3 lambda_inf),
    # faded out within 0.1 m of the body so the wall layer is not blurred
    sig = max(0.3 * meta['lambda_m'] / dx, 0.8)
    X, Y = np.meshgrid(x, y)
    dist = (np.hypot(X - R, Y) - R) if body == 'cylinder' else np.hypot(np.clip(-X, 0, None) + np.clip(X - TP, 0, None),
                                                                         np.clip(Y - R, 0, None))
    w = np.clip((dist - 0.02) / 0.08, 0.0, 1.0)

    def smooth(A):
        ok = np.isfinite(A)
        num = gaussian_filter(np.where(ok, A, 0.0), sig, mode='nearest')
        den = gaussian_filter(ok.astype(float), sig, mode='nearest')
        S = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
        return np.where(ok, w * S + (1 - w) * A, np.nan)

    return x / D, y / D, smooth(rho), smooth(T)


trunc = {(r['body'], r['Kn']): r for r in json.load(open(camp / 'results' / GAS / 'truncation.json'))}
lev_r = np.geomspace(1.0, 25.0, 25)
TMAX = {'Ar': 34.0, 'N2': 25.0}[GAS]
lev_T = np.linspace(0.0, TMAX, int(TMAX) + 1)
norm_r = BoundaryNorm(lev_r, CMAP.N, extend='max')
norm_T = BoundaryNorm(lev_T, CMAP.N, extend='max')

widths = [WIN[k][1] - WIN[k][0] for k in KNS]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.75), gridspec_kw=dict(width_ratios=[1, 1, 1]))
fig.subplots_adjust(left=0.105, right=0.86, top=0.93, bottom=0.085, wspace=0.12, hspace=0.12)
for r_, body in enumerate(('cylinder', 'plate')):
    for c_, kn in enumerate(KNS):
        ax = axs[r_, c_]
        x, y, rho, T = load(body, kn)
        X, Y = np.meshgrid(x, y)
        cr = ax.contourf(X, Y, np.clip(rho, 1.0, None), levels=lev_r, cmap=CMAP, norm=norm_r, extend='max')
        cr.set_rasterized(True)
        ax.contour(X, Y, rho, levels=lev_r[3::2], colors='k', linewidths=0.25)
        cT = ax.contourf(X, -Y, T, levels=lev_T, cmap=CMAP, norm=norm_T, extend='max')
        cT.set_rasterized(True)
        ax.contour(X, -Y, T, levels=lev_T[6::3], colors='k', linewidths=0.25)
        ax.axhline(0, color='w', lw=0.8)
        if body == 'cylinder':
            ax.add_patch(Circle((0.5, 0), 0.5, fc='0.75', ec='k', lw=0.7, zorder=5))
        else:
            ax.add_patch(Rectangle((0, -0.5), TP / D, 1.0, fc='0.75', ec='k', lw=0.7, zorder=5))
        x0, x1, yh = WIN[kn]
        ax.set_xlim(x0 / D, x1 / D); ax.set_ylim(-yh / D, yh / D); ax.set_aspect('equal')
        ax.tick_params(labelsize=7.5, length=2.5)
        kw = dict(transform=ax.transAxes, fontsize=8, color='w', fontweight='bold')
        ax.text(0.03, 0.95, r'$\rho/\rho_\infty$', va='top', **kw)
        ax.text(0.03, 0.05, r'$T_{\rm tr}/T_\infty$', va='bottom', **kw)
        ax.text(0.97, 0.95, f'({"abcdef"[3 * r_ + c_]})', va='top', ha='right', **kw)
        room = trunc[(body, kn)]['room'] if 'room' in trunc[(body, kn)] else None
        if room is not None:
            ax.text(0.97, 0.05, rf'$s_{{10}}/w_p={room:.2f}$', va='bottom', ha='right', **kw)
        if r_ == 0:
            ax.set_title(rf'$Kn_\infty={kn:g}$', fontsize=10)
        if r_ == 1:
            ax.set_xlabel(r'$x/D$', fontsize=9)
        if c_ == 0:
            ax.set_ylabel(('cylinder' if body == 'cylinder' else 'flat plate') + '\n' + r'$y/D$', fontsize=9)
cax1 = fig.add_axes([0.875, 0.53, 0.018, 0.40])
cb1 = fig.colorbar(cr, cax=cax1, ticks=[1, 2, 5, 10, 20])
cb1.ax.set_yticklabels(['1', '2', '5', '10', '20']); cb1.set_label(r'$\rho/\rho_\infty$ (upper halves)', fontsize=8.5)
cax2 = fig.add_axes([0.875, 0.085, 0.018, 0.40])
cb2 = fig.colorbar(cT, cax=cax2, ticks=list(range(0, int(TMAX) + 1, 5)))
cb2.set_label(r'$T_{\rm tr}/T_\infty$ (lower halves)', fontsize=8.5)
for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=7.5)
fig.savefig(out, dpi=300)
fig.savefig(out.with_suffix('.png'), dpi=170)
print('wrote', out)

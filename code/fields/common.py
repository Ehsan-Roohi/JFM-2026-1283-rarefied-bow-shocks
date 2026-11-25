"""Shared reader / gridding / layout helpers for the composite field figures.
Presentation only: the DS2V values are read, normalised by the freestream value of the
same case and drawn; no number is modified."""
import pathlib, re
import numpy as np
from scipy.interpolate import LinearNDInterpolator

R, D = 0.1524, 0.3048
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'data/ds2v'
CACHE = HERE / 'cache'


def read_ds2ff(path):
    txt = open(path, encoding='latin-1').read()
    head = txt.split('ZONE')[0]
    names = re.findall(r'"([^"]*)"', head)
    zline_end = txt.index('\n', txt.index('ZONE'))
    npts = int(re.search(r'\bI=\s*(\d+)', txt[txt.index('ZONE'):zline_end]).group(1))
    body = txt[zline_end:].split(None, npts * len(names) + 1)[: npts * len(names)]
    vals = np.array([float(t) for t in body]).reshape(npts, len(names))
    return {n: vals[:, i] for i, n in enumerate(names)}


def n_inf_from_input(path):
    for line in open(path, encoding='latin-1'):
        t = line.split('!')[0].strip()
        if re.fullmatch(r'[0-9.]+e\+(19|20|21|18|17)', t, flags=re.I):
            return float(t)
    raise SystemExit('no number density in ' + str(path))


def load_case(gas, kn):
    """DS2FF columns (cached as npz) + freestream number density of that case."""
    p = ROOT / 'kn' / f'{gas}_Kn{kn:g}.dat'
    CACHE.mkdir(exist_ok=True)
    c = CACHE / f'{gas}_Kn{kn:g}.npz'
    if c.exists():
        z = np.load(c)
        d = {k: z[k] for k in z.files}
    else:
        d0 = read_ds2ff(p)
        d = {k: d0[k] for k in ('X', 'Y', 'ND', 'TTR', 'TRT')}
        np.savez_compressed(c, **d)
    return d, n_inf_from_input(p.with_name(f'{gas}_Kn{kn:g}.DS2VD')), p


def grid_field(d, q, win, nx=900):
    """Linear interpolation of the cell-centre values q onto a regular grid covering the
    window win = (x0, x1, y1) given in units of D (y from 0).  The cell centres are
    mirrored about y = 0 (symmetry plane) so that the strip between the axis and the
    first row of cell centres is filled; nothing else is altered."""
    x0, x1, y1 = win
    ny = max(2, int(round(nx * y1 / (x1 - x0))))
    xg = np.linspace(x0, x1, nx)
    yg = np.linspace(0.0, y1, ny)
    X, Y = np.meshgrid(xg, yg)
    px, py = d['X'] / D, d['Y'] / D
    m = (px > x0 - 0.3 * (x1 - x0)) & (px < x1 + 0.3 * (x1 - x0)) & (py < y1 * 1.3)
    px, py, qq = px[m], py[m], q[m]
    pts = np.c_[np.r_[px, px], np.r_[py, -py]]
    F = LinearNDInterpolator(pts, np.r_[qq, qq])
    return X, Y, F(X, Y)


# ---------------------------------------------------------------- plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 9.5, 'axes.linewidth': 0.7})
CMAP = plt.get_cmap('jet')            # blue-cyan-green-yellow-red, as in Tecplot

# density levels shared by ALL density figures (quasi-logarithmic, round values);
# 0.8-1.2 is one band so that freestream scatter (about +-4 %) never shows
LEV_RHO = np.array([0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 25])
NORM_RHO = BoundaryNorm(LEV_RHO, CMAP.N, extend='min')
# black lines at every band edge except 0.1 and 1.2: the 1.2 edge lies in the weak outer
# gradient and only traces statistical scatter at Kn = 0.1 and 1
LINES_RHO = [l for l in LEV_RHO if l not in (0.1, 1.2)]

# page geometry (inches); the same frame is used in every figure
FIG_W = 7.2
LEFT, PAN_W, HGAP = 0.50, 2.76, 0.42
ASPECT = 2.1                          # frame width / height = window width / height
PAN_H = PAN_W / ASPECT
BOT, VGAP, TOP = 0.42, 0.30, 0.08
CB_GAP, CB_W = 0.10, 0.13
TICK_FS, LAB_FS, TXT_FS = 8.0, 9.5, 9.0
MINUS = chr(0x2212)


def make_figure(nrows):
    fig_h = BOT + nrows * PAN_H + (nrows - 1) * VGAP + TOP
    fig = plt.figure(figsize=(FIG_W, fig_h))
    axs = []
    for r in range(nrows):
        y0 = BOT + (nrows - 1 - r) * (PAN_H + VGAP)
        for c in range(2):
            x0 = LEFT + c * (PAN_W + HGAP)
            axs.append(fig.add_axes([x0 / FIG_W, y0 / fig_h, PAN_W / FIG_W, PAN_H / fig_h]))
    xc = LEFT + 2 * PAN_W + HGAP + CB_GAP
    cax = fig.add_axes([xc / FIG_W, BOT / fig_h, CB_W / FIG_W, (fig_h - BOT - TOP) / fig_h])
    return fig, axs, cax


def _fmt(v, pos=None):
    return f'{v:g}'.replace('-', MINUS)


def style_panel(ax, win, xtick, ytick, letter, label, xlabel, ylabel, label_loc='lower right'):
    x0, x1, y1 = win
    assert abs((x1 - x0) / y1 - ASPECT) < 1e-6, win
    ax.add_patch(Circle((0.5, 0.0), 0.5, fc='0.78', ec='k', lw=0.7, zorder=5))
    ax.set_xlim(x0, x1); ax.set_ylim(0, y1); ax.set_aspect('equal')
    ax.xaxis.set_major_locator(MultipleLocator(xtick)); ax.yaxis.set_major_locator(MultipleLocator(ytick))
    ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.6, pad=2)
    ax.xaxis.set_major_formatter(_fmt); ax.yaxis.set_major_formatter(_fmt)
    kw = dict(transform=ax.transAxes, fontsize=TXT_FS, color='w', fontweight='bold', zorder=6)
    ax.text(0.025, 0.94, f'({letter})', va='top', ha='left', **kw)
    if label_loc == 'lower right':
        ax.text(0.975, 0.07, label, va='bottom', ha='right', **kw)
    else:                             # under the panel letter, in the freestream corner
        ax.text(0.025, 0.79, label, va='top', ha='left', **kw)
    if xlabel:
        ax.set_xlabel(r'$x/D$', fontsize=LAB_FS, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(r'$y/D$', fontsize=LAB_FS, labelpad=2)


def density_panel(ax, d, ninf, win, allow_over=False):
    X, Y, Z = grid_field(d, d['ND'] / ninf, win)
    assert np.isfinite(Z).all(), 'window leaves the DS2V output domain'
    # the main-paper fields stay below the top level (25); only the supplementary Mach-15
    # argon field exceeds it at the stagnation point and is then drawn in the top colour
    assert allow_over or Z.max() < LEV_RHO[-1], 'density above the top colour level'
    cf = ax.contourf(X, Y, np.minimum(Z, 0.999 * LEV_RHO[-1]), levels=LEV_RHO, cmap=CMAP, norm=NORM_RHO, extend='min')
    cf.set_rasterized(True)
    ax.contour(X, Y, Z, levels=LINES_RHO, colors='k', linewidths=0.3)
    return cf, Z


def density_colorbar(fig, cf, cax, label_every=1):
    """Ticks at every level; in the one-row figures only every second level is labelled
    (0.2, 0.8, 1.6, 2.5, 4, 6, 10, 25) because the bar is short."""
    cb = fig.colorbar(cf, cax=cax, ticks=LEV_RHO, spacing='uniform')
    cb.ax.set_yticklabels([f'{v:g}' if (i % label_every) == label_every - 1 else ''
                           for i, v in enumerate(LEV_RHO)])
    cb.ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.6)
    cb.set_label(r'$\rho/\rho_\infty$', fontsize=LAB_FS, labelpad=3)
    cb.outline.set_linewidth(0.7)
    return cb


def save(fig, name):
    fig.savefig(HERE / (name + '.pdf'), dpi=300)
    fig.savefig(HERE / (name + '.png'), dpi=200)
    print('wrote', name, f'{(HERE / (name + ".pdf")).stat().st_size / 1024:.0f} KB')

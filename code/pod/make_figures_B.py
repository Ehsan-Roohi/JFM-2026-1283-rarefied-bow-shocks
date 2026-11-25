"""Restyled figures 17, 19, 22, 23 (JFM-2026-1283).  Presentation only.

All plotted numbers are read from the frozen local copies of the cluster
inputs; the array selections (angle row, reconstruction, normalisation) follow
restored_rgb_figures.py / professional_figures_20261107.py line by line.

Run:  python -I make_figures_B.py
"""
from pathlib import Path
import csv
import glob
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, MultipleLocator, NullFormatter, NullLocator

HERE = Path(__file__).resolve().parent
EV = HERE.parents[1] / 'data/evidence'
FI = EV / 'figure_inputs'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10, 'axes.linewidth': 0.7,
    'axes.labelsize': 11, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5, 'lines.linewidth': 1.35,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0, 'ytick.minor.size': 2.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    'legend.frameon': False,
})

BLUE, RED, TEAL, ORANGE, PURPLE, GREY = '#1f4e9c', '#c0392b', '#2a9d8f', '#e9a03b', '#6c4f9c', '#555555'
WIDTH = 7.2  # inches; printed at about 5.3 in

# Knudsen-number sequence: cool-to-warm ramp + the line styles named in the caption
KN_COLOURS = [BLUE, TEAL, ORANGE, RED]
KN_STYLES = ['-', (0, (5.5, 2.2)), (0, (6.5, 1.8, 1.3, 1.8)), (0, (2.0, 1.4))]

# One identity per scalar, shared by figures 19 and 22 (table order)
SCALARS = ['rho', 'u', 'p', 'mach_proxy', 'ttr', 'trot']
SC_LABEL = {'rho': r'$\rho$', 'u': r'$u_x$', 'p': r'$p$', 'mach_proxy': r'$M_{\rm tr}$',
            'ttr': r'$T_{\rm tr}$', 'trot': r'$T_{\rm rot}$'}
SC_STYLE = {  # colour, line style, marker, filled?
    'rho': (BLUE, '-', 'o', True),
    'u': (TEAL, (0, (5.5, 2.0)), '^', True),
    'p': (PURPLE, (0, (6.5, 1.8, 1.3, 1.8)), 'D', False),
    'mach_proxy': (GREY, (0, (1.3, 1.5)), 'v', False),
    'ttr': (RED, (0, (3.2, 1.3)), 's', True),
    'trot': (ORANGE, (0, (7, 1.5, 1.2, 1.5, 1.2, 1.5)), 's', False),
}
NORMLABEL = {'rho': r'$\rho/\rho_\infty$', 'ttr': r'$T_{\rm tr}/T_\infty$', 'trot': r'$T_{\rm rot}/T_\infty$'}

NANG, NXI = 65, 301
ANGLES = np.linspace(100, 180, NANG)
XI = np.linspace(-1, 3, NXI)


def bank(gas, var):
    """Local copy of <P>/<gas>_<var>_attached_sigma1.0.npz (hash-prefixed)."""
    hits = glob.glob(str(FI / f'*_{gas}_{var}_attached_sigma1.0.npz'))
    assert len(hits) == 1, (gas, var, hits)
    return Path(hits[0])


def fullbank(z):
    """Identical to fullbank() of restored_rgb_figures.py."""
    support = z['support'].astype(bool)
    x = z['mean'] + z['coefficients'] @ z['modes']
    a = np.full((len(x), NANG * NXI), np.nan)
    a[:, support] = x
    return a.reshape(len(x), NANG, NXI), support.reshape(NANG, NXI)


def summary_rows(gas):
    """Density-attached, primary smoothing factor f = 1 (selected_rows of the cluster script)."""
    S = json.loads((EV / 'POD_SUMMARY.json').read_text())
    rows = [r for r in S if r['gas'] == gas and r['basis'] == 'density_attached' and r.get('smoothing_factor') == 1.0]
    return {r['variable']: r for r in rows}


def letter(ax, s, dx=0.0, dy=1.035, **kw):
    ax.text(dx, dy, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=10.5, **kw)


def save(fig, name):
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    clipped = tb.x0 < -1e-3 or tb.y0 < -1e-3 or tb.x1 > w + 1e-3 or tb.y1 > h + 1e-3
    print(f'{name}: {w:.2f} x {h:.2f} in; ink bbox x[{tb.x0:.3f},{tb.x1:.3f}] y[{tb.y0:.3f},{tb.y1:.3f}]', 'CLIPPED!' if clipped else 'ok')
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible() and t.get_fontsize() < 8]
    assert not small, small
    fig.savefig(HERE / (name + '.pdf'), dpi=300)
    fig.savefig(HERE / (name + '.png'), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 17: nitrogen profiles, physical and density-aligned coordinates
# --------------------------------------------------------------------------
def fig17_data():
    markers = list(csv.DictReader((EV / 'MARKERS.csv').open()))
    z = np.load(bank('N2', 'rho'))
    _, supp = fullbank(z)
    ia = int(np.argmax(supp.sum(axis=1)))
    angle = float(ANGLES[ia])
    out = dict(angle=angle, ia=ia, kn=z['kn'].tolist(), curves={})
    for v in ['rho', 'ttr', 'trot']:
        z = np.load(bank('N2', v))
        a, _ = fullbank(z)
        assert z['kn'].tolist() == out['kn']
        for k, kn in enumerate(z['kn']):
            mr = next(r for r in markers if r['gas'] == 'N2' and float(r['kn']) == kn
                      and float(r['smoothing_factor']) == 1 and float(r['angle']) == angle and r['supported'] == 'True')
            ss = float(mr['s']) + XI * float(mr['delta'])
            out['curves'][v, k] = (ss * 1000, XI, a[k, ia])
    return out


def fig17():
    d = fig17_data()
    assert abs(d['angle'] - 113.75) < 1e-12 and d['kn'] == [0.01, 0.025, 0.05, 0.1]
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.75))
    fig.subplots_adjust(left=0.068, right=0.988, bottom=0.098, top=0.885, wspace=0.30, hspace=0.40)
    for j, v in enumerate(['rho', 'ttr', 'trot']):
        for k, kn in enumerate(d['kn']):
            s_mm, xi, y = d['curves'][v, k]
            for i, x in enumerate([s_mm, xi]):
                axs[i, j].plot(x, y, color=KN_COLOURS[k], ls=KN_STYLES[k], zorder=3 + k,
                               solid_capstyle='round', dash_capstyle='butt')
        for i in (0, 1):
            ax = axs[i, j]
            ax.set_ylabel(NORMLABEL[v], labelpad=2)
            ax.yaxis.set_major_locator(MultipleLocator(2 if v == 'rho' else 5))
            ax.yaxis.set_minor_locator(MultipleLocator(1 if v == 'rho' else 1))
            ax.set_ylim((0.4, 8.3) if v == 'rho' else (0, 16.2) if v == 'ttr' else (0, 14.8))
        axs[0, j].set_xlabel(r'$s$ (mm)', labelpad=2)
        axs[0, j].set_xlim(-12, 462)
        axs[0, j].xaxis.set_major_locator(MultipleLocator(100))
        axs[0, j].xaxis.set_minor_locator(MultipleLocator(50))
        axs[1, j].set_xlabel(r'$\xi_\rho$', labelpad=1)
        axs[1, j].set_xlim(-1.1, 3.1)
        axs[1, j].xaxis.set_major_locator(MultipleLocator(1))
        axs[1, j].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[1, j].axvline(0, color='0.72', lw=0.6, zorder=1)
    for ax, s in zip(axs.flat, 'abcdef'):
        letter(ax, rf'$({s})$')
    # flow direction (the caption: upstream is to the right)
    for ax in (axs[0, 0], axs[1, 0]):
        ax.annotate('', xy=(0.60, 0.80), xytext=(0.92, 0.80), xycoords='axes fraction',
                    arrowprops=dict(arrowstyle='-|>', color='0.35', lw=0.8, mutation_scale=8))
        ax.text(0.76, 0.83, 'flow', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='0.25')
    handles = [Line2D([], [], color=KN_COLOURS[k], ls=KN_STYLES[k], lw=1.5) for k in range(4)]
    labels = [rf'$Kn_\infty={kn:g}$' for kn in d['kn']]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.528, 1.0), ncol=4,
               handlelength=3.4, columnspacing=2.2, handletextpad=0.6, fontsize=10)
    save(fig, 'Profile_collapse_N2_Kn')
    return d


# --------------------------------------------------------------------------
# Figures 19 and 22: cumulative variance fraction and in-sample reconstruction error
# --------------------------------------------------------------------------
def scalar_handles(present):
    hs = []
    for v in SCALARS:
        if v not in present:
            continue
        c, ls, m, filled = SC_STYLE[v]
        hs.append(Line2D([], [], color=c, ls=ls, marker=m, ms=5.2, mfc=c if filled else 'white', mec=c, mew=1.0, lw=1.35))
    return hs, [SC_LABEL[v] for v in SCALARS if v in present]


def plot_scalar(ax, x, y, v, z):
    c, ls, m, filled = SC_STYLE[v]
    ax.plot(x, y, color=c, ls=ls, marker=m, ms=5.2, mfc=c if filled else 'white', mec=c, mew=1.0,
            zorder=z, clip_on=False)


def two_panels(height):
    # equal unit spacing in both panels: argon has ranks 1-2, nitrogen 1-3
    xl = {'Ar': (0.62, 2.38), 'N2': (0.62, 3.38)}
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, height), sharey=True,
                            gridspec_kw=dict(width_ratios=[xl['Ar'][1] - xl['Ar'][0], xl['N2'][1] - xl['N2'][0]]))
    for ax, gas in zip(axs, ['Ar', 'N2']):
        ax.set_xlim(*xl[gas])
        ax.xaxis.set_major_locator(FixedLocator([1, 2] if gas == 'Ar' else [1, 2, 3]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Number of POD modes', labelpad=3)
    letter(axs[0], r'$(a)$ Argon')
    letter(axs[1], r'$(b)$ Nitrogen')
    return fig, axs


def fig19():
    fig, axs = two_panels(3.25)
    fig.subplots_adjust(left=0.098, right=0.985, bottom=0.145, top=0.83, wspace=0.07)
    data = {}
    for ax, gas in zip(axs, ['Ar', 'N2']):
        rows = summary_rows(gas)
        for z, v in enumerate(SCALARS):
            if v not in rows:
                continue
            r = rows[v]
            e = np.array(r['spectrum'])[:r['snapshots'] - 1]
            x, y = np.arange(1, len(e) + 1), np.cumsum(e)
            plot_scalar(ax, x, y, v, 3 + z)
            data[gas, v] = (x, y)
        ax.grid(axis='y', color='0.9', lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_ylim(0.70, 1.012)
    axs[0].yaxis.set_major_locator(MultipleLocator(0.1))
    axs[0].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[0].set_ylabel('Cumulative variance fraction', labelpad=4)
    hs, ls = scalar_handles(SCALARS)
    fig.legend(hs, ls, loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=6, handlelength=3.3,
               columnspacing=1.6, handletextpad=0.5, fontsize=10)
    save(fig, 'POD_spectrum_Kn')
    return data


# piecewise-log axis: the eleven decades between 10^-13 % and 10^-2 %, which contain no data,
# are compressed into a narrow band; data values are untouched.
_L = np.array([-14.6, -13.0, -2.0, 1.0])
_T = np.array([0.0, 0.80, 1.22, 4.22])


def _fwd(y):
    y = np.asarray(y, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.log10(np.where(y > 0, y, np.nan))
    return _lin(L, _L, _T)


def _inv(t):
    return 10.0 ** _lin(np.asarray(t, float), _T, _L)


def _lin(x, xp, fp):
    """Piecewise-linear map with linear extrapolation at both ends (monotonic)."""
    x = np.asarray(x, float)
    out = np.interp(x, xp, fp)
    lo = x < xp[0]
    hi = x > xp[-1]
    out = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), out)
    out = np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), out)
    return out


def fig22(broken=True):
    fig, axs = two_panels(3.25)
    fig.subplots_adjust(left=0.118, right=0.985, bottom=0.145, top=0.83, wspace=0.07)
    data = {}
    for ax, gas in zip(axs, ['Ar', 'N2']):
        rows = summary_rows(gas)
        for z, v in enumerate(SCALARS):
            if v not in rows:
                continue
            rr = [a for a in rows[v]['reconstruction'] if a['r'] > 0]
            x = np.array([a['r'] for a in rr])
            y = np.array([a['relative_L2_percent'] for a in rr])
            plot_scalar(ax, x, y, v, 3 + z)
            data[gas, v] = (x, y)
    ax0 = axs[0]
    if broken:
        ax0.set_yscale('function', functions=(_fwd, _inv))
        ax0.set_ylim(10 ** _L[0], 10 ** _L[-1])
        ax0.yaxis.set_major_locator(FixedLocator([1e-14, 1e-13, 1e-2, 0.1, 1, 10]))
        ax0.yaxis.set_major_formatter(FixedFormatter([r'$10^{-14}$', r'$10^{-13}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$']))
        minor = [m * 10.0 ** k for k in (-15, -14) for m in range(2, 10)] + [m * 10.0 ** k for k in (-2, -1, 0) for m in range(2, 10)]
        minor = [m for m in minor if (10 ** _L[0] <= m <= 10 ** _L[1]) or (10 ** _L[2] <= m <= 10 ** _L[3])]
        ax0.yaxis.set_minor_locator(FixedLocator(minor))
        ax0.yaxis.set_minor_formatter(NullFormatter())
        tb0, tb1 = _T[1], _T[2]  # compressed band in transformed units
        tspan = _T[-1] - _T[0]
        f0, f1 = (tb0 - _T[0]) / tspan, (tb1 - _T[0]) / tspan
        fm = 0.5 * (f0 + f1)
        for ax in axs:
            ax.axhspan(_inv(tb0), _inv(tb1), color='0.93', lw=0, zorder=0)
            # axis-break marks on both spines
            for xs in (0.0, 1.0):
                ax.plot([xs, xs], [fm - 0.016, fm + 0.016], transform=ax.transAxes, color='white', lw=2.2,
                        clip_on=False, zorder=6, solid_capstyle='butt')
                for off in (-0.016, 0.016):
                    dxa = 0.016 * (axs[1].get_position().width / ax.get_position().width)
                    ax.plot([xs - dxa, xs + dxa], [fm + off - 0.012, fm + off + 0.012], transform=ax.transAxes,
                            color='k', lw=0.7, clip_on=False, zorder=7)
        axs[1].text(0.045, fm, r'scale compressed', transform=axs[1].transAxes,
                    ha='left', va='center', fontsize=8.5, color='0.35', style='italic', zorder=2)
        axs[0].text(0.06, (0.5 * f0), 'full rank:\nround-off', transform=axs[0].transAxes, ha='left', va='center',
                    fontsize=8.5, color='0.35', style='italic', linespacing=1.05)
    else:
        ax0.set_yscale('log')
        ax0.set_ylim(1e-15, 30)
        ax0.yaxis.set_major_locator(FixedLocator([1e-14, 1e-11, 1e-8, 1e-5, 1e-2, 10]))
        ax0.yaxis.set_minor_locator(FixedLocator([10.0 ** k for k in range(-15, 2)]))
        ax0.yaxis.set_minor_formatter(NullFormatter())
        for ax in axs:
            ax.grid(axis='y', color='0.9', lw=0.5, zorder=0)
            ax.set_axisbelow(True)
    ax0.set_ylabel('Weighted relative\nreconstruction error (%)', labelpad=3, linespacing=1.1)
    hs, ls = scalar_handles(SCALARS)
    fig.legend(hs, ls, loc='upper center', bbox_to_anchor=(0.55, 1.0), ncol=6, handlelength=3.3,
               columnspacing=1.6, handletextpad=0.5, fontsize=10)
    save(fig, 'POD_reconstruction_error_Kn' if broken else 'POD_reconstruction_error_Kn_alt_fulllogaxis')
    return data


# --------------------------------------------------------------------------
# Figure 23: mean and first two modes of T_tr and T_rot (nitrogen, aligned)
# --------------------------------------------------------------------------
DIVERGING = LinearSegmentedColormap.from_list(
    'blue_grey_red', ['#12306b', BLUE, '#7ea6d8', '#e2e2e2', '#e59a8c', RED, '#7d1d14'], N=256)
DIVERGING.set_bad('white')
JET = matplotlib.colormaps['jet'].copy()
JET.set_bad('white')


def fig23_data():
    out = {}
    for v in ['ttr', 'trot']:
        z = np.load(bank('N2', v))
        support = z['support'].astype(bool)
        for j, value in enumerate([z['mean'], z['modes'][0], z['modes'][1]]):
            a = np.full(NANG * NXI, np.nan)
            a[support] = value
            out[v, j] = a.reshape(NANG, NXI)
        out[v, 'support'] = support.reshape(NANG, NXI)
    return out


def fig23(shared_modes=True):
    d = fig23_data()
    sup = d['ttr', 'support']
    assert (sup == d['trot', 'support']).all()
    rows = np.where(sup.any(axis=1))[0]
    lo, hi = rows.min(), rows.max()
    dth = ANGLES[1] - ANGLES[0]
    dxi = XI[1] - XI[0]
    th_edges = np.concatenate([ANGLES - dth / 2, [ANGLES[-1] + dth / 2]])
    xi_edges = np.concatenate([XI - dxi / 2, [XI[-1] + dxi / 2]])

    mean_lo = min(np.nanmin(d[v, 0]) for v in ('ttr', 'trot'))
    mean_hi = max(np.nanmax(d[v, 0]) for v in ('ttr', 'trot'))
    assert 1.0 <= mean_lo and mean_hi <= 15.0, (mean_lo, mean_hi)
    mean_norm = Normalize(1.0, 15.0)
    amax = {j: max(float(np.nanmax(np.abs(d[v, j]))) for v in ('ttr', 'trot')) for j in (1, 2)}
    if shared_modes:
        lim = {1: 0.32, 2: 0.32}
        assert max(amax.values()) <= 0.32
    else:
        lim = {1: 0.32, 2: 0.12}
        assert amax[1] <= 0.32 and amax[2] <= 0.12

    H = 4.62
    fig = plt.figure(figsize=(WIDTH, H))
    L, R_, gap = 0.80, 0.14, 0.30          # inches
    pw = (WIDTH - L - R_ - 2 * gap) / 3
    ph, vgap, B = 1.30, 0.42, 0.50
    axs = np.empty((2, 3), object)
    ims = {}
    for i, v in enumerate(['ttr', 'trot']):
        for j in range(3):
            x0 = L + j * (pw + gap)
            y0 = B + (1 - i) * (ph + vgap)
            ax = fig.add_axes([x0 / WIDTH, y0 / H, pw / WIDTH, ph / H])
            axs[i, j] = ax
            f = np.ma.masked_invalid(d[v, j])
            if j == 0:
                im = ax.pcolormesh(xi_edges, th_edges, f, cmap=JET, norm=mean_norm, rasterized=True, shading='flat')
            else:
                im = ax.pcolormesh(xi_edges, th_edges, f, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim[j], lim[j]),
                                   rasterized=True, shading='flat')
            ims[i, j] = im
            ax.set_xlim(-1.0, xi_edges[-1])
            ax.set_ylim(ANGLES[lo] - dth / 2, ANGLES[hi] + dth / 2)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.yaxis.set_major_locator(FixedLocator([111.25, 113.75]))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(which='both', color='k')
            if j == 0:
                ax.set_yticklabels(['111.25', '113.75'])
                ax.set_ylabel(r'$\theta$ (deg)', labelpad=3)
            else:
                ax.set_yticklabels([])
            if i == 1:
                ax.set_xlabel(r'$\xi_\rho$', labelpad=1)
            else:
                ax.set_xticklabels([])
            name = [r'$T_{\rm tr}$', r'$T_{\rm rot}$'][i]
            what = ['mean', 'mode 1', 'mode 2'][j]
            ax.text(0.0, 1.045, rf'$({"abcdef"[3 * i + j]})$ {name}, {what}', transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=10)
    # colour bars above the top row
    cb_y = (B + 2 * ph + vgap + 0.46) / H
    cb_h = 0.085 / H

    def cbar(im, x0, w, label, ticks):
        cax = fig.add_axes([x0 / WIDTH, cb_y, w / WIDTH, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticks)
        cb.outline.set_linewidth(0.6)
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cax.tick_params(length=2.5, width=0.6, labelsize=9, pad=1.5, direction='out', top=True, bottom=False)
        cb.set_label(label, labelpad=3, fontsize=10)
        cb.solids.set_rasterized(True)
        return cb

    cbar(ims[0, 0], L + 0.08 * pw, 0.84 * pw, r'mean of $T/T_\infty$', [1, 3, 5, 7, 9, 11, 13, 15])
    if shared_modes:
        x0 = L + pw + gap
        wtot = 2 * pw + gap
        cbar(ims[0, 1], x0 + 0.2 * wtot, 0.6 * wtot, r'mode amplitude, $\phi_1$ and $\phi_2$',
             [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        save(fig, 'POD_modes_N2_thermal')
    else:
        cbar(ims[0, 1], L + pw + gap + 0.08 * pw, 0.84 * pw, r'mode 1, $\phi_1$', [-0.3, -0.15, 0, 0.15, 0.3])
        cbar(ims[0, 2], L + 2 * (pw + gap) + 0.08 * pw, 0.84 * pw, r'mode 2, $\phi_2$', [-0.1, -0.05, 0, 0.05, 0.1])
        save(fig, 'POD_modes_N2_thermal_alt_percolumn')
    return d


# --------------------------------------------------------------------------
# Figure 18: nitrogen Mach-sweep bank, Mach-10 case, physical and density-aligned coordinates
# --------------------------------------------------------------------------
MACH = HERE.parents[1] / 'data/pod/mach'   # local copy of the cluster folder restored_mach_pod (sha256 matched)
MACH_NORMLABEL = {'rho': r'$\rho/\rho_{\rm ref}$', 'ttr': r'$T_{\rm tr}/T_\infty$', 'trot': r'$T_{\rm rot}/T_\infty$'}
MACH_STYLES = ['-', (0, (5.5, 2.2)), (0, (6.5, 1.8, 1.3, 1.8)), (0, (2.0, 1.4)), (0, (7, 1.5, 1.2, 1.5, 1.2, 1.5))]


def fig18_data():
    """Same selections as restored_rgb_figures.py lines 185-190, for every Mach number of the bank."""
    mmarkers = json.loads((MACH / 'MARKERS.json').read_text())
    z = np.load(MACH / 'N2_rho_attached_Mach.npz')
    _, supp = fullbank(z)
    ia = int(np.argmax(supp.sum(axis=1)))
    angle = float(ANGLES[ia])
    out = dict(angle=angle, ia=ia, mach=z['kn'].tolist(), curves={})
    for v in ['rho', 'ttr', 'trot']:
        z = np.load(MACH / f'N2_{v}_attached_Mach.npz')
        a, _ = fullbank(z)
        assert z['kn'].tolist() == out['mach']
        for k, mach in enumerate(z['kn']):
            mr = next(r for r in mmarkers if r['gas'] == 'N2' and r['mach'] == mach and r['angle'] == angle and r['supported'])
            ss = mr['s'] + XI * mr['delta']
            out['curves'][v, k] = (ss * 1000, XI, a[k, ia])
    return out


def _fig18_axes(top, hspace=0.40):
    fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.75))
    fig.subplots_adjust(left=0.068, right=0.988, bottom=0.098, top=top, wspace=0.30, hspace=hspace)
    for j, v in enumerate(['rho', 'ttr', 'trot']):
        for i in (0, 1):
            ax = axs[i, j]
            ax.set_ylabel(MACH_NORMLABEL[v], labelpad=2)
            ax.yaxis.set_major_locator(MultipleLocator(1 if v == 'rho' else 5))
            ax.yaxis.set_minor_locator(MultipleLocator(0.5 if v == 'rho' else 1))
        axs[0, j].set_xlabel(r'$s$ (mm)', labelpad=2)
        axs[0, j].xaxis.set_major_locator(MultipleLocator(20))
        axs[0, j].xaxis.set_minor_locator(MultipleLocator(10))
        axs[1, j].set_xlabel(r'$\xi_\rho$', labelpad=1)
        axs[1, j].set_xlim(-1.1, 3.1)
        axs[1, j].xaxis.set_major_locator(MultipleLocator(1))
        axs[1, j].xaxis.set_minor_locator(MultipleLocator(0.5))
        axs[1, j].axvline(0, color='0.72', lw=0.6, zorder=1)
    for ax, s in zip(axs.flat, 'abcdef'):
        letter(ax, rf'$({s})$')
    return fig, axs


def _flow(ax, x0=0.60, x1=0.92, y=0.80):
    ax.annotate('', xy=(x0, y), xytext=(x1, y), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.35', lw=0.8, mutation_scale=8))
    ax.text(0.5 * (x0 + x1), y + 0.03, 'flow', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='0.25')


def fig18():
    """Exactly the content of the current figure: the Mach-10 member of the Mach-sweep bank only."""
    d = fig18_data()
    assert abs(d['angle'] - 100.0) < 1e-12 and d['mach'] == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    k = d['mach'].index(10.0)
    fig, axs = _fig18_axes(top=0.885)
    for j, v in enumerate(['rho', 'ttr', 'trot']):
        s_mm, xi, y = d['curves'][v, k]
        for i, x in enumerate([s_mm, xi]):
            axs[i, j].plot(x, y, color=BLUE, ls='-', zorder=3, solid_capstyle='round')
            axs[i, j].set_ylim((0.6, 5.0) if v == 'rho' else (0, 13.4) if v == 'ttr' else (0, 12.7))
        axs[0, j].set_xlim(149, 211.5)
    _flow(axs[0, 0])
    _flow(axs[1, 0])
    fig.legend([Line2D([], [], color=BLUE, ls='-', lw=1.5)], [r'$M_\infty=10$,  $Kn_\infty\approx0.01$'],
               loc='upper center', bbox_to_anchor=(0.528, 1.0), ncol=1, handlelength=3.4, handletextpad=0.6, fontsize=10)
    save(fig, 'Profile_collapse_N2_Mach')
    return d


def fig18_all():
    """Variant, NOT the content of the current figure: all ten Mach numbers of the same bank."""
    d = fig18_data()
    n = len(d['mach'])
    ramp = LinearSegmentedColormap.from_list('mach_ramp', [BLUE, TEAL, ORANGE, RED])
    cols = [ramp(k / (n - 1)) for k in range(n)]
    fig, axs = _fig18_axes(top=0.845, hspace=0.42)
    for j, v in enumerate(['rho', 'ttr', 'trot']):
        ymax = max(float(np.nanmax(d['curves'][v, k][2])) for k in range(n))
        for k in range(n):
            s_mm, xi, y = d['curves'][v, k]
            for i, x in enumerate([s_mm, xi]):
                axs[i, j].plot(x, y, color=cols[k], ls=MACH_STYLES[k % 5], lw=1.15, zorder=3 + k)
        for i in (0, 1):
            axs[i, j].set_ylim((0.6, 1.06 * ymax) if v == 'rho' else (0, 1.07 * ymax))
        axs[0, j].set_xlim(138, 257)
    _flow(axs[0, 0], 0.66, 0.94, 0.84)
    _flow(axs[1, 0], 0.60, 0.92, 0.84)
    handles = [Line2D([], [], color=cols[k], ls=MACH_STYLES[k % 5], lw=1.4) for k in range(n)]
    labels = [rf'$M_\infty={m:g}$' for m in d['mach']]
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]   # legend fills column by column; this reads row by row
    fig.legend([handles[q] for q in order], [labels[q] for q in order], loc='upper center', bbox_to_anchor=(0.528, 1.005),
               ncol=5, handlelength=3.2, columnspacing=1.8, handletextpad=0.6, labelspacing=0.3, fontsize=10)
    save(fig, 'Profile_collapse_N2_Mach_alt_allMach')
    return d


if __name__ == '__main__':
    which = sys.argv[1:] or ['17', '18', '19', '22', '23']
    if '18' in which:
        fig18()
        fig18_all()
    if '17' in which:
        fig17()
    if '19' in which:
        fig19()
    if '22' in which:
        fig22(True)
        fig22(False)
    if '23' in which:
        fig23(True)
        fig23(False)

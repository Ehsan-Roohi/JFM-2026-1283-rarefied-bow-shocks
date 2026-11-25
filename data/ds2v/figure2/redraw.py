"""Redraw Figure 2 (outer density ridge vs SWD) from the project's point files.
The background density field is the raster already embedded in the current figure PDF
(the raw Mach-series field is only on the cluster); its extent is recovered from the axis ticks.
Changes: ridge drawn on top of the SWD markers, smaller open markers, colour bar matched to the axes,
and a lower panel with the ridge-minus-SWD distance along each bracketed ray."""
import sys, os, csv, io, math
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(SCR, 'mpl_env'))
sys.path.insert(1, os.path.join(SCR, 'pylib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
from PIL import Image
old_pdf,rec_dir,out_base=sys.argv[1:4]
R=.1524
D=2*R
img=Image.open(old_pdf).convert('RGB')
extent=[-0.1986870170228052, 0.6470083852787352, 0, 0.3962589358100675]
# ---- point data --------------------------------------------------------------
rd = lambda f: list(csv.DictReader(open(os.path.join(rec_dir, f))))
ridge = [r for r in rd('SWD_CORRECTED_RIDGE_POINTS.csv') if r['supported'] == 'True']
comp = [r for r in rd('SWD_BEFORE_AFTER_POINTS.csv') if r['new_minus_swd_mm'] not in ('', 'None')]
orig = [r for r in rd('ORIGINAL_SHOCK_COMPARISON_POINTS.csv') if r['gas'] == 'Ar' and float(r['Kn']) == .01 and float(r['Mach']) == 10]
ridge.sort(key=lambda r: float(r['angle_deg']))
dmm = np.array([float(r['new_minus_swd_mm']) for r in comp]); th = np.array([float(r['angle_deg']) for r in comp])
rms = float(np.sqrt(np.mean(dmm ** 2)))
print('bracketed angles', len(dmm), 'RMS %.3f mm, range %.3f..%.3f' % (rms, dmm.min(), dmm.max()))

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'axes.linewidth': .7})
fig = plt.figure(figsize=(5.6, 3.85))
ax = fig.add_axes([0.11, 0.335, 0.76, 0.655])
ax.imshow(np.asarray(img), extent=extent, origin='upper', interpolation='nearest', aspect='equal', zorder=0, rasterized=True)
ax.add_patch(plt.Circle((R, 0), R, fc='white', ec='black', lw=.9, zorder=2))
for method, col, mk, lab in [('akhlaghi_sigma1', '#9d248f', 'o', r'SWD centre, $\sigma=1$'),
                             ('akhlaghi_sigma0.1', '#00a8bf', 's', r'SWD aft side, $\sigma=0.1$')]:
    rr = sorted([q for q in orig if q['method'] == method], key=lambda q: float(q['y']))
    idx = sorted(set(list(range(0, len(rr), 8)) + [len(rr) - 1])); show = [rr[i] for i in idx]
    ax.plot([float(q['x']) for q in show], [float(q['y']) for q in show], ls='None', marker=mk, ms=(5.0 if mk == 'o' else 3.4),
            mfc='none', mec=col, mew=(1.1 if mk == 'o' else .8), label=lab, zorder=3)
ax.plot([float(r['x_m']) for r in ridge], [float(r['y_m']) for r in ridge], color='black', lw=.85, zorder=4,
        label='Outer density-gradient peak')
ax.set_xlim(extent[0], extent[1]); ax.set_ylim(0, extent[3])
from matplotlib.ticker import FuncFormatter
ax.set_xticks(np.array([-.5,0,.5,1,1.5,2])*D); ax.set_yticks(np.array([0,.25,.5,.75,1,1.25])*D)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v/D:g}')); ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v/D:g}'))
ax.set_xlim(extent[0],extent[1]); ax.set_ylim(0,extent[3]); ax.set_xlabel('$x/D$'); ax.set_ylabel('$y/D$')
ax.tick_params(direction='in', top=True, right=True)
ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=.95, edgecolor='.75', borderpad=.35,
          handlelength=1.6, labelspacing=.25)
ax.text(.012, .96, '(a)', transform=ax.transAxes, va='top', color='white', fontsize=10)
cax = ax.inset_axes([1.025, 0, .035, 1])
sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 30), cmap='turbo')
cb = fig.colorbar(sm, cax=cax); cb.set_ticks([0, 10, 20, 30]); cb.set_label(r'$\rho/\rho_{\mathrm{ref}}$', labelpad=2)
cb.ax.tick_params(labelsize=9, direction='in')
bx = fig.add_axes([0.11, 0.095, 0.76, 0.165])
bx.axhline(0, color='.6', lw=.6)
bx.plot(th, dmm/304.8, color='black', lw=.9, marker='o', ms=2.6, mfc='white', mew=.7)
bx.set_xlabel(r'Ray angle $\theta$ (deg)'); bx.set_ylabel(r'$\Delta s/D$')
bx.set_ylim(-1.5/304.8, 1.5/304.8); bx.set_yticks([-.004,0,.004]); bx.set_yticklabels(['−0.004','0','0.004'])
bx.tick_params(direction='in', top=True, right=True)
bx.text(.012, .92, '(b)', transform=bx.transAxes, va='top')
bx.text(.975, .88, 'RMS %.4f, %d rays' % (rms/304.8, len(dmm)), transform=bx.transAxes, va='top', ha='right', fontsize=8.5)
for ext in ('pdf',):
    fig.savefig(out_base + '.' + ext, dpi=300)
print('saved', out_base)

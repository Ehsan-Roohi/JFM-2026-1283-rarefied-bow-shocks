"""Figure 11: translational and rotational temperature fields of the DS2V nitrogen
solution at Mach 10, Kn = 0.01 (the Kn = 0.01 case of the Knudsen sweep), T/T_inf with
T_inf = 200 K.  Two identical frames, one shared colour scale 0-25.

usage: python -I fig_temperature_N2.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from common import *

T_INF = 200.0
WIN, XT, YT = (-0.575, 2.05, 1.25), 0.5, 0.5
LEV_T = np.linspace(0.0, 25.0, 21)            # 20 bands of 1.25; freestream (1.0) is inside the first band
NORM_T = BoundaryNorm(LEV_T, CMAP.N)
LINES_T = [5, 10, 15, 20]                    # all well above the freestream value (1)

d, ninf, p = load_case('N2', 0.01)
fig, axs, cax = make_figure(1)
for i, (ax, key, lab) in enumerate(zip(axs, ('TTR', 'TRT'), (r'$T_{\rm tr}/T_\infty$', r'$T_{\rm rot}/T_\infty$'))):
    X, Y, Z = grid_field(d, d[key] / T_INF, WIN)
    assert np.isfinite(Z).all()
    cf = ax.contourf(X, Y, Z, levels=LEV_T, cmap=CMAP, norm=NORM_T)
    cf.set_rasterized(True)
    ax.contour(X, Y, Z, levels=LINES_T, colors='k', linewidths=0.3)
    style_panel(ax, WIN, XT, YT, 'ab'[i], lab, xlabel=True, ylabel=i == 0, label_loc='upper left')
    print(key, p, f'min {Z.min():.3f} max {Z.max():.2f} (cell-centre max {d[key].max() / T_INF:.2f})')
cb = fig.colorbar(cf, cax=cax, ticks=np.arange(0, 26, 5))
cb.ax.tick_params(labelsize=TICK_FS, length=2.5, width=0.6)
cb.set_label(r'$T/T_\infty$', fontsize=LAB_FS, labelpad=3)
cb.ax.minorticks_off()
cb.outline.set_linewidth(0.7)
save(fig, 'Temperature_fields_N2_M10')

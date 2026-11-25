"""Upstream-running (hot) molecules, early temperature rise and P_xx-P_yy anisotropy on the
stagnation line of the cylinder and the flat plate, Ar and N2, Mach 10, Kn 0.01-0.2.
Steady part of the time-resolved runs (last 40 frames, 1.44-2.4 ms). High-particle data where
available (npz_hi/hi_*.npz), otherwise npz/*.npz.
Coordinate: xi = (x - x_rho50)/lambda_inf, x_rho50 where (rho-1)/(rho_RH-1) = 0.5.
usage: python analyze_upstream.py   (in video_runs/)  -> analysis/"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 10, 'axes.linewidth': 0.7})
OUT = pathlib.Path('analysis'); OUT.mkdir(exist_ok=True)
KNS = ['0p01', '0p05', '0p1', '0p2']; KV = [0.01, 0.05, 0.1, 0.2]
RH = {'Ar': 3.884, 'N2': 5.714}; TRH = {'Ar': 32.12, 'N2': 20.39}     # normal-shock rho and T ratios, M=10
COL = dict(zip(KNS, ['#1f4e9c', '#2a9d8f', '#e9a03b', '#c0392b']))
LAST = slice(-40, None)


def load(gas, body, kn):
    p = pathlib.Path(f'npz_hi/hi_{gas}_{body}_Kn{kn}.npz')
    src = 'hi' if p.exists() else 'lo'
    z = np.load(p if p.exists() else f'npz/{gas}_{body}_Kn{kn}.npz')
    m = json.loads(str(z['meta'])); F = z['fields'][LAST].astype(np.float32)
    k = int(np.ceil(m['nx'] / 480))
    x = m['domain'][0] + (np.arange(F.shape[-1]) + 0.5) * m['dx'] * k
    row = np.nanmean(F[:, :, :2, :], axis=(0, 2))            # stagnation row, time mean: [q, x]
    lam = m['lambda_m']
    rhat = (row[0] - 1) / (RH[gas] - 1); That = (row[1] - 1) / (TRH[gas] - 1)
    up = x < 0
    x, rhat, That, row = x[up], rhat[up], That[up], row[:, up]
    x50 = np.interp(0.5, np.maximum.accumulate(np.nan_to_num(rhat)), x)
    # particles: hot fraction, bins of ~0.5 lambda
    A, Hh = z['all'][LAST].sum(0).astype(float), z['hot'][LAST].sum(0).astype(float)
    e = z['edges']; xm = 0.5 * (e[1:] + e[:-1])
    nb = max(1, int(round(0.5 * lam / (xm[1] - xm[0])))); n = len(xm) // nb * nb
    A, Hh, xm = A[-n:].reshape(-1, nb).sum(1), Hh[-n:].reshape(-1, nb).sum(1), xm[-n:].reshape(-1, nb).mean(1)
    ok = A > 200
    return dict(src=src, lam=lam, xi=(x - x50) / lam, rhat=rhat, That=That, Trot=row[2] / (TRH[gas]),
                aniso=row[3], xi_p=(xm[ok] - x50) / lam, hot=Hh[ok] / A[ok])


def first_cross(xi, y, lev):
    y = np.nan_to_num(y); i = np.argmax(y >= lev)
    return np.interp(lev, [y[i - 1], y[i]], [xi[i - 1], xi[i]]) if i > 0 else np.nan


D = {(g, b, k): load(g, b, k) for g in ('Ar', 'N2') for b in ('cylinder', 'plate') for k in KNS}
rows = []
for (g, b, k), c in D.items():
    xr10, xT10 = first_cross(c['xi'], c['rhat'], 0.1), first_cross(c['xi'], c['That'], 0.1)
    xT50 = first_cross(c['xi'], c['That'], 0.5)
    hot = lambda xx: np.interp(xx, c['xi_p'], c['hot'])
    s = c['xi'] < 3
    ia = np.nanargmax(c['aniso'][s])
    rows.append(dict(gas=g, body=b, Kn=k.replace('p', '.'), data=c['src'],
                     hot_at_rho10=hot(xr10), hot_at_rho50=hot(0.0), hot_at_T10=hot(xT10),
                     T_lead_10=xr10 - xT10, T_lead_50=0.0 - xT50,
                     aniso_peak=c['aniso'][s][ia], aniso_peak_xi=c['xi'][s][ia]))
keys = list(rows[0])
with open(OUT / 'upstream_summary.csv', 'w') as f:
    f.write(','.join(keys) + '\n')
    for r in rows: f.write(','.join(f'{r[q]:.4f}' if isinstance(r[q], float) else str(r[q]) for q in keys) + '\n')

# ---- figure 1: stagnation-line structure --------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
for i, g in enumerate(('Ar', 'N2')):
    for k in KNS:
        for b, ls in (('cylinder', '-'), ('plate', '--')):
            c = D[(g, b, k)]
            ax[i, 0].plot(c['xi_p'], 100 * c['hot'], ls, color=COL[k], lw=1.1,
                          label=f"$Kn$={k.replace('p', '.')}" if b == 'cylinder' else None)
            ax[i, 2].plot(c['xi'], c['aniso'], ls, color=COL[k], lw=1.1)
        for kk, lw in (('0p01', 1.3), ('0p2', 1.3)):
            if k != kk: continue
            c = D[(g, 'cylinder', k)]
            ax[i, 1].plot(c['xi'], c['rhat'], '-', color=COL[k], lw=lw, label=rf"$\hat\rho$, $Kn$={k.replace('p', '.')}")
            ax[i, 1].plot(c['xi'], c['That'], ':', color=COL[k], lw=lw + 0.4, label=rf"$\hat T$, $Kn$={k.replace('p', '.')}")
    ax[i, 0].set_ylabel(('Argon' if g == 'Ar' else 'Nitrogen') + '\nhot molecules, $v_x<0$ (%)')
    ax[i, 1].set_ylabel('normalised rise'); ax[i, 2].set_ylabel(r'$(P_{xx}-P_{yy})/p$')
    ax[i, 1].set_ylim(-0.05, 1.6); ax[i, 2].axhline(0, color='0.6', lw=0.5)
    for a in ax[i]: a.axvline(0, color='0.75', lw=0.5); a.set_xlim(-12, 4)
ax[0, 0].legend(frameon=False, fontsize=7.5, loc='upper left'); ax[0, 1].legend(frameon=False, fontsize=7, loc='upper left')
ax[0, 0].set_title('upstream-running molecules'); ax[0, 1].set_title(r'$T$ rises before $\rho$ (cylinder)')
ax[0, 2].set_title(r'anisotropy $(P_{xx}-P_{yy})/p$')
for a in ax[1]: a.set_xlabel(r'$(x-x_{\rho,50})/\lambda_\infty$   (flow $\rightarrow$, body on the right)')
for n, a in enumerate(ax.flat): a.text(0.97, 0.93, f'({"abcdef"[n]})', transform=a.transAxes, ha='right')
fig.tight_layout(); fig.savefig(OUT / 'upstream_structure.png', dpi=200); fig.savefig(OUT / 'upstream_structure.pdf')

# ---- figure 2: trends with Kn -------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(11, 3.1))
for g, mk in (('Ar', 'o'), ('N2', 's')):
    for b, ls, fc in (('cylinder', '-', None), ('plate', '--', 'w')):
        rr = [r for r in rows if r['gas'] == g and r['body'] == b]
        kw = dict(marker=mk, ls=ls, color='k' if g == 'Ar' else 'tab:red', ms=4.5, mfc=fc or ('k' if g == 'Ar' else 'tab:red'),
                  label=f"{'argon' if g == 'Ar' else 'nitrogen'}, {b}")
        ax[0].plot(KV, [100 * r['hot_at_rho10'] for r in rr], **kw)
        ax[1].plot(KV, [100 * r['hot_at_rho50'] for r in rr], **kw)
        ax[2].plot(KV, [r['T_lead_50'] for r in rr], **kw)
        ax[3].plot(KV, [r['aniso_peak'] for r in rr], **kw)
for a, t, yl in zip(ax, ['(a) at 10% of density rise', '(b) at 50% of density rise', r'(c) $T$ ahead of $\rho$ (half rise)',
                         '(d) peak anisotropy'],
                    ['hot molecules (%)', 'hot molecules (%)', r'$(x_{\rho,50}-x_{T,50})/\lambda_\infty$', r'max $(P_{xx}-P_{yy})/p$']):
    a.set(xscale='log', xlabel=r'$Kn_\infty$', title=t, ylabel=yl)
ax[3].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'upstream_trends_Kn.png', dpi=200); fig.savefig(OUT / 'upstream_trends_Kn.pdf')
for r in rows:
    print(f"{r['gas']} {r['body']:8s} Kn={r['Kn']:5s} [{r['data']}] hot@rho10={100*r['hot_at_rho10']:5.1f}% hot@rho50={100*r['hot_at_rho50']:5.1f}% "
          f"Tlead50={r['T_lead_50']:5.2f} lam  Tlead10={r['T_lead_10']:5.2f}  aniso_max={r['aniso_peak']:.2f} at xi={r['aniso_peak_xi']:.1f}")

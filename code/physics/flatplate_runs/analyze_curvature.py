"""Does the wall or the curvature break the collapse?  Cylinder vs flat plate.

Reads the SPARTA grid dumps (fields.<step>.grid) of the matched cases made by
make_cases.py, averages the ten sampling blocks, and on the stagnation line
(cells touching the symmetry plane, x < 0) computes for every case
  * density/temperature profiles in units of lambda_inf,
  * the 10/50/90 % density-rise points, the room between front and body
    (s10/lambda) and the front widths relative to the planar shock,
and for each body across Kn
  * the collapse test in the aligned coordinate xi=(s-s50)/(s10-s90)
    (spread of the profiles about their mean for -1<xi<2), and
  * POD compactness E1 of density and temperature, physical vs aligned.

usage: python analyze_curvature.py <campaign_dir> [--gas Ar] [--out results]
"""
import argparse, glob, json, os, pathlib, re
import numpy as np

# planar SPARTA shock (same gas model, Mach 10), widths in lambda_inf: from
# JFM1283_physics/test/planar_vs_cylinder_mfp.json
PLANAR = {'Ar': dict(w10_90=8.9385, w10_50=4.9406, w50_90=3.9979, Tpeak=32.4327, ratio=3.8835),
          'N2': dict(w10_90=4.9658, w10_50=2.7222, w50_90=2.2436, Tpeak=23.3158, ratio=5.7143)}


def read_dump(path):
    with open(path) as fh:
        lines = fh.readlines()
    i = next(k for k, l in enumerate(lines) if l.startswith('ITEM: CELLS'))
    cols = lines[i].split()[2:]
    data = np.loadtxt(lines[i + 1:])
    return cols, data


def case_profile(cdir):
    meta = json.load(open(os.path.join(cdir, 'case.json')))
    files = sorted(glob.glob(os.path.join(cdir, 'fields.*.grid')), key=lambda f: int(re.findall(r'\.(\d+)\.grid', f)[0]))
    files = [f for f in files if int(re.findall(r'\.(\d+)\.grid', f)[0]) > meta['warm_steps']]
    if len(files) < meta['nblocks']:
        return meta, None
    acc = None
    for f in files[-meta['nblocks']:]:
        cols, d = read_dump(f)
        acc = d if acc is None else acc + d
    d = acc / meta['nblocks']
    c = {n: d[:, k] for k, n in enumerate(cols)}
    dy = meta['dy']
    band = (c['ylo'] < 2.0 * dy - 1e-12) & (c['xc'] < 0)
    x = c['xc'][band]
    nrho = c['f_avg[2]'][band]
    T = c['f_avg[8]'][band]
    xs = np.unique(np.round(x, 9))
    prof_n = np.array([nrho[np.isclose(x, v)].mean() for v in xs])
    prof_T = np.array([T[np.isclose(x, v)].mean() for v in xs])
    s = -xs                                     # distance upstream of the body (x=0 at both bodies)
    order = np.argsort(-s)                      # upstream -> downstream
    return meta, dict(s=s[order], rho=prof_n[order] / meta['nrho'], T=prof_T[order] / 200.0)


def crossing(s, q, level):
    for i in range(1, len(q)):
        if q[i - 1] < level <= q[i]:
            f = (level - q[i - 1]) / (q[i] - q[i - 1])
            return s[i - 1] + f * (s[i] - s[i - 1])
    return np.nan


def pod_e1(M):
    M = M - M.mean(axis=0)
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[0] ** 2 / (sv ** 2).sum()) if sv.sum() > 0 else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('campaign')
    ap.add_argument('--gas', default='Ar')
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    out = pathlib.Path(a.campaign) / a.out / a.gas
    out.mkdir(parents=True, exist_ok=True)
    pl = PLANAR[a.gas]
    lv = {p: 1 + p * (pl['ratio'] - 1) for p in (0.1, 0.5, 0.9)}
    prof, rows = {}, []
    for cdir in sorted(glob.glob(os.path.join(a.campaign, 'cases', f'{a.gas}_*'))):
        meta, p = case_profile(cdir)
        if p is None:
            print('incomplete:', os.path.basename(cdir))
            continue
        lam = meta['lambda_m']
        s10, s50, s90 = (crossing(p['s'], p['rho'], lv[q]) for q in (0.1, 0.5, 0.9))
        r = dict(body=meta['body'], Kn=meta['Kn'], room=s10 / lam / pl['w10_90'],
                 full=(s10 - s90) / lam / pl['w10_90'], up=(s10 - s50) / lam / pl['w10_50'],
                 down=(s50 - s90) / lam / pl['w50_90'], Tpeak=float(np.nanmax(p['T'])) / pl['Tpeak'],
                 s10_mm=1e3 * s10, s50_mm=1e3 * s50, s90_mm=1e3 * s90)
        rows.append(r)
        prof[(meta['body'], meta['Kn'])] = dict(p, s10=s10, s50=s50, s90=s90, lam=lam)
        print(f"{meta['body']:8s} Kn={meta['Kn']:<6g} room/w_pl={r['room']:.2f} full={r['full']:.3f} "
              f"up={r['up']:.3f} down={r['down']:.3f} Tpeak/planar={r['Tpeak']:.3f}")
    json.dump(rows, open(out / 'truncation.json', 'w'), indent=1)

    # share of the departure from the planar shock due to the wall alone (plate)
    # and to curvature (cylinder minus plate), at equal Kn
    shares = []
    for kn in sorted({r['Kn'] for r in rows}):
        c = next((r for r in rows if r['body'] == 'cylinder' and r['Kn'] == kn), None)
        p = next((r for r in rows if r['body'] == 'plate' and r['Kn'] == kn), None)
        if not c or not p:
            continue
        item = dict(Kn=kn)
        for key in ('full', 'up', 'down'):
            dc, dp = 1 - c[key], 1 - p[key]
            item[f'departure_cyl_{key}'] = dc
            item[f'departure_plate_{key}'] = dp
            item[f'wall_share_{key}'] = dp / dc if dc > 0.03 else float('nan')
        shares.append(item)
        print(f"Kn={kn:<6g} departure cyl/plate (full width): {item['departure_cyl_full']:.3f}/{item['departure_plate_full']:.3f}"
              f"  -> wall share {item['wall_share_full']:.2f}, curvature share {1 - item['wall_share_full']:.2f}")
    json.dump(shares, open(out / 'wall_vs_curvature.json', 'w'), indent=1)

    # Shock-centred coordinate exactly as in the nozzle study (Roohi & Mahdavi,
    # Phys. Fluids 38, 082008, 2026): xi_j = (x - x_s)/delta_j, x_s at the maximum
    # density gradient of the outer front, delta_j = (rho_d - rho_u)/max|drho/dx|,
    # rho_u = 1 (freestream), rho_d at the first gradient minimum downstream of x_s.
    # The near-wall rise is excluded by searching only around the 10-90 % crossings.
    def shock_centre(q):
        """Outer-front rule of the paper: the outermost local peak of the density
        gradient (rho > 1.2) that is separated from the near-wall rise by a local
        gradient minimum at least 5 % below the peak.  Returns None if there is no
        separated front (the gradient keeps rising towards the wall)."""
        z = -q['s']                                       # downstream coordinate (x), upstream first
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2); k /= k.sum()
        rho = np.convolve(np.pad(q['rho'], 6, mode='edge'), k, mode='valid')
        g = np.gradient(rho, z)
        n = len(g)
        for i in range(2, n - 2):
            if rho[i] <= 1.2 or not (g[i] >= g[i - 1] and g[i] > g[i + 1]):
                continue
            j = i + 1
            while j < n - 1 and not (g[j] <= g[j - 1] and g[j] <= g[j + 1]):
                j += 1
            if j >= n - 1 or g[j] > 0.95 * g[i]:
                continue                                  # no separating minimum -> not a separate front
            if q['s'][j] < 2 * (q['s'][0] - q['s'][1]):
                continue                                  # minimum lies at the wall
            return z[i], (rho[j] - 1.0) / g[i], z
        return None

    summary, aligned = {}, {}
    xi = np.linspace(-2.0, 2.0, 401)
    fronts = {}
    for k, q in prof.items():
        sc = shock_centre(q)
        fronts[k] = sc
        room = next(r['room'] for r in rows if (r['body'], r['Kn']) == k)
        print(f"{k[0]:8s} Kn={k[1]:<6g} room/w_pl={room:.2f}  separate front: "
              + (f"yes, x_s={1e3*sc[0]:.1f} mm, delta_j={1e3*sc[1]:.1f} mm" if sc else 'no'))
    for body in ('cylinder', 'plate'):
        keys = sorted(k for k in prof if k[0] == body and fronts[k] is not None)
        if len(keys) < 2:
            continue
        A_rho, A_T, B_rho, P_rho, P_T = [], [], [], [], []
        smax = min(prof[k]['s'].max() for k in keys)
        gphys = np.linspace(0.0, smax, 400)
        for k in keys:
            q = prof[k]
            xs, delta, z = fronts[k]
            zj = (z - xs) / delta                         # nozzle coordinate, downstream positive
            o = np.argsort(zj)
            A_rho.append(np.interp(xi, zj[o], q['rho'][o]))
            A_T.append(np.interp(xi, zj[o], q['T'][o]))
            zb = (q['s50'] - q['s']) / (q['s10'] - q['s90'])   # robustness: 10-90 % scaling
            ob = np.argsort(zb)
            B_rho.append(np.interp(xi, zb[ob], q['rho'][ob]))
            o2 = np.argsort(q['s'])                       # physical coordinate (m), common window
            P_rho.append(np.interp(gphys, q['s'][o2], q['rho'][o2]))
            P_T.append(np.interp(gphys, q['s'][o2], q['T'][o2]))
            aligned[k] = (zj, xs, delta)
        A_rho, A_T, B_rho = np.array(A_rho), np.array(A_T), np.array(B_rho)
        spread = lambda M: float(np.sqrt(((M - M.mean(0)) ** 2).mean()) / (M.max() - M.min()))
        summary[body] = dict(Kn=[k[1] for k in keys],
                             no_front_Kn=sorted(k[1] for k in prof if k[0] == body and fronts[k] is None),
                             xs_mm=[1e3 * aligned[k][1] for k in keys],
                             delta_j_mm=[1e3 * aligned[k][2] for k in keys],
                             spread_rho=spread(A_rho), spread_T=spread(A_T), spread_rho_10_90=spread(B_rho),
                             E1_rho_aligned=pod_e1(A_rho), E1_T_aligned=pod_e1(A_T),
                             E1_rho_physical=pod_e1(np.array(P_rho)), E1_T_physical=pod_e1(np.array(P_T)))
        print(body, json.dumps(summary[body]))
    json.dump(summary, open(out / 'collapse.json', 'w'), indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 10})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    cmap = plt.get_cmap('viridis')
    for j, body in enumerate(('cylinder', 'plate')):
        keys = sorted(k for k in prof if k[0] == body)
        for n, k in enumerate(keys):
            q = prof[k]
            z = aligned[k][0] if k in aligned else (q['s50'] - q['s']) / (q['s10'] - q['s90'])
            col = cmap(n / max(1, len(keys) - 1))
            ls = ['-', '--', '-.', ':', (0, (5, 1))][n % 5]
            ax[0, j].plot(z, q['rho'], ls=ls, color=col, label=f'Kn={k[1]:g}')
            ax[1, j].plot(z, q['T'], ls=ls, color=col)
        ax[0, j].set_title(body)
        ax[1, j].set_xlabel(r'$\xi_j=(x-x_s)/\delta_j$  (downstream $\rightarrow$, as in the nozzle study)')
        ax[0, j].set_xlim(-4, 3)
    ax[0, 0].set_ylabel(r'$\rho/\rho_\infty$')
    ax[1, 0].set_ylabel(r'$T_{\rm tr}/T_\infty$')
    ax[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'collapse_cylinder_vs_plate.png', dpi=200)
    fig.savefig(out / 'collapse_cylinder_vs_plate.pdf')

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for body, mk in (('cylinder', 'o'), ('plate', 's')):
        rr = sorted((r for r in rows if r['body'] == body), key=lambda r: r['room'])
        for key, col in (('up', '#1f4e9c'), ('full', 'k'), ('down', '#c0392b')):
            ax.plot([r['room'] for r in rr], [r[key] for r in rr], marker=mk, color=col, lw=0.9,
                    mfc=col if body == 'cylinder' else 'white', label=f'{body}: {key}')
    ax.axhline(1, color='0.6', lw=0.6)
    ax.set_xlabel('room between front and body / planar shock width')
    ax.set_ylabel('width / planar width')
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'truncation_cylinder_vs_plate.png', dpi=200)
    fig.savefig(out / 'truncation_cylinder_vs_plate.pdf')


if __name__ == '__main__':
    main()

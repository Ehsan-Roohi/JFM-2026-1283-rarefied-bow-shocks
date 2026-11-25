"""Cylinder front versus the planar shock in units of the freestream mean free path.

A planar normal shock has no external length: in units of lambda_inf its
structure depends only on M and the molecular model, for every Kn.  The cylinder
fronts of the DS2V Knudsen sweep (170-degree ray) are therefore compared with
ONE planar SPARTA profile (Mach 10, same gas model), each aligned at its own
10 % density-rise crossing.  If the cylinder front were a planar shock, all
curves would coincide.  Where they part, the body is acting.

usage: python planar_in_mfp_units.py --planar-dir DIR --ds2v-root ROOT --out DIR
"""
import argparse, json, pathlib, re
import numpy as np
from scipy.interpolate import LinearNDInterpolator

R, D = 0.1524, 0.3048
LAM001 = {'Ar': 3.0472e-3, 'N2': 3.0488e-3}          # planar mfp column, = VHS lambda at Kn=0.01
RATIO = {'Ar': (8 / 3 * 100) / (2 / 3 * 100 + 2), 'N2': (2.4 * 100) / (0.4 * 100 + 2)}
FOLDER = {'Ar': ['Ar-Bird Prop-Kn', 'Ar-Bird Prop-Kn-Part2'], 'N2': ['N2-Bird Prop-Kn']}
KNS = [0.01, 0.025, 0.05, 0.1]
TH = np.deg2rad(170.0)
DIR = np.array([np.cos(TH), np.sin(TH)])


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
    raise SystemExit('no number density in ' + path)


def first_crossing(coord, q, level):
    """first index scanning in the given order where q >= level; linear interpolation."""
    for i in range(1, len(q)):
        if np.isfinite(q[i - 1]) and np.isfinite(q[i]) and q[i - 1] < level <= q[i]:
            f = (level - q[i - 1]) / (q[i] - q[i - 1])
            return coord[i - 1] + f * (coord[i] - coord[i - 1])
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planar-dir', required=True)
    ap.add_argument('--ds2v-root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, curves = {}, {}
    for gas in ('Ar', 'N2'):
        z = np.load(pathlib.Path(a.planar_dir) / f'{gas}_planar_long.npz')
        x, m = z['x'], z['mean']
        rho = m[:, 0] / m[0, 0]
        T = m[:, 2] / 200.0
        lam = LAM001[gas]
        lv = {p: 1 + p * (RATIO[gas] - 1) for p in (0.1, 0.5, 0.9)}
        x10, x50, x90 = (first_crossing(x, rho, lv[p]) for p in (0.1, 0.5, 0.9))
        curves[(gas, 'planar')] = ((x - x10) / lam, rho, T)
        res[(gas, 'planar')] = dict(w10_90_mfp=(x90 - x10) / lam, w10_50_mfp=(x50 - x10) / lam, w50_90_mfp=(x90 - x50) / lam,
                                    rho_end=float(rho[-1]), T_peak=float(T.max()))
        for kn in KNS:
            root = pathlib.Path(a.ds2v_root)
            d = read_ds2ff(root / f'{gas}_Kn{kn:g}.dat')
            ninf = n_inf_from_input(root / f'{gas}_Kn{kn:g}.DS2VD')
            pts = np.c_[d['X'], d['Y']]
            s = np.linspace(0.0005, 0.6, 2400)
            xy = np.array([R, 0.0]) + np.outer(R + s, DIR)
            nd = LinearNDInterpolator(pts, d['ND'])(xy) / ninf
            tt = LinearNDInterpolator(pts, d['TTR'])(xy) / 200.0
            ok = np.isfinite(nd)
            s, nd, tt = s[ok], nd[ok], tt[ok]
            lamk = lam * kn / 0.01
            order = np.argsort(-s)                 # upstream -> downstream
            so, ndo = s[order], nd[order]
            s10, s50, s90 = (first_crossing(so, ndo, lv[p]) for p in (0.1, 0.5, 0.9))
            curves[(gas, kn)] = ((s10 - s) / lamk, nd, tt)
            res[(gas, kn)] = dict(n_inf=ninf, lambda_mm=1e3 * lamk, s10_mm=1e3 * s10, s50_mm=1e3 * s50, s90_mm=1e3 * s90,
                                  room_mfp=s10 / lamk, w10_90_mfp=(s10 - s90) / lamk if np.isfinite(s90) else np.nan,
                                  w10_50_mfp=(s10 - s50) / lamk if np.isfinite(s50) else np.nan,
                                  rho_max=float(np.nanmax(nd)), T_peak=float(np.nanmax(tt)))
    rows = []
    for (gas, key), r in res.items():
        r = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        rows.append(dict(gas=gas, case=str(key), **r))
        print(gas, key, r)
    json.dump(rows, open(out / 'planar_vs_cylinder_mfp.json', 'w'), indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10})
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
    cols = {0.01: '#1f4e9c', 0.025: '#2a9d8f', 0.05: '#e9a03b', 0.1: '#c0392b'}
    for j, gas in enumerate(('Ar', 'N2')):
        for i, (idx, lab) in enumerate(((1, r'$\rho/\rho_\infty$'), (2, r'$T_{\rm tr}/T_\infty$'))):
            a_ = ax[i, j]
            xi, *q = curves[(gas, 'planar')]
            a_.plot(xi, q[idx - 1], 'k--', lw=1.3, label='planar shock', zorder=5)
            for kn in KNS:
                if (gas, kn) not in curves:
                    continue
                xi, *q = curves[(gas, kn)]
                a_.plot(xi, q[idx - 1], color=cols[kn], lw=1.3, label=rf'$Kn_\infty={kn:g}$')
                a_.axvline(res[(gas, kn)]['room_mfp'], color=cols[kn], lw=0.7, ls=':')
            a_.set_xlim(-15, 40)
            if i == 1:
                a_.set_xlabel(r'distance downstream of the 10% point, $/\lambda_\infty$')
            if j == 0:
                a_.set_ylabel(lab)
            if i == 0:
                a_.set_title('argon' if gas == 'Ar' else 'nitrogen', fontsize=10)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=9, handlelength=1.8,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / 'planar_vs_cylinder_mfp.pdf')
    fig.savefig(out / 'planar_vs_cylinder_mfp.png', dpi=200)


if __name__ == '__main__':
    main()

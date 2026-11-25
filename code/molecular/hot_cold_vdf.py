"""Hot/cold decomposition of the molecular velocity distributions on the
cylinder (170-degree ray) and in the planar shock, following the definition of
the PINN kinetic-shock work: molecules with v < 0 have left the shock and run
upstream ("hot"); molecules with v > 0 form the incoming beam side ("cold").

Velocity along the sampling traverse: planar v_x, cylinder v.n with n the inward
ray direction (0.9848, -0.1736, 0).  All quantities are 1-D marginals, i.e.
reduced distributions g(v), as in the PINN work.

usage:
  python hot_cold_vdf.py --planar planar_velocity_distributions.csv \
      --cylinder Ar=<csv> --cylinder N2=<csv> --out <dir> [--stride 10]

<csv> for the cylinder is either the preliminary file (one snapshot) or the
final <gas>_cylinder_final CSV, from which the pooled 'all_10_blocks' window is
used at the requested stride.
"""
import argparse, csv, json, math, pathlib
from collections import defaultdict
import numpy as np

KB = 1.380649e-23
MASS = {'Ar': 6.63e-26, 'N2': 4.65e-26}
UINF = {'Ar': 2634.1, 'N2': 2883.3}
STATIONS = ['upstream_reference', 's10', 's50', 's90', 'downstream_reference']
LABEL = {'upstream_reference': 'upstream', 's10': '10%', 's50': '50%', 's90': '90%',
         'downstream_reference': 'wall side'}


def load_planar(path):
    """-> {(gas, station): (edges_lo, edges_hi, prob)} in v/U, window 'all'."""
    out = defaultdict(list)
    for r in csv.DictReader(open(path, encoding='utf-8')):
        if r['window'] != 'all':
            continue
        gas = r['case'].split('_')[0]
        out[(gas, r['station'])].append(r)
    res = {}
    for key, rows in out.items():
        lo, hi, p = [], [], []
        for r in rows:
            lo.append(float(r['lo_Uinf']) if r['lo_Uinf'] else -np.inf)
            hi.append(float(r['hi_Uinf']) if r['hi_Uinf'] else np.inf)
            p.append(float(r['probability']))
        res[key] = (np.array(lo), np.array(hi), np.array(p))
    return res


def load_cylinder(path, gas, stride):
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['gas'] == gas]
    if not rows:
        raise SystemExit(f'no {gas} rows in {path}')
    windows = {r['window'] for r in rows}
    if 'all_10_blocks' in windows:
        rows = [r for r in rows if r['window'] == 'all_10_blocks' and r['stride_us'] == str(stride)]
        kind = f'final, 4-6 ms, {stride} us stride'
    else:
        kind = 'preliminary, single snapshot at 4.001 ms'
    res, meta = {}, {}
    by = defaultdict(list)
    for r in rows:
        if r['velocity_component'] == 'inward_station_normal_vn':
            by[r['station']].append(r)
    for st, rr in by.items():
        lo = np.array([float(r['lower_v_over_Uinf']) if r['lower_v_over_Uinf'] else -np.inf for r in rr])
        hi = np.array([float(r['upper_v_over_Uinf']) if r['upper_v_over_Uinf'] else np.inf for r in rr])
        p = np.array([float(r['physical_probability_in_bin']) for r in rr])
        res[(gas, st)] = (lo, hi, p)
        meta[st] = dict(visits=int(rr[0]['station_simulator_visits']), wall_distance_mm=1e3 * float(rr[0]['wall_distance_m']),
                        snapshots=int(rr[0]['snapshot_count']))
    return res, meta, kind


def stats(gas, lo, hi, p):
    """1-D moments in m/s from a binned marginal (interior bins only; tails reported)."""
    fin = np.isfinite(lo) & np.isfinite(hi)
    tails = float(p[~fin].sum())
    v = 0.5 * (lo[fin] + hi[fin]) * UINF[gas]
    w = p[fin] / p[fin].sum()
    u = float((w * v).sum())
    c = v - u
    var = float((w * c ** 2).sum())
    hot = v < 0
    c4 = w * c ** 4
    cold_w = w[~hot] / w[~hot].sum()
    cold_v = v[~hot]
    cold_var = float((cold_w * (cold_v - (cold_w * cold_v).sum()) ** 2).sum())
    return dict(u_over_U=u / UINF[gas], T_1D_K=MASS[gas] * var / KB, hot_fraction=float(w[hot].sum()),
                hot_share_c4=float(c4[hot].sum() / c4.sum()) if c4.sum() > 0 else float('nan'),
                T_1D_without_hot_K=MASS[gas] * cold_var / KB, tail_probability=tails)


def smooth(y, s=1.5):
    """light Gaussian smoothing of a sampled PDF, for plotting only (statistics use raw data)"""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(np.asarray(y, float), s, mode='nearest')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planar', required=True)
    ap.add_argument('--cylinder', action='append', required=True, help='GAS=path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=10)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    planar = load_planar(a.planar)
    cyl, meta, kinds = {}, {}, {}
    for item in a.cylinder:
        gas, path = item.split('=', 1)
        d, m, k = load_cylinder(path, gas, a.stride)
        cyl.update(d)
        meta[gas], kinds[gas] = m, k

    table = []
    for gas in ('Ar', 'N2'):
        for st in STATIONS:
            for geom, src in (('cylinder', cyl), ('planar', planar)):
                if (gas, st) not in src:
                    continue
                s = stats(gas, *src[(gas, st)])
                row = dict(gas=gas, geometry=geom, station=LABEL[st], **{k: round(v, 5) for k, v in s.items()})
                if geom == 'cylinder':
                    row['wall_distance_mm'] = round(meta[gas][st]['wall_distance_mm'], 2)
                    row['simulator_visits'] = meta[gas][st]['visits']
                    row['data'] = kinds[gas]
                else:
                    row['data'] = 'planar, 4-6 ms, 10 us stride, 10 blocks'
                table.append(row)
    keys = ['gas', 'geometry', 'station', 'wall_distance_mm', 'u_over_U', 'T_1D_K', 'T_1D_without_hot_K',
            'hot_fraction', 'hot_share_c4', 'tail_probability', 'simulator_visits', 'data']
    with open(out / 'hot_cold_table.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(table)
    for r in table:
        print(f"{r['gas']:3s} {r['geometry']:9s} {r['station']:10s} u/U={r['u_over_U']:7.4f} T={r['T_1D_K']:8.1f} K "
              f"T(no hot)={r['T_1D_without_hot_K']:8.1f} K hot={100*r['hot_fraction']:6.2f}% "
              f"hot share of c^4={100*r['hot_share_c4']:6.2f}%")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                         'font.size': 10, 'axes.linewidth': 0.7})
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.05]))
    for i, gas in enumerate(('Ar', 'N2')):
        for j, st in enumerate(['s10', 's50', 's90']):
            ax = axes[i, j]
            for geom, src, ls, col in (('cylinder', cyl, '-', 'k'), ('planar', planar, '--', '0.45')):
                if (gas, st) not in src:
                    continue
                lo, hi, p = src[(gas, st)]
                fin = np.isfinite(lo) & np.isfinite(hi)
                x = 0.5 * (lo[fin] + hi[fin])
                pdf = smooth(p[fin] / (hi[fin] - lo[fin]))
                ax.plot(x, pdf, ls, color=col, lw=1.0, label=geom)
                if geom == 'cylinder':
                    ax.fill_between(x, pdf, where=x < 0, color='tab:red', alpha=0.25, lw=0, step='mid')
                    ax.fill_between(x, pdf, where=x >= 0, color='tab:blue', alpha=0.12, lw=0, step='mid')
            ax.axvline(0, color='0.6', lw=0.5)
            ax.set_xlim(-1.5, 1.6)
            if i == 1:
                ax.set_xlabel(r'$v_n/U_\infty$')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(('Argon' if gas == 'Ar' else 'Nitrogen') + '\n' + r'PDF of $v_n/U_\infty$')
            if i == 0:
                ax.set_title(f'{LABEL[st]} of density rise')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
    short = {'upstream': 'up', '10%': '10%', '50%': '50%', '90%': '90%', 'wall side': 'wall'}
    for i, gas in enumerate(('Ar', 'N2')):          # hot fraction at every station
        ax = axes[i, 3]
        for geom, mk, col in (('cylinder', 'o', 'k'), ('planar', 's', '0.45')):
            rr = [r for r in table if r['gas'] == gas and r['geometry'] == geom]
            ax.plot(range(len(rr)), [100 * r['hot_fraction'] for r in rr], mk + ('-' if geom == 'cylinder' else '--'),
                    color=col, ms=4, lw=1, mfc=col if geom == 'cylinder' else 'w', label=geom)
        ax.set_xticks(range(len(rr)), [short[r['station']] for r in rr] if i == 1 else [])
        ax.set_ylim(-2, 45); ax.set_ylabel(r'hot fraction, $v_n<0$ (%)')
        if i == 0: ax.set_title('all stations')
        if i == 1: ax.set_xlabel('station')
    axes[0, 3].legend(frameon=False, fontsize=8, loc='lower right')
    note = '; '.join(f'{g}: {k}' for g, k in kinds.items())
    for k, ax in enumerate(axes.flat):
        ax.text(0.97, 0.95, f'({"abcdefgh"[k]})', transform=ax.transAxes, ha='right', va='top')
    print('provenance:', note)
    fig.tight_layout()
    fig.savefig(out / 'vdf_hot_cold_cylinder_planar.pdf')
    fig.savefig(out / 'vdf_hot_cold_cylinder_planar.png', dpi=200)
    json.dump(dict(cylinder_data=kinds, definition='hot: v_n<0 (running upstream), cold: v_n>=0; 1-D reduced marginals; '
                   'c = v - u for the c^4 share; T_1D = m var(v)/k_B', stride_us=a.stride),
              open(out / 'hot_cold_provenance.json', 'w'), indent=2)


if __name__ == '__main__':
    main()

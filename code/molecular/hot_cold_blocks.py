"""Hot/cold molecules from the compact block files (block_01..10.npz) of one case,
pooled over the ten blocks at one snapshot stride.  Reads the marginals of the
velocity along the traverse ("normal"), the tangential marginal and, for
nitrogen, the rotational energy summed in each normal-velocity bin.

Hot: v_n < 0 (running upstream); cold: v_n >= 0.

usage: python hot_cold_blocks.py <case_dir_with_block_npz> [--stride 10] [--csv out.csv]
"""
import argparse, csv, json, pathlib
import numpy as np

KB = 1.380649e-23


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case_dir')
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('--csv')
    a = ap.parse_args()
    files = sorted(pathlib.Path(a.case_dir).glob('block_[0-9][0-9].npz'))
    if len(files) != 10:
        raise SystemExit(f'expected 10 blocks, found {len(files)} in {a.case_dir}')
    W = R = None
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z['metadata_json']))
        j = list(z['strides']).index(a.stride)
        w = z['weight'][j]
        r = z['rot_bin'][j] if meta['rotation'] else None
        W = w if W is None else W + w
        if r is not None:
            R = r if R is None else R + r
    marg = {name: (start, nbin, lo, hi) for name, start, nbin, lo, hi in meta['marginals']}
    U = meta['speed_scale_m_s']
    m = meta['mass_kg']
    rows = []
    for k, st in enumerate(meta['station_names']):
        s0, nb, lo, hi = marg['normal']
        wn = W[k, s0:s0 + nb + 2]           # underflow, nb interior bins, overflow
        edges = np.linspace(lo, hi, nb + 1)
        x = 0.5 * (edges[:-1] + edges[1:])
        wi = wn[1:-1]
        tot = wn.sum()
        hot = x < 0
        row = dict(case=meta['case'], station=st, stride_us=a.stride, total_weight=tot,
                   hot_fraction=(wi[hot].sum() + wn[0]) / tot)
        t0, tn, tlo, thi = marg['tangent']
        wt = W[k, t0 + 1:t0 + tn + 1]
        te = np.linspace(tlo, thi, tn + 1)
        tx = 0.5 * (te[:-1] + te[1:])
        row['mean_vt_over_U'] = float((wt * tx).sum() / wt.sum())
        if R is not None:
            rn = R[k]
            ri = rn[1:-1]
            for name, sel in (('cold', ~hot), ('hot', hot)):
                ww = wi[sel].sum()
                row[f'Trot_{name}_K'] = float(ri[sel].sum() / ww / KB) if ww > 0 else float('nan')
            row['Trot_all_K'] = float(rn.sum() / tot / KB)
        rows.append(row)
        extra = ''.join(f"  Trot {n}={row[f'Trot_{n}_K']:7.1f} K" for n in ('all', 'cold', 'hot') if f'Trot_{n}_K' in row)
        print(f"{meta['case']:12s} {st:21s} hot={100*row['hot_fraction']:6.2f}%  <v_t>/U={row['mean_vt_over_U']:+.4f}{extra}")
    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == '__main__':
    main()

"""Run on Unity: keep only the stagnation-line rows (ylo < 2*dy, xc < 0) of every
grid dump of one case, so that only a small file has to be transferred.
Plain Python 3, no third-party modules.

usage: python3 extract_band.py <case_dir>        writes <case_dir>/band/fields.*.grid
"""
import glob, json, os, sys

case = sys.argv[1]
meta = json.load(open(os.path.join(case, 'case.json')))
lim = 2.0 * meta['dy'] - 1e-12
out = os.path.join(case, 'band')
os.makedirs(out, exist_ok=True)
n = 0
for path in sorted(glob.glob(os.path.join(case, 'fields.*.grid'))):
    dst = os.path.join(out, os.path.basename(path))
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(path):
        n += 1
        continue
    with open(path) as src, open(dst + '.tmp', 'w') as w:
        cols = None
        for line in src:
            if cols is None:
                w.write(line)
                if line.startswith('ITEM: CELLS'):
                    cols = line.split()[2:]
                    ix, iy = cols.index('xc'), cols.index('ylo')
                continue
            f = line.split()
            if float(f[iy]) < lim and float(f[ix]) < 0:
                w.write(line)
    os.replace(dst + '.tmp', dst)
    n += 1
print(f'{case}: {n} band dumps')

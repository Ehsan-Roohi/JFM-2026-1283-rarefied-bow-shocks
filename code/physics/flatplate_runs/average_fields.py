"""Run on Unity: average the sampling-block grid dumps of one case and write a
compact gzipped CSV (xc, yc, flow fraction, nrho, u, v, temp, trot).  Grids larger
than 120 000 cells are coarsened 2x2 with flow-volume weighting.  Plain Python 3.

usage: python3 average_fields.py <case_dir>     writes <case_dir>/avg_fields.csv.gz
"""
import glob, gzip, json, os, re, sys

case = sys.argv[1]
m = json.load(open(os.path.join(case, 'case.json')))
xmin, xmax, _, ymax = m['domain']
dx, dy, nx, ny = m['dx'], m['dy'], m['nx'], m['ny']
c = 2 if nx * ny > 120000 else 1
files = sorted(glob.glob(os.path.join(case, 'fields.*.grid')), key=lambda f: int(re.findall(r'\.(\d+)\.grid', f)[0]))
files = [f for f in files if int(re.findall(r'\.(\d+)\.grid', f)[0]) > m['warm_steps']][-m['nblocks']:]
acc = {}
for f in files:
    with open(f) as src:
        cols = None
        for line in src:
            if cols is None:
                if line.startswith('ITEM: CELLS'):
                    cols = line.split()[2:]
                    ix, iy, iv = cols.index('xc'), cols.index('yc'), cols.index('c_geom')
                    q = [cols.index(k) for k in ('f_avg[2]', 'f_avg[4]', 'f_avg[5]', 'f_avg[8]', 'f_avg[7]')]
                continue
            p = line.split()
            i = int((float(p[ix]) - xmin) / dx) // c
            j = int(float(p[iy]) / dy) // c
            vol = float(p[iv])
            a = acc.setdefault((i, j), [0.0] * 7)
            a[0] += vol
            a[1] += vol * dx * dy                     # placeholder for full-cell volume
            for k, col in enumerate(q):
                a[2 + k] += vol * float(p[col])
out = os.path.join(case, 'avg_fields.csv.gz')
nb = len(files)
with gzip.open(out + '.tmp', 'wt') as w:
    w.write(f'# {m["body"]} {m["gas"]} Kn={m["Kn"]} blocks={nb} coarsen={c} dx={dx*c} dy={dy*c} nrho_inf={m["nrho"]}\n')
    w.write('xc,yc,flow_fraction,nrho,u,v,temp,trot\n')
    full = dx * dy * c * c * nb
    for (i, j), a in sorted(acc.items()):
        xc = xmin + (i + 0.5) * dx * c
        yc = (j + 0.5) * dy * c
        if a[0] <= 0:
            w.write(f'{xc:.6g},{yc:.6g},0,,,,,\n')
            continue
        vals = [a[2 + k] / a[0] for k in range(5)]
        w.write(f'{xc:.6g},{yc:.6g},{a[0]/full:.4g},' + ','.join(f'{v:.6g}' for v in vals) + '\n')
os.replace(out + '.tmp', out)
print(case, 'blocks', nb, 'cells', len(acc), 'coarsen', c, 'bytes', os.path.getsize(out))

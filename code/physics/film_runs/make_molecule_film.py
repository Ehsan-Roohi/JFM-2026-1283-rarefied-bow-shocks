"""Molecule film in the style of the APS films: every DSMC molecule on the stagnation line of the
cylinder (top) and the flat plate (bottom), followed by its id. Blue: moving downstream (v_x>0);
orange: moving upstream (v_x<0); ahead of the front an orange molecule draws its recent path.
Dotted line: counting station 5 mean free paths ahead of the 10% density point; the counter gives the
molecules that have crossed it upstream since the film began. Strip under each panel: stagnation-line
density (blue) and temperature (red), each scaled 0 upstream to 1 for the ideal normal shock.
usage: python make_molecule_film.py <gas> <kn_tag e.g. 0p01> [--suffix _test] [--frames N]"""
import collections, json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})
BG, FG, DIM = '#0b1220', '#e8edf5', '#8a96ab'
BLUE, ORANGE, RED, CYAN = '#4f9dff', '#ff9a3c', '#ff5c5c', '#62c3ff'
RH = {'Ar': 3.884, 'N2': 5.714}; TRH = {'Ar': 32.12, 'N2': 20.39}
GN = {'Ar': 'argon', 'N2': 'nitrogen'}
HERE = pathlib.Path(__file__).resolve().parents[3] / 'data/physics/film_runs'
VID = HERE.parent / 'video_runs'

gas, kn = sys.argv[1], sys.argv[2]
suffix = sys.argv[sys.argv.index('--suffix') + 1] if '--suffix' in sys.argv else ''
maxf = int(sys.argv[sys.argv.index('--frames') + 1]) if '--frames' in sys.argv else None


def profile(body):
    """steady stagnation-line rho_hat, T_hat and the 10/50% density points from the high-particle movie run"""
    z = np.load(VID / f'npz_hi/hi_{gas}_{body}_Kn{kn}.npz'); m = json.loads(str(z['meta']))
    F = z['fields'][-40:].astype(np.float32); k = int(np.ceil(m['nx'] / 480))
    x = m['domain'][0] + (np.arange(F.shape[-1]) + 0.5) * m['dx'] * k
    row = np.nanmean(F[:, :, :2, :], axis=(0, 2)); up = x < 0
    x, r, T = x[up], (row[0, up] - 1) / (RH[gas] - 1), (row[1, up] - 1) / (TRH[gas] - 1)
    rr = np.maximum.accumulate(np.nan_to_num(r))
    return x, r, T, np.interp(0.1, rr, x), np.interp(0.5, rr, x)


def load(body):
    p = HERE / 'mol' / f'film_{gas}_{body}_Kn{kn}{suffix}_mol.npz'
    if not p.exists() and suffix == "_test":
        p = HERE / 'mol' / f'film_{gas}_cylinder_Kn{kn}{suffix}_mol.npz'      # test: only one body exists
    z = np.load(p); d = {k: z[k] for k in z.files}                 # materialise once (npz access is lazy)
    return d, json.loads(str(d['meta']))


cases = {}
for body in ('cylinder', 'plate'):
    z, m = load(body); px, pr, pT, x10, x50 = profile(body)
    cases[body] = dict(z=z, m=m, off=z['offsets'], px=px, pr=pr, pT=pT, x10=x10, x50=x50)
lam = cases['cylinder']['m']['lambda_m']; U = cases['cylinder']['m']['U']
xmin = min(c['x50'] for c in cases.values()) - 22 * lam          # same window for both panels
nfr = min(len(c['off']) - 1 for c in cases.values())
if maxf: nfr = min(nfr, maxf)
dt_frame = cases['cylinder']['m']['frame_s']
ys = cases['cylinder']['m']['strip_y']

fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
H = {}
for i, body in enumerate(('cylinder', 'plate')):
    c = cases[body]; top = 0.90 - i * 0.43
    ax = fig.add_axes([0.06, top - 0.27, 0.86, 0.27], facecolor=BG)
    pa = fig.add_axes([0.06, top - 0.36, 0.86, 0.08], facecolor=BG)
    for a in (ax, pa):
        a.set_xlim(xmin / lam, 0.6); a.tick_params(colors=DIM, labelsize=12)
        for s in a.spines.values(): s.set_color('#2a3550')
    ax.set_ylim(0, ys / lam); ax.set_yticks([])
    ax.axvspan(0, 0.6, color='#5a6478')                               # the body
    sta = c['x10'] - 5 * lam
    ax.axvline(sta / lam, color=FG, ls=':', lw=1.2)
    ax.text(0.005, 1.03, ('Cylinder' if body == 'cylinder' else 'Flat plate') + ' (stagnation line)',
            transform=ax.transAxes, color=FG, fontsize=17, va='bottom')
    cnt = ax.text(0.995, 1.03, '', transform=ax.transAxes, color=ORANGE, fontsize=16, ha='right', va='bottom')
    pa.plot(c['px'] / lam, c['pr'], color=CYAN, lw=2); pa.plot(c['px'] / lam, c['pT'], color=RED, lw=2)
    pa.set_ylim(-0.05, 1.15); pa.set_yticks([0, 1]); pa.axvline(sta / lam, color=FG, ls=':', lw=1.2)
    if i == 1: pa.set_xlabel(r'distance from the body, $x/\lambda_\infty$', color=DIM, fontsize=14)
    else: pa.set_xticklabels([])
    sc_b = ax.scatter([], [], s=2.2, c=BLUE, lw=0, alpha=0.75)
    sc_o = ax.scatter([], [], s=6, c=ORANGE, lw=0)
    lc = LineCollection([], colors=ORANGE, linewidths=0.8, alpha=0.6); ax.add_collection(lc)
    H[body] = dict(ax=ax, sc_b=sc_b, sc_o=sc_o, lc=lc, cnt=cnt, sta=sta, trail={}, prev={}, n=0)
fig.text(0.06, 0.955, f'Mach 10 {GN[gas]},  Kn = {kn.replace("p", ".")}   ·   every DSMC molecule, followed by its id',
         color=FG, fontsize=20)
fig.text(0.06, 0.025, 'blue: moving downstream    orange: moving upstream (path drawn ahead of the front)    '
         'strip: density (blue) and temperature (red), 0 upstream to 1 behind an ideal normal shock',
         color=DIM, fontsize=13)
clock = fig.text(0.92, 0.955, '', color=DIM, fontsize=15, ha='right')
TRAIL = 8


def frame(f):
    for body, h in H.items():
        c = cases[body]; a, b = c['off'][f], c['off'][f + 1]; z = c['z']
        ids, x, y, vx = z['id'][a:b], z['x'][a:b], z['y'][a:b], z['vx'][a:b]
        keep = x > xmin; ids, x, y, vx = ids[keep], x[keep], y[keep], vx[keep]
        up = vx < 0
        h['sc_b'].set_offsets(np.c_[x[~up], y[~up]] / lam); h['sc_o'].set_offsets(np.c_[x[up], y[up]] / lam)
        # paths of upstream-moving molecules ahead of the front
        tr, segs = h['trail'], []
        ahead = up & (x < c['x10'])
        for i_, x_, y_ in zip(ids[ahead], x[ahead], y[ahead]):
            dq = tr.setdefault(i_, collections.deque(maxlen=TRAIL)); dq.append((x_ / lam, y_ / lam))
            if len(dq) > 1: segs.append(list(dq))
        live = set(ids[ahead].tolist())
        for k_ in list(tr):
            if k_ not in live: del tr[k_]
        h['lc'].set_segments(segs)
        # molecules crossing the counting station upstream
        prev = h['prev']; now = dict(zip(ids.tolist(), x.tolist()))
        for k_, xn in now.items():
            xp = prev.get(k_)
            if xp is not None and xp >= h['sta'] > xn: h['n'] += 1
        h['prev'] = now
        h['cnt'].set_text(f'crossed the dotted line upstream: {h["n"]}')
    clock.set_text(f't = {f * dt_frame * 1e6:6.1f} μs')
    return []


import subprocess, shutil, time
out = HERE / 'films'; out.mkdir(exist_ok=True)
tmp = out / f'_frames_{gas}_{kn}{suffix}'; shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
t0 = time.time()
for f in range(nfr):
    frame(f); fig.savefig(tmp / f'{f:05d}.png', facecolor=BG)
    if f % 100 == 0: print('frame', f, f'{time.time() - t0:.0f} s', flush=True)
subprocess.run([plt.rcParams['animation.ffmpeg_path'], '-loglevel', 'error', '-y', '-framerate', '30', '-i', str(tmp / '%05d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(out / f'molecules_{gas}_Kn{kn}{suffix}.mp4')], check=True)
shutil.rmtree(tmp)
print('film', gas, kn, nfr, 'frames', {b: h['n'] for b, h in H.items()})

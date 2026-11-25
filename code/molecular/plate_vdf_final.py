"""Run on Unity in cases/: final flat-plate VDF from vdf_<gas>_plate_Kn0p01 (200 samples, 2.4-4.4 ms).
One pass builds a 2-D histogram (x in 0.2 mm bins) x (v_x/U); the density on the stagnation line is the
particle count per x bin normalised by its upstream value (constant fnum), which locates the 10/50/90% stations."""
import glob, json
import numpy as np
RH = {'Ar': 3.884, 'N2': 5.714}
for g in ('Ar', 'N2'):
    d = f'vdf_{g}_plate_Kn0p01'; m = json.load(open(d + '/case.json'))
    xe = np.arange(-0.25, 0.0 + 1e-9, 2e-4); ve = np.linspace(-1.5, 1.6, 311)
    H2 = np.zeros((len(xe) - 1, 310)); nf = 0
    for f in sorted(glob.glob(d + '/part.*.dat')):
        q = np.loadtxt(f, skiprows=9, ndmin=2); nf += 1
        H2 += np.histogram2d(q[:, 1], q[:, 3] / m['U'], [xe, ve])[0]
    xc = 0.5 * (xe[1:] + xe[:-1]); cnt = H2.sum(1); rho = cnt / cnt[:25].mean()      # first 5 mm = freestream
    st = {f'{int(f*100)}%': int(np.argmax(rho >= 1 + f * (RH[g] - 1))) for f in (0.1, 0.5, 0.9)}
    st['wall side'] = int(np.argmin(abs(xc + 0.035)))
    out = {'H_' + k: H2[i - 2:i + 2].sum(0) for k, i in st.items()}                   # 0.8 mm slab
    np.savez(f'{g}_plate_Kn0p01_vdf.npz', edges=ve, x_rho=xc, rho=rho, H2=H2, nfiles=nf,
             stations=json.dumps({k: float(xc[i]) for k, i in st.items()}), meta=json.dumps(m), **out)
    print(g, nf, {k: round(float(xc[i]), 4) for k, i in st.items()}, {k: int(v.sum()) for k, v in out.items()}, flush=True)

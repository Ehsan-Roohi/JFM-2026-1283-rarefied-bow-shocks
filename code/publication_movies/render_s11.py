"""Rebuild supplementary Fig. S11 from the archived high-particle SPARTA runs.

Usage: python render_s11.py [--runs VIDEO_RUNS] [--output OUTPUT.pdf]
Each stress sample represents a nonoverlapping 24-microsecond SPARTA window.
The separate velocity files hold means over the same last 40 windows.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / 'data' / 'physics' / 'video_runs'
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reproduced" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon
import numpy as np
from pypdf import PdfReader
from scipy.ndimage import gaussian_filter

LABELS = {"Ar": "Argon", "N2": "Nitrogen"}
ROWS = (("Ar", "0p01"), ("N2", "0p01"), ("Ar", "0p1"), ("N2", "0p1"))
BODIES = ("cylinder", "plate")
SOLID = "#485258"
NO_DATA = "#b5bdb6"
CONTOUR_LEVELS = (0.5, 1.0)


def surface_points(path):
    lines = path.read_text(encoding="ascii").splitlines()
    coordinates = lines[lines.index("Points") + 1:lines.index("Lines")]
    return np.array([[float(v) for v in row.split()[1:3]] for row in coordinates if row.strip()])


def load_case(runs, gas, body, kn, diameter):
    stem = f"hi_{gas}_{body}_Kn{kn}"
    with np.load(runs / "npz_hi" / f"{stem}.npz", allow_pickle=False) as store:
        meta = json.loads(str(store["meta"]))
        fields = store["fields"][-40:]
        time = store["t"]
    window = meta["dt"] * meta["frame_steps"]
    assert (meta["gas"], meta["body"], meta["Kn"]) == (gas, body, float(kn.replace("p", ".")))
    assert fields.shape[:2] == (40, 4) and time.shape == (100,)
    np.testing.assert_allclose(np.diff(time), window, rtol=1e-11)
    np.testing.assert_allclose([time[-40] - window, time[-40], time[-1]],
                               [0.001440, 0.001464, 0.002400], rtol=1e-11)

    density, temperature = fields[:, 0], fields[:, 1]
    ratio = fields[:, 3].astype("float32")
    finite = np.isfinite(ratio)
    valid = (finite & np.isfinite(density) & (density > 0)
             & np.isfinite(temperature) & (temperature > 0) & (abs(ratio) <= 3))
    discarded = int(np.count_nonzero(finite & ~valid))
    ratio[~valid] = np.nan
    # Retain the archived figure definition: equal mean of valid window ratios,
    # then a finite-data normalized spatial Gaussian of width one displayed cell.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        mean = np.nanmean(ratio, axis=0)
    supported = np.isfinite(mean)
    weight = gaussian_filter(supported.astype(float), 1.0)
    numerator = gaussian_filter(np.where(supported, mean, 0).astype(float), 1.0)
    mean = np.divide(numerator, weight, out=np.full_like(numerator, np.nan), where=supported)
    assert np.nanmax(abs(mean)) <= 3

    block = math.ceil(meta["nx"] / 480)
    dx, dy = meta["dx"] * block, meta["dy"] * block
    x0 = meta["domain"][0]
    ny, nx = mean.shape
    extent = (x0 / diameter, (x0 + nx * dx) / diameter, 0, ny * dy / diameter)
    x = (x0 + (np.arange(nx) + 0.5) * dx) / diameter
    y = ((np.arange(ny) + 0.5) * dy) / diameter

    with np.load(runs / "uv_hi" / f"{stem}_uv.npz", allow_pickle=False) as velocities:
        assert json.loads(str(velocities["meta"])) == meta
        u, v = velocities["u"], velocities["v"]
    assert u.shape == v.shape == (meta["ny"], meta["nx"])
    # Keep the originally sampled native velocity nodes and streamplot density.
    # Index cell centres before striding so the plotted velocity is not shifted.
    stride = max(1, u.shape[1] // 300)
    uv_x = (x0 + (np.arange(u.shape[1])[::stride] + 0.5) * meta["dx"]) / diameter
    uv_y = ((np.arange(u.shape[0])[::stride] + 0.5) * meta["dy"]) / diameter
    u, v = u[::stride, ::stride], v[::stride, ::stride]
    no_velocity = ~np.isfinite(u) | ~np.isfinite(v)
    return dict(mean=mean, x=x, y=y, extent=extent,
                u=np.ma.array(u, mask=no_velocity), v=np.ma.array(v, mask=no_velocity),
                uv_x=uv_x, uv_y=uv_y, discarded=discarded,
                finite_samples=int(np.count_nonzero(finite)))


def draw(runs, output):
    boundaries = {body: surface_points(runs / f"{body}.surf") for body in BODIES}
    diameter = float(np.ptp(boundaries["cylinder"][:, 1]))
    np.testing.assert_allclose([diameter, np.ptp(boundaries["plate"][:, 1])], 0.3048)
    panels = {(g, b, k): load_case(runs, g, b, k, diameter) for g, k in ROWS for b in BODIES}
    minimum = min(float(np.nanmin(case["mean"])) for case in panels.values())
    maximum = max(float(np.nanmax(case["mean"])) for case in panels.values())
    limit = math.ceil(2 * max(abs(minimum), abs(maximum))) / 2
    norm = Normalize(vmin=-limit, vmax=limit, clip=False)
    cmap = plt.get_cmap("RdBu_r").with_extremes(bad=NO_DATA)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 10.5, "axes.linewidth": 0.65,
                         "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42})
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 6.0),
                             gridspec_kw=dict(left=0.185, right=0.99, bottom=0.169,
                                              top=0.935, hspace=0.23, wspace=0.10))
    for row, (gas, kn) in enumerate(ROWS):
        span = 1.0 if kn == "0p01" else 2.5
        for col, body in enumerate(BODIES):
            ax = axes[row, col]
            case = panels[gas, body, kn]
            mean, extent = case["mean"], case["extent"]
            ax.set_facecolor(NO_DATA)
            image = ax.imshow(np.ma.masked_invalid(mean), origin="lower", aspect="equal",
                              extent=extent, interpolation="nearest", cmap=cmap, norm=norm,
                              zorder=0)
            assert np.nanmin(mean) >= norm.vmin and np.nanmax(mean) <= norm.vmax
            ax.streamplot(case["uv_x"], case["uv_y"], case["u"], case["v"], density=1.2,
                          color="#45494c", linewidth=0.45, arrowsize=0.5, zorder=2)
            # Never replace missing stress by zero when locating an isoline.
            ax.contour(case["x"], case["y"], np.ma.masked_invalid(mean), levels=CONTOUR_LEVELS,
                       colors="black", linewidths=0.72, linestyles=("dashed", "solid"),
                       corner_mask=False, zorder=3)
            ax.add_patch(Polygon(boundaries[body] / diameter, closed=True, facecolor=SOLID,
                                 edgecolor="#333a3e", linewidth=0.55, zorder=4))
            ax.set(xlim=(-span, extent[1]), ylim=(0, min(span + 0.3, extent[3])))
            ax.set_yticks([0, 1] if row < 2 else [0, 1, 2])
            ax.set_xticks([-1, 0, 1, 2] if row < 2 else [-2, 0, 2])
            ax.tick_params(labelsize=9.2, length=2.5, pad=2.0, labelleft=col == 0,
                           labelbottom=(row == 1 or row == 3))
            if row == 3:
                ax.set_xlabel(r"$x/D$", labelpad=1)
            if col == 0:
                ax.set_ylabel(r"$y/D$", labelpad=2)
            letter = chr(ord("a") + 2 * row + col)
            ax.text(-0.033, 0.971, f"({letter})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.3, clip_on=False)
            if row == 0:
                ax.set_title(("Circular cylinder", "Normal flat plate")[col], pad=6,
                             fontsize=12)
        # Use the equal-aspect axes' actual location, for alignment across rows.
        pos = axes[row, 0].get_position()
        mid = (pos.y0 + pos.y1) / 2
        fig.text(0.013, mid + 0.010, LABELS[gas], fontsize=11.7, va="bottom")
        fig.text(0.013, mid - 0.011, rf"$Kn_\infty={kn.replace('p', '.')}$",
                 fontsize=10.4, va="top")

    cax = fig.add_axes((0.195, 0.061, 0.42, 0.021))
    cb = fig.colorbar(image, cax=cax, orientation="horizontal",
                      ticks=np.linspace(-limit, limit, 5))
    cax.xaxis.set_label_position("top")
    cb.set_label(r"$(P_{xx}-P_{yy})/p$", fontsize=11, labelpad=2.5)
    cax.tick_params(labelsize=9.4, length=2.2, pad=2)
    fig.legend(handles=(Patch(facecolor=SOLID, edgecolor="#333a3e", label="Solid body"),
                        Patch(facecolor=NO_DATA, edgecolor="#737e77", label="No resolved stress")),
               loc="center left", bbox_to_anchor=(0.663, 0.076), frameon=False,
               fontsize=9.7, handlelength=1.15, handleheight=0.74, handletextpad=0.42,
               labelspacing=0.25)
    # Validate the in-memory single-page PDF before replacing the named output.
    pdf = io.BytesIO()
    fig.savefig(pdf, format="pdf", dpi=350,
                metadata={"Title": "Normal-stress anisotropy and mean streamlines at Mach 10",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    assert len(PdfReader(pdf).pages) == 1
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".s11-", suffix=".pdfpart",
                                     delete=False) as pending:
        temporary = Path(pending.name)
        try:
            pending.write(pdf.getvalue())
        except OSError:
            pending.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"diameter={diameter:.4f} m; valid field range=[{minimum:.9f}, {maximum:.9f}]; "
          f"linear colour scale=[{-limit:g}, {limit:g}]")
    for key, vals in panels.items():
        print(key, "omitted among previously finite samples",
              vals["discarded"], "/", vals["finite_samples"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced" / "figures" / "Aniso_contours.pdf")
    args = parser.parse_args()
    draw(args.runs, args.output)

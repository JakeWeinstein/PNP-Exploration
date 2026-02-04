import matplotlib.pyplot as plt
from firedrake import Function, FunctionSpace
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.animation as animation
import numpy as np
import matplotlib.tri as tri
import imageio
import io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

# Save all render outputs under FireDrakeEnvCG/Inverse/Renders
RENDERS_DIR = Path(__file__).resolve().parent / "Renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

def plot_solutions(U_prev, z_vals, mode, num_steps, dt, t,
                   save=True, show=True, output_png=None, return_fig=False,
                   c0_lim=None, c1_lim=None, phi_lim=None):
    """
    Plot concentrations and potential for a given mixed state U_prev.

    If return_fig is True, the caller manages closing the figure.
    Limits can be fixed with tuples (vmin, vmax) per field.
    """
    # Create figure with subplots for each field
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # Fixed spacing to prevent jitter between frames
    fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.12, wspace=0.28)

    # Plot concentration of species 0
    tripcolor(U_prev.sub(0), axes=axes[0], cmap='viridis',
              vmin=None if c0_lim is None else c0_lim[0],
              vmax=None if c0_lim is None else c0_lim[1])
    axes[0].set_title(f'Concentration c₀ (z={z_vals[0]:+d}) at t={t:.3f}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='c₀')

    # Plot concentration of species 1
    tripcolor(U_prev.sub(1), axes=axes[1], cmap='plasma',
              vmin=None if c1_lim is None else c1_lim[0],
              vmax=None if c1_lim is None else c1_lim[1])
    axes[1].set_title(f'Concentration c₁ (z={z_vals[1]:+d}) at t={t:.3f}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_aspect('equal')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='c₁')

    # Plot electric potential
    tripcolor(U_prev.sub(2), axes=axes[2], cmap='coolwarm',
              vmin=None if phi_lim is None else phi_lim[0],
              vmax=None if phi_lim is None else phi_lim[1])
    axes[2].set_title(f'Electric Potential phi at t={t:.3f}')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_aspect('equal')
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='phi')

    bv_status = "with BV" if mode == 1 else "with Robin" if mode == 2 else "Normal"
    fig.suptitle(f'PNP Solution ({bv_status}): {num_steps} time steps, dt={dt}', fontsize=14, y=1.02)
    if save:
        if output_png is None:
            output_png = f'pnp_solution_t{t:.3f}.png'
        # Force outputs into Renders directory while keeping the requested filename
        output_path = RENDERS_DIR / Path(output_png).name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    if show:
        plt.show()
    if return_fig:
        return fig, axes
    else:
        plt.close(fig)
        return None, None
    
def create_animations(snapshots, mode, mesh, W, z_vals, num_steps, dt,
                      fps=10, fmt="mp4", fix_limits=True):
    """
    Use plot_solutions to render each frame, then stitch into an animation.

    snapshots: dict with lists 't', 'c0', 'c1', 'phi' (numpy arrays)
    W: mixed FunctionSpace matching those snapshots
    fix_limits: if True, use global vmin/vmax per field to keep color scales constant
    """
    times = list(snapshots.get("t", []))
    c0_list = list(snapshots.get("c0", []))
    c1_list = list(snapshots.get("c1", []))
    phi_list = list(snapshots.get("phi", []))

    lengths = [len(times), len(c0_list), len(c1_list), len(phi_list)]
    usable_frames = min(lengths) if all(l > 0 for l in lengths) else 0
    if usable_frames == 0:
        print("  Warning: no usable frames (t/c0/c1/phi mismatch or empty). Skipping animations.")
        return

    times = times[:usable_frames]
    c0_list = c0_list[:usable_frames]
    c1_list = c1_list[:usable_frames]
    phi_list = phi_list[:usable_frames]

    # Compute global limits to keep color scales steady (optional)
    if fix_limits:
        def finite_minmax(arrs, pad_frac=0.05):
            flat = np.concatenate([np.ravel(a) for a in arrs])
            flat = flat[np.isfinite(flat)]
            if flat.size == 0:
                return (None, None)
            vmin, vmax = flat.min(), flat.max()
            if vmin == vmax:
                # avoid zero range
                delta = 1e-9 if vmin == 0 else abs(vmin)*1e-3
                vmin -= delta
                vmax += delta
            # pad range to stabilize colorbars
            span = vmax - vmin
            vmin -= pad_frac * span
            vmax += pad_frac * span
            return (vmin, vmax)
        c0_lim = finite_minmax(c0_list)
        c1_lim = finite_minmax(c1_list)
        # Skip the first timestep for phi when deriving global limits to avoid transient spikes
        phi_limit_data = phi_list[1:] if len(phi_list) > 1 else phi_list
        phi_lim = finite_minmax(phi_limit_data)
    else:
        c0_lim = c1_lim = phi_lim = None

    images = []
    print(f"\nRendering {usable_frames} frames via plot_solutions...")

    for i in range(usable_frames):
        frame_func = Function(W)
        frame_func.sub(0).dat.data[:] = c0_list[i]
        frame_func.sub(1).dat.data[:] = c1_list[i]
        frame_func.sub(2).dat.data[:] = phi_list[i]

        fig, _ = plot_solutions(frame_func, z_vals, mode, num_steps, dt, times[i],
                                save=False, show=False, output_png=None, return_fig=True,
                                c0_lim=c0_lim, c1_lim=c1_lim, phi_lim=phi_lim)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)  # fixed size; avoid bbox_inches to keep frames same shape
        buf.seek(0)
        images.append(imageio.v2.imread(buf))
        plt.close(fig)

    fname = RENDERS_DIR / f"pnp_animation{mode}.{fmt}"
    try:
        # imageio mp4 requires imageio-ffmpeg; if missing, this will raise
        imageio.mimsave(fname, images, fps=fps)
        print(f"  Saved animation to {fname}")
        return
    except Exception as e:
        print(f"  Warning: failed to save {fmt} via imageio ({e}); trying matplotlib FFMpegWriter...")
        try:
            Writer = animation.FFMpegWriter
            writer = Writer(fps=fps)
            fig = plt.figure()
            im = plt.imshow(images[0])
            with writer.saving(fig, fname, dpi=150):
                for frame in images:
                    im.set_data(frame)
                    writer.grab_frame()
            plt.close(fig)
            print(f"  Saved animation to {fname} via matplotlib/ffmpeg")
            return
        except Exception as e2:
            print(f"  Warning: matplotlib ffmpeg save failed ({e2}); falling back to GIF.")
            alt = RENDERS_DIR / f"pnp_animation{mode}.gif"
            try:
                imageio.mimsave(alt, images, fps=fps)
                print(f"  Saved animation to {alt}")
            except Exception as ee:
                print(f"  Failed to save animation as gif as well: {ee}")

    print("Animation generation complete!")

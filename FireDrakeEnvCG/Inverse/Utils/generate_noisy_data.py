import numpy as np
from Utils.forsolve import *


def generate_noisy_data(solver_params, noise_std=0.01, seed=None, print_interval=100):
    """
    Run the PNP forward solver and return final-step clean and noisy
    vectors for c0, c1, and phi.

    Parameters
    ----------
    solver_params : list
        [n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params]
    noise_std : float, optional
        Standard deviation of additive Gaussian noise applied to each vector.
    seed : int, optional
        Seed for reproducible noise generation.
    print_interval : int, optional
        How often to print progress inside the forward solver.

    Returns
    -------
    (c0_vec, c1_vec, phi_vec, c0_noisy, c1_noisy, phi_noisy)
        Six 1-D numpy arrays corresponding to the final timestep.
    """
    rng = np.random.default_rng(seed)
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc
    

    # forward_solver already steps through all timesteps and returns final state
    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)
    set_initial_conditions(ctx, solver_params, blob=True)

    U_prev = forsolve(ctx, solver_params)

    n_species = len(z_vals)
    phi_idx = n_species  # phi is stored after all species

    c0_vec = np.array(U_prev.sub(0).dat.data_ro)
    c1_vec = np.array(U_prev.sub(1).dat.data_ro)
    phi_vec = np.array(U_prev.sub(phi_idx).dat.data_ro)

    c0_noisy = c0_vec + rng.normal(0.0, noise_std, size=c0_vec.shape)
    c1_noisy = c1_vec + rng.normal(0.0, noise_std, size=c1_vec.shape)
    phi_noisy = phi_vec + rng.normal(0.0, noise_std, size=phi_vec.shape)

    return c0_vec, c1_vec, phi_vec, c0_noisy, c1_noisy, phi_noisy

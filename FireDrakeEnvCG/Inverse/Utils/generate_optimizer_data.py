import numpy as np
from Utils.forsolve import generate_solver, forsolve_step


def generate_optimizer_data(solver_params, print_interval=100):
    """
    Run the PNP solver, capture per-timestep field snapshots, and return
    clean matrices for c0, c1, and phi (no noise added).

    Parameters
    ----------
    solver_params : list
        [n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params]
    print_interval : int, optional
        How often to print progress messages.

    Returns
    -------
    (c0, c1, phi)
        Three numpy arrays shaped (ndofs, num_steps).
    """
    solver, U, U_prev, meta = generate_solver(solver_params)
    num_steps = meta["num_steps"]

    def allocate_field_matrix(sub_index):
        size = len(U_prev.sub(sub_index).dat.data_ro)
        return np.zeros((size, num_steps))

    c0_mat = allocate_field_matrix(0)
    c1_mat = allocate_field_matrix(1)
    phi_mat = allocate_field_matrix(2)

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f"Stepping through solve, step {step}")
        forsolve_step(solver, U, U_prev)
        c0_mat[:, step] = np.array(U_prev.sub(0).dat.data_ro)
        c1_mat[:, step] = np.array(U_prev.sub(1).dat.data_ro)
        phi_mat[:, step] = np.array(U_prev.sub(2).dat.data_ro)

    return c0_mat, c1_mat, phi_mat

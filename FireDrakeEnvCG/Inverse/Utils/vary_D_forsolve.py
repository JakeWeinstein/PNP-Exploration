from firedrake import *
from Utils.forsolve import forsolve

def vary_D_forsolve(D_vals):
    params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }
    solver_params = [2, 1, 1e-4, 0.1, [1, -1], D_vals, [0.0, 0.0], 0.05, 1, 0, params]
    return forsolve(solver_params)

from Utils.forsolve import *
from Utils.pnp_plotter import plot_solutions, create_animations
import numpy as np

params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }
D_true = [1.0,1.0]
solver_params = [2, 1, 1e-3, 0.1, [1, -1], D_true, [0.0, 0.0], 0.05, 1, 0, params]

ctx = build_context(solver_params)
ctx = build_forms(ctx, solver_params)

set_initial_conditions(ctx, solver_params, blob=True)

U_final = forsolve(ctx, solver_params)

print(np.shape(U_final.sub(0).dat.data_ro))

plot_solutions(U_final, [1, -1], 0, 1000, 1e-4, 0.1)
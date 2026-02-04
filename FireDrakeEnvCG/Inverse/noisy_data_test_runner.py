from Utils.generate_noisy_data import generate_noisy_data
import numpy as np


params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }
D_vals = [1.0,1.0]
solver_params = [2, 1, 1e-4, 0.1, [1, -1], D_vals, [0.0, 0.0], 0.05, 1, 0, params]

c0_mat, c1_mat, phi_mat, c0_noisy, c1_noisy, phi_noisy = generate_noisy_data(solver_params)
print(np.linalg.norm(c0_mat-c1_noisy))
print(np.shape(c0_mat))
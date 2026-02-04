from Utils.generate_noisy_data import generate_noisy_data
from Utils.forsolve import *
import numpy as np
# import scipy.optimize as opt
from Helpers.Infer_D_from_data_helpers import *
import firedrake.adjoint as adj

params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }
D_true = [1.0,1.0]
solver_params_gen = [2, 1, 1e-3, 0.1, [1, -1], D_true, [0.0, 0.0], 0.05, 1, 0, params]

# data generation should not be taped
with adj.stop_annotating():
    c0_vec, c1_vec, phi_vec, c0_noisy, c1_noisy, phi_noisy = generate_noisy_data(solver_params_gen)

theta0 = [1.1,1.1]
solver_params = [2, 1, 1e-3, 0.1, [1, -1], theta0, [0.0, 0.0], 0.05, 1, 0, params]

ctx = build_context(solver_params)
ctx = build_forms(ctx, solver_params)

fun = make_objective_and_grad(ctx,solver_params,c0_vec,c1_vec)

#THese are coefficients on the function space, need to convert to something else, figure this out

# ref_mats = (c0_mat,c1_mat,phi_mat)

#Here we'll run a test_optimization with no noise

#SDHJAIODHJSIAJKDHASIJSHIDJHAISUD USE ADJOINT BUILT IN MINIMIZE

rf = fun()

m_vals = adj.minimize(rf,"BFGS",tol=1e-7, options = {"disp": True})

mlst = [v.dat.data for _,v in enumerate(m_vals)]

dlst = np.exp(mlst)

print(f"D0 = {dlst[0]}, D1 = {dlst[1]}")
# theta_hat = res.x
# print("Estimated params:", theta_hat)
# print("Final objective:", res.fun)

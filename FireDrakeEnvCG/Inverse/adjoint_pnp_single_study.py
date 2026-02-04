"""
Flat, self-contained adjoint study for the PNP problem (no helper functions, no ctx dict).
Run 1 (annotation off): forward solve with D_true to generate target data.
Run 2 (annotation on): forward solve with D_guess, build misfit, compute adjoint/gradient wrt D0, D1.
"""

import numpy as np
import firedrake as fd
import firedrake.adjoint as adj


# ----------------- parameters -----------------
params = {
    "snes_type": "newtonls",
    "snes_max_it": 50,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
n_species = 2
order = 1
dt = 1e-3
t_end = 0.02
z_vals = [1, -1]
D_true = [1.0, 1.0]
# guess we differentiate about
D_guess = [1.1, 1.1]
a_vals = [0.0, 0.0]
phi_applied = 0.05
c0_val = 1.0
phi0_val = 0.0

# ----------------- mesh, spaces, unknowns -----------------
mesh = fd.UnitSquareMesh(32, 32)
V_scalar = fd.FunctionSpace(mesh, "CG", order)
W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])

# state fields (target run)
U_tgt = fd.Function(W)
U_prev_tgt = fd.Function(W)

# state fields (adjoint run)
U = fd.Function(W)
U_prev = fd.Function(W)

# Controls (constants) for adjoint run (initialised to guess)
D = [fd.Constant(float(D_guess[i])) for i in range(n_species)]
z = [fd.Constant(int(z_vals[i])) for i in range(n_species)]

# ----------------- helper to set ICs (simple inline) -----------------
def set_ic(U_prev_field):
    x, y = fd.SpatialCoordinate(mesh)
    c_bulk = fd.Constant(c0_val)
    A = fd.Constant(0.5)
    x0 = fd.Constant(0.5)
    y0 = fd.Constant(0.2)
    sigma = fd.Constant(0.08)
    gaussian = c_bulk + A * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    U_prev_field.sub(0).interpolate(gaussian)
    U_prev_field.sub(1).interpolate(gaussian)
    U_prev_field.sub(n_species).assign(fd.Constant(0.0))


# ----------------- target data generation (no annotation) -----------------
with adj.stop_annotating():
    set_ic(U_prev_tgt)

    ci = fd.split(U_tgt)[:-1]
    phi = fd.split(U_tgt)[-1]
    ci_prev = fd.split(U_prev_tgt)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    F = 96485.3329
    R = 8.314462618
    T = 298.15
    F_over_RT = F / (R * T)
    F_res_tgt = 0
    for i in range(n_species):
        c = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]
        drift = F_over_RT * z[i] * phi
        Jflux = fd.Constant(D_true[i]) * (fd.grad(c) + c * fd.grad(drift))
        F_res_tgt += ((c - c_old) / dt) * v * fd.dx + fd.dot(Jflux, fd.grad(v)) * fd.dx

    eps = fd.Constant(1.0)
    F_res_tgt += eps * fd.dot(fd.grad(phi), fd.grad(w)) * fd.dx
    F_res_tgt -= sum(z[i] * F * ci[i] * w for i in range(n_species)) * fd.dx

    bc_phi = fd.DirichletBC(W.sub(n_species), fd.Constant(phi0_val), 1)
    bc_ci = [fd.DirichletBC(W.sub(i), fd.Constant(c0_val), 3) for i in range(n_species)]
    bcs = bc_ci + [bc_phi]

    J_form_tgt = fd.derivative(F_res_tgt, U_tgt)
    problem_tgt = fd.NonlinearVariationalProblem(F_res_tgt, U_tgt, bcs=bcs, J=J_form_tgt)
    solver_tgt = fd.NonlinearVariationalSolver(problem_tgt, solver_parameters=params)

    num_steps = int(t_end / dt)
    for step in range(num_steps):
        solver_tgt.solve()
        U_prev_tgt.assign(U_tgt)

    c0_target_vec = np.array(U_tgt.sub(0).dat.data_ro)
    c1_target_vec = np.array(U_tgt.sub(1).dat.data_ro)


# ----------------- adjoint run -----------------
adj.get_working_tape().clear_tape()
adj.continue_annotation()

set_ic(U_prev)
# set controls to the guess explicitly
D[0].assign(D_guess[0])
D[1].assign(D_guess[1])

ci = fd.split(U)[:-1]
phi = fd.split(U)[-1]
ci_prev = fd.split(U_prev)[:-1]
v_tests = fd.TestFunctions(W)
v_list = v_tests[:-1]
w = v_tests[-1]

F_res = 0
for i in range(n_species):
    c = ci[i]
    c_old = ci_prev[i]
    v = v_list[i]
    drift = (96485.3329 / (8.314462618 * 298.15)) * z[i] * phi
    Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift))
    F_res += ((c - c_old) / dt) * v * fd.dx + fd.dot(Jflux, fd.grad(v)) * fd.dx

eps = fd.Constant(1.0)
F_res += eps * fd.dot(fd.grad(phi), fd.grad(w)) * fd.dx
F_res -= sum(z[i] * 96485.3329 * ci[i] * w for i in range(n_species)) * fd.dx

bc_phi = fd.DirichletBC(W.sub(n_species), fd.Constant(phi0_val), 1)
bc_ci = [fd.DirichletBC(W.sub(i), fd.Constant(c0_val), 3) for i in range(n_species)]
bcs = bc_ci + [bc_phi]

J_form = fd.derivative(F_res, U)
problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J_form)
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

num_steps = int(t_end / dt)
for step in range(num_steps):
    solver.solve()
    U_prev.assign(U)

# objective using target vectors (projected as Functions)
c0_target_fn = fd.Function(V_scalar); c0_target_fn.dat.data[:] = c0_target_vec
c1_target_fn = fd.Function(V_scalar); c1_target_fn.dat.data[:] = c1_target_vec

J = 0.5 * fd.assemble(((U.sub(0) - c0_target_fn) ** 2 + (U.sub(1) - c1_target_fn) ** 2) * fd.dx)

print("Tape blocks recorded:", len(adj.get_working_tape().get_blocks()))
print("Objective value J:", float(J))

rf = adj.ReducedFunctional(J, [adj.Control(D[0]), adj.Control(D[1])])
grad = rf.derivative()

print("Adjoint/gradient wrt D0:", float(grad[0]))
print("Adjoint/gradient wrt D1:", float(grad[1]))
print("Any None in gradient?", any(g is None for g in grad))

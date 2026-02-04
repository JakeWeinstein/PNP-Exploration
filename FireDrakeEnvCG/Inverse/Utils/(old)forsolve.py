from firedrake import *


def forsolve(solver_params, print_interval=100):
    """
    Single-entry forward solver that builds the problem and advances it
    through all time steps, returning the final mixed state and z_vals.

    solver_params = [
        n_species, order, dt, t_end, z_vals, D_vals,
        a_vals, phi_applied, c0, phi0, params
    ]
    """
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    if n_species < 2:
        raise ValueError("At least two species (c0, c1) plus phi are required.")

    num_steps = int(t_end / dt)
    F = 96485.3329
    R = 8.314462618
    T = 298.15
    F_over_RT = F / (R * T)

    phi_applied = Constant(phi_applied)

    mesh = UnitSquareMesh(32, 32)
    ds = Measure("ds", domain=mesh)

    V_scalar = FunctionSpace(mesh, "CG", order)
    mixed_spaces = [V_scalar for _ in range(n_species)] + [V_scalar]  # last is phi
    W = MixedFunctionSpace(mixed_spaces)

    U = Function(W)
    U_prev = Function(W)

    ci = split(U)[:-1]
    phi = split(U)[-1]
    ci_prev = split(U_prev)[:-1]
    v_tests = TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    F_res = 0
    for i in range(n_species):
        c = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]
        D = Constant(D_vals[i])
        z = Constant(z_vals[i])
        F_res += ((c - c_old) / dt * v) * dx
        drift_potential = F_over_RT * z * phi
        Jflux = D * (grad(c) + c * grad(drift_potential))
        F_res += dot(Jflux, grad(v)) * dx

    eps = Constant(1.0)
    F_res += eps * dot(grad(phi), grad(w)) * dx
    F_res -= sum(Constant(z_vals[i]) * F * ci[i] * w for i in range(n_species)) * dx

    bc_phi = DirichletBC(W.sub(n_species), Constant(phi0), 1)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n_species)]
    bcs = bc_ci + [bc_phi]

    J = derivative(F_res, U)

    x, y = SpatialCoordinate(mesh)
    c_bulk = Constant(c0)
    A = Constant(0.5)
    x0 = Constant(0.5)
    y0 = Constant(0.2)
    sigma = Constant(0.08)
    gaussian = c_bulk + A * exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))

    U_prev.sub(0).interpolate(gaussian)
    U_prev.sub(1).interpolate(gaussian)
    U_prev.sub(n_species).assign(Constant(0.0))

    U.assign(U_prev)

    problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print(f"Stepping through solve, step {step}")
        try:
            solver.solve()
        except Exception as e:
            print(f"Failed to converge: {e}")
            break
        U_prev.assign(U)

    return U_prev, z_vals



from firedrake import *
from firedrake.adjoint import *
from Utils.pnp_plotter import *

continue_annotation()

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


order = 1
n = 2

F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F/(R*T)


nx=32
ny=32
mesh = UnitSquareMesh(nx, ny)
V_scalar = FunctionSpace(mesh, "CG", order)
W = MixedFunctionSpace([V_scalar for _ in range(n)] + [V_scalar])

U = Function(W)
U_prev = Function(W)

D_vals = [0.01,0.01]
z_vals = [1,-1]

phi0 = 0
c0 = 0

t_end = 0.1
dt = 1e-3
print_interval = 10
D = [Function(V_scalar).assign(D_vals[i]) for i in range(n)]   # <-- infer these
z = [Constant(int(z_vals[i])) for i in range(n)]

ci = split(U)[:-1]
phi = split(U)[-1]
ci_prev = split(U_prev)[:-1]

v_tests = TestFunctions(W)
v_list = v_tests[:-1]
w = v_tests[-1]

F_res = 0
for i in range(n):
    c = ci[i]; c_old = ci_prev[i]; v = v_list[i]
    drift = F_over_RT * z[i] * phi
    Jflux = D[i]*(grad(c) + c*grad(drift))
    F_res += ((c - c_old)/dt)*v*dx + dot(Jflux, grad(v))*dx

eps = Constant(1.0)
F_res += eps*dot(grad(phi), grad(w))*dx
F_res -= sum(z[i]*F*ci[i]*w for i in range(n))*dx

bc_phi = DirichletBC(W.sub(n), Constant(phi0), 1)
bc_ci  = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
bcs = bc_ci + [bc_phi]

J_form = derivative(F_res, U)

x, y = SpatialCoordinate(mesh)
c_bulk = Constant(c0)

A = Constant(0.5); x0 = Constant(0.5); y0 = Constant(0.2); sigma = Constant(0.08)
gaussian = c_bulk + A*exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))
U_prev.sub(0).interpolate(gaussian)
U_prev.sub(1).interpolate(gaussian)

U_prev.sub(n).assign(Constant(0.0))

num_steps = int(t_end/dt)

J = derivative(F_res, U)

problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
   
solver = NonlinearVariationalSolver(problem, solver_parameters=params)

for step in range(num_steps):
        if step % print_interval == 0:
            print("step", step)
        solver.solve()
        # carry the solution forward in time
        U_prev.assign(U)
        # obj += assemble(U.sub(0)*U.sub(0)*dx)

obj = assemble(U_prev.sub(0)*U_prev.sub(0)*dx)
alpha = 1e-5
obj += alpha*assemble(inner(grad(D[0]),grad(D[0]))*dx)
obj += alpha*assemble(inner(grad(D[1]),grad(D[1]))*dx)

# plot_solutions(U_prev, z_vals,0, num_steps, dt, t_end)

objhat = ReducedFunctional(obj, [Control(D[0]),Control(D[1])])

dJ = objhat.derivative()


print(dJ[0].dat.data)
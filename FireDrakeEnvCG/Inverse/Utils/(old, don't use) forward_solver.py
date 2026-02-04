from firedrake import *
import argparse
import json

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# print("available files: ", os.listdir())


from FireDrakeEnvCG.Inverse.Utils.pnp_plotter import plot_solutions, create_animations

parser = argparse.ArgumentParser(
    description='PNP solver with optional Butler-Volmer boundary conditions',
    epilog='Examples:\n  python pnp_solver.py 0  # Run without BV (default case)\n  python pnp_solver.py 1  # Run with Butler-Volmer BCs',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('bc_mode', type=int, choices=[0, 1, 2],
                    help='Type of boundary conditions: 2=robin, 1=butler-volmer, 0=dirichlet')
# args = parser.parse_args()
args = 0
BC_MODE_DEFAULT = 0
BC_MODE_BV = 1
BC_MODE_ROBIN = 2
# mode = args.bc_mode
mode = 0

use_butler_volmer = (mode == BC_MODE_BV)
use_robin = (mode == BC_MODE_ROBIN)
print(f"\n{'='*60}")
print(f"Butler-Volmer boundary conditions: {'ENABLED' if use_butler_volmer else 'DISABLED'}")
print(f"{'='*60}\n")

n_species = n = 2
order = 1
dt = 1e-4
t_end = 0.1
num_steps = int(t_end / dt)
# Output frequency (smaller = more snapshots for smoother animation)
output_interval = 10

F = 96485.3329
R = 8.314462618
T = 298.15
F_over_RT = F/(R*T)

D_vals = [0.1, 0.1]
z_vals = [1, -1]
a_vals = [0.0, 0.0]

phi_applied = Constant(0.05)

# Parameters for unique BCs
if use_butler_volmer:
    j0 = Constant(0.01)
    alpha = Constant(0.5)
    n_electrons = Constant(1.0)
    phi_eq = Constant(0.0)
elif use_robin:
    # Robin Parameters: Flux J = kappa * (c - c_inf)
    # kappa: Mass transfer coefficient (m/s)
    # c_inf: Ambient/Bulk concentration
    kappa = Constant(1.0)
    c_inf = Constant(1.0)

mesh = UnitSquareMesh(32, 32)
ds = Measure("ds", domain=mesh)

electrode_marker = 1

V_scalar = FunctionSpace(mesh, "CG", order)
mixed_spaces = [V_scalar for _ in range(n)] + [V_scalar]  # last is phi
W = MixedFunctionSpace(mixed_spaces)

U = Function(W)
U_prev = Function(W)

splitCur = split(U)
splitPrev = split(U_prev)

v_tests = TestFunctions(W)

ci = split(U)[:-1]
phi = split(U)[-1]
ci_prev = split(U_prev)[:-1]
phi_prev = split(U_prev)[-1]

v_list = v_tests[:-1]
w = v_tests[-1]

# We are not using the mu term for now (solver was failing to converge)
# mu = kT * ln(1 - sum_j a_j * c_j)
kB = Constant(1.380649e-23)
kBT = Constant(R*T)
sum_a_c = sum( Constant(a_vals[i]) * ci[i] for i in range(n) )
# mu_steric = kBT * ln(1 - sum_a_c)   # if a_vals are zero, this term vanishes
# grad(mu_steric) computed by UFL via grad(mu_steric)

F_res = 0
for i in range(n):
    c = ci[i]
    c_old = ci_prev[i]
    v = v_list[i]
    D = Constant(D_vals[i])
    z = Constant(z_vals[i])
    # Backward Euler time derivative
    F_res += ( (c - c_old)/dt * v )*dx

    # diffusion + drift (no steric)
    drift_potential = F_over_RT * z * phi # + mu_steric
    Jflux = D*(grad(c) + c * grad(drift_potential))
    F_res += dot(Jflux, grad(v))*dx

# Unique BC boundary flux
if use_butler_volmer:
    eta = phi - phi_applied
    j_BV = j0*( exp(-alpha*n_electrons*F_over_RT*eta) - exp((1-alpha)*n_electrons*F_over_RT*eta) )

    for i in range(n):
        v = v_list[i]
        if i == 0:
            # Oxidized species: consumed at electrode (negative flux)
            F_res += ( (1.0/(n_electrons*F)) * j_BV * v )*ds(electrode_marker)
        else:
            # Reduced species: produced at electrode (positive flux)
            F_res -= ( (1.0/(n_electrons*F)) * j_BV * v )*ds(electrode_marker)
            
elif use_robin:
    # Robin Flux: J.n = kappa * (c - c_inf)
    # Weak form boundary term: + (J.n) * v * ds
    for i in range(n):
        v = v_list[i]
        c = ci[i]
        F_res += kappa * (c - c_inf) * v * ds(electrode_marker)

eps = Constant(1.0)
phi_test = w
F_res += eps*dot(grad(phi), grad(phi_test))*dx
F_res -= sum( Constant(z_vals[i])*F * ci[i]*phi_test for i in range(n) )*dx

c0 = 1.0

if use_butler_volmer or use_robin:
    # With BV: electrode at phi_applied, ground at opposite side
    bc_phi_electrode = DirichletBC(W.sub(n), phi_applied, 1)
    bc_phi_ground = DirichletBC(W.sub(n), Constant(0.0), 3)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]
else:
    # Dirichlet bcs
    phi0 = 0
    bc_phi = DirichletBC(W.sub(n), Constant(phi0), 1)
    bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi]
    # phi_left  = Constant(0.05)
    # phi_right = Constant(0.0)

    # bc_phi_1 = DirichletBC(W.sub(n), phi_left,  1)
    # bc_phi_3 = DirichletBC(W.sub(n), phi_right, 3)

    # bc_ci = [DirichletBC(W.sub(i), Constant(c0), 3) for i in range(n)]
    # bcs = bc_ci + [bc_phi_1, bc_phi_3]


J = derivative(F_res, U)

# for i in range(n):
#     U_prev.sub(i).assign(Constant(c0))
# --- Gaussian blob IC for species 0 ---
x, y = SpatialCoordinate(mesh)

c_bulk = Constant(c0)          # background concentration
A = Constant(0.5)              # amplitude of the bump (try 0.1–1.0)
x0 = Constant(0.5)             # center x
y0 = Constant(0.2)             # center y (near bottom if that’s your electrode)
sigma = Constant(0.08)         # width (smaller = tighter)

gaussian = c_bulk + A*exp(-((x - x0)**2 + (y - y0)**2) / (2.0*sigma**2))

U_prev.sub(0).interpolate(gaussian)     # c0 = blob
U_prev.sub(1).interpolate(gaussian)       # c1 = uniform

if use_butler_volmer or use_robin:
    x, y = SpatialCoordinate(mesh)
    phi_init = phi_applied * (1 - y)
    U_prev.sub(n).interpolate(phi_init)
    U.assign(U_prev)
else:
    U_prev.sub(n).assign(Constant(0.0))


# if use_butler_volmer or use_robin:
params = {
            'snes_type': 'newtonls',
            'snes_max_it': 100,
            'snes_atol': 1e-8,
            'snes_rtol': 1e-8,
            'snes_linesearch_type': 'bt',
            'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'
        }
U_initial_state = Function(W).assign(U_prev)

U.assign(U_initial_state)
U_prev.assign(U_initial_state)
problem = NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)

print("solver params: ", json.dumps(params, indent=2))
# print(f"Testing Configuration {i}: {sp['snes_type']} + {sp['snes_linesearch_type']}")
solver = NonlinearVariationalSolver(problem, solver_parameters=params)

t = 0.0
print(f"Starting time evolution: {num_steps} steps from t=0 to t={t_end}")
print(f"Time step dt = {dt}")

# store snapshots for animation (includes initial state at t=0)
snapshots = {'t': [], 'c0': [], 'c1': [], 'phi': []}

if use_butler_volmer:
    suffix = "_bv"
elif use_robin:
    suffix = "_robin"
else:
    suffix = "_default"

phi_file = VTKFile(f"phi{suffix}.pvd")
c_files = [VTKFile(f"c{i}{suffix}.pvd") for i in range(n)]

print(f"\n{'*'*20} INITIAL CONDITIONS VERIFICATION {'*'*20}")

for i in range(n):
    c_data = U_prev.sub(i).dat.data_ro
    print(f"Species c{i}: Min={c_data.min():.4f}, Max={c_data.max():.4f}, Mean={c_data.mean():.4f}")

phi_data = U_prev.sub(n).dat.data_ro
print(f"Potential phi: Min={phi_data.min():.4f}, Max={phi_data.max():.4f}, Mean={phi_data.mean():.4f}")
print(f"{'*'*60}\n")

# record and plot initial conditions
snapshots['t'].append(t)
snapshots['c0'].append(U_prev.sub(0).dat.data_ro.copy())
snapshots['c1'].append(U_prev.sub(1).dat.data_ro.copy())
snapshots['phi'].append(U_prev.sub(n).dat.data_ro.copy())

plot_solutions(U_prev, z_vals, mode, num_steps, dt, t)

for step in range(num_steps):
    try:
        solver.solve()
    except Exception as e:
        print(f"Failed to converge: {e}")
        break

    t += dt

    if step % output_interval == 0 or step == num_steps - 1:
        print(f"  Step {step+1}/{num_steps}, t = {t:.4f}")

        phi_file.write(U.sub(n), time=t)
        for i in range(n):
            c_files[i].write(U.sub(i), time=t)

        snapshots['t'].append(t)
        snapshots['c0'].append(U.sub(0).dat.data_ro.copy())
        snapshots['c1'].append(U.sub(1).dat.data_ro.copy())
        snapshots['phi'].append(U.sub(n).dat.data_ro.copy())

    U_prev.assign(U)

print(f"Time evolution complete! Final time: t = {t:.4f}")

plot_solutions(U_prev, z_vals, mode, num_steps, dt, t)

create_animations(snapshots, mode, mesh, W, z_vals, num_steps, dt, fix_limits=True)

    
 
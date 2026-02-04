import numpy as np
import firedrake as fd
import firedrake.adjoint as adj
from Utils.forsolve import *

def vec_to_function(ctx, vec, *, space_key="V_scalar"):
    """
    Build a Firedrake Function on ctx[space_key] with DOF coefficients from vec.
    vec must have length == ctx[space_key].dim().
    """
    V = ctx[space_key]
    f = fd.Function(V)
    v = np.asarray(vec, dtype=float).ravel()

    if v.size != f.dat.data.size:
        raise ValueError(
            f"Target vector length {v.size} != DOFs {f.dat.data.size} for {space_key}"
        )

    # assign coefficients (CG1: these are nodal values)
    f.dat.data[:] = v
    return f

def make_objective_and_grad(ctx, solver_params, c0_target, c1_target, blob_ic=True):
    """
    Returns a function f(theta) -> (J, grad) suitable for scipy.optimize.minimize with jac=True.
    Follows the Firedrake adjoint tutorial structure: reset annotation, assign controls,
    solve forward, build ReducedFunctional, then take derivative.
    """

    # Controls and targets live on the provided context
    D_consts = ctx["D_consts"]
    logD_funcs = ctx["logD_funcs"]
    c0_target_f = vec_to_function(ctx, c0_target)
    c1_target_f = vec_to_function(ctx, c1_target)

    def fun():

        # Start from a clean tape (Firedrake adjoint tutorial pattern)
        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()

        # reset ICs so each evaluation is consistent
        set_initial_conditions(ctx, solver_params, blob=blob_ic)

        # forward solve
        U_final = forsolve(ctx, solver_params)

        # objective
        Jobj = 0.5 * fd.assemble(
            (fd.inner(U_final.sub(0) - c0_target_f,U_final.sub(0) - c0_target_f) + fd.inner(U_final.sub(1) - c1_target_f,U_final.sub(0) - c0_target_f))
            * fd.dx
        )

        # alpha = 1e-6
        # Jobj += alpha*fd.assemble(fd.inner(fd.grad(D_consts[0]),fd.grad(D_consts[0]))*fd.dx)
        # Jobj += alpha*fd.assemble(fd.inner(fd.grad(D_consts[1]),fd.grad(D_consts[1]))*fd.dx)


        print("Jobj type:", type(Jobj))
        print("Tape blocks:", len(adj.get_working_tape().get_blocks()))

        def eval_cb(j, m):
            print (f"j = {j}, m = {m[0].dat.data}, {m[1].dat.data}, D = {np.exp(m[0].dat.data)},{np.exp(m[1].dat.data)}" )
        # def eval_cb(j, m):
        #      print (f"j = {j}, m = {m}" )

        # def derivative_cb(j, dj, m):
        #     print (f"j = {j}, dj = {dj}, m = {m}")
        # Reduced functional and gradient wrt controls


        rf = adj.ReducedFunctional(
            Jobj, 
            [adj.Control(logD_funcs[0]), adj.Control(logD_funcs[1])],
            eval_cb_post = eval_cb
            # derivative_cb_post = derivative_cb
        )
        
        return rf

    return fun

def cb(xk):
    print("Current parameters:", xk)

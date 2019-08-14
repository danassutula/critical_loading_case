
from dolfin import *

import os
import sys
import math
import logging
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout)

from .critload import CriticalLoadSolver


def remove_existing_files(dirname, file_ext):

    if not isinstance(file_ext, (list, tuple)):
        file_ext = (file_ext,)

    filenames = [filename for filename in os.listdir(dirname) if
                 any(filename.endswith(ext) for ext in file_ext)]

    for filename in filenames:
        os.remove(os.path.join(dirname, filename))


if __name__ == "__main__":

    ### Setup

    PLOT_RESULTS = True
    RESULTS_DIR = "results"

    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(os.path.join(os.curdir, RESULTS_DIR))

    remove_existing_files(RESULTS_DIR, (".pvd", ".vtu"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    plt.interactive(True)
    plt.close('all')


    ### Mesh

    L = 2.0
    H = 1.0
    W = 1.0

    ny = 30
    nx = ny * round(L/H)
    nz = ny * round(W/H)

    mesh = RectangleMesh(Point(0,0), Point(L, H), nx, ny, 'left/right')
    # mesh = BoxMesh(Point(0, 0, 0), Point(L, H, W), nx, ny, nz)


    ### Functions

    fe_u = VectorElement('CG', cell=mesh.cell_name(), degree=1)
    fe_m = VectorElement('CG', cell=mesh.cell_name(), degree=1)
    fe_s = FiniteElement('CG', cell=mesh.cell_name(), degree=1) # For plotting scalars

    V_u = FunctionSpace(mesh, fe_u)
    V_m = FunctionSpace(mesh, fe_m)
    V_p = FunctionSpace(mesh, fe_u)
    V_s = FunctionSpace(mesh, fe_s)

    u = Function(V_u)
    m = Function(V_m)
    p = Function(V_p)

    dm = Function(V_m)
    dp = Function(V_p)

    u.rename('u','unnamed label')
    p.rename('p','unnamed label')
    m.rename('m','unnamed label')

    outfile_u = File(os.path.join(RESULTS_DIR, 'u.pvd'))
    outfile_p = File(os.path.join(RESULTS_DIR, 'p.pvd'))
    outfile_m = File(os.path.join(RESULTS_DIR, 'm.pvd'))

    write_index = 0
    write_cycle = 5

    # u_rel = Function(V_u)
    # u_rel.rename('u_rel','unnamed label')

    dimdofs_V_u = [V_u_i.dofmap().dofs() for V_u_i in V_u.split()]
    domain_size = assemble(1*dx(mesh))

    def write_solution_snapshot():
        global write_index

        if write_index % write_cycle == 0:

            # u_rel.vector()[:] = u.vector()
            # for dofs_i, u_i in zip(dimdofs_V_u, u):
            #     u_rel.vector()[dofs_i] -= assemble(u_i*dx) / domain_size

            # outfile_u << u_rel

            outfile_u << u
            outfile_m << m
            outfile_p << p

        write_index += 1


    ### BC's

    # def fixed_boundary(x, on_boundary):
    #     return x[0] < DOLFIN_EPS and x[1] < DOLFIN_EPS

    def fixed_boundary(x, on_boundary):
        return x[0] < DOLFIN_EPS \
            and (x[1] < DOLFIN_EPS or x[1] > 1-DOLFIN_EPS)

    # def fixed_boundary(x, on_boundary):
    #     return 0.5 - DOLFIN_EPS < x[0] < 0.5 + DOLFIN_EPS \
    #         and (x[1] < DOLFIN_EPS or x[1] > 1-DOLFIN_EPS)

    bcs = [DirichletBC(V_u, Constant((0.0,)*len(u)), fixed_boundary, method="pointwise"),]

    # m0 = Constant((1.0,)*len(m))
    m0 = Expression(('0.0','x[0]+1e-6 > 1.0 ? -1.0 : 0.0'), degree=1)
    m.assign(interpolate(m0, V_m))
    assert norm(m) > 0

    # p0 = Constant((1e0, 1e0))
    p0 = Expression(('x[0]+1e-6 < 1.0 ? 1.0e6 : 0.0','x[0]+1e-6 < 1.0 ? 1.0e6 : 0.0'), degree=1)
    p.assign(interpolate(p0, V_p))
    # assert norm(p) > 0

    # # u0 = Constant((1.0,)*len(u))
    # u0 = Constant((1.0, 1.0))
    # u.assign(interpolate(u0, V_u))
    # assert norm(u) > 0

    ### Physics

    e = sym(grad(u))
    U = inner(e, e)*dx


    ### Objective to maximize

    # NOTE: Generally, `assemble(0.5*dot(u, m)*dx) != assemble(U)` due to the
    # weak imposition of Dirichlet boundary conditions. It is better to define
    # `J = U` as this is less sensitivie to the Dirichlet BC's.

    J = U
    # J = 0.5*dot(u, m)*dx

    # subdomain = CompiledSubDomain('x[0] > 0.8-1e-11')
    # # subdomain = CompiledSubDomain('x[0] > 0.5-1e-11 && x[1] > 0.5-1e-11')
    # # subdomain = CompiledSubDomain('pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2) < 0.04')
    #
    # subdomain_markers = MeshFunction('size_t', mesh, mesh.geometry().dim())
    # subdomain.mark(subdomain_markers, 1)
    #
    # J = inner(e, e) * dx(subdomain_data=subdomain_markers, subdomain_id=1)


    ### Load Solver

    maxsteps = 500
    tolerance = 1e-3

    # stepsizes_m = [0.30,]
    # stepsizes_m = [0.20,]
    # stepsizes_m = [0.001,]

    stepsizes_m = [0.2, 0.1, 0.05]
    relative_stepsize_p = 0.025

    cutoffs_m = np.linspace(0.0, 0.95, 5).tolist()
    norms_p = [1e-6, 1e-3, 1e-1, 1, 1e1, 1e3, 1e6]

    n = max(len(stepsizes_m), len(cutoffs_m), len(norms_p))

    stepsizes_m = stepsizes_m + stepsizes_m[-1:] * (n - len(stepsizes_m))
    cutoffs_m = cutoffs_m + cutoffs_m[-1:] * (n - len(cutoffs_m))
    norms_p = norms_p + norms_p[-1:] * (n - len(norms_p))

    solver = CriticalLoadSolver(J, U, u, m, p, bcs=[],
        dx_m=dx, dx_p=dx, kappa_m=0.01, kappa_p=0.05)

    J_vals = []

    for stepsize_m, cutoff_m, norm_p in zip(stepsizes_m, cutoffs_m, norms_p):

        is_converged, J_vals_ = solver.optimize(maxsteps,
            stepsize_m, relative_stepsize_p, cutoff_m, norm_p, tolerance,
            num_edge_items=4, external_callable=write_solution_snapshot)

        J_vals.extend(J_vals_)


    dm.vector()[:] = solver.compute_dm_hat()
    dp.vector()[:] = solver.compute_dp_hat()

    p_vec = p.vector()
    dp_vec = dp.vector()

    dp_vec[np.flatnonzero((p_vec==0.0)*(dp_vec<0.0))] = 0.0


    ### Plot

    def plot_objective_evolution():
        figname = "Objective Function vs. Iterations"
        plt.figure(figname)
        # plt.clf()
        plt.plot(J_vals, '-k', markerfacecolor='w')
        plt.xlabel('Iteration number')
        plt.ylabel('Objective, $J$')
        plt.annotate(f'J_max={J_vals[-1]:.3e}', xy=(1, 0), xytext=(-12, 12), va='bottom',
                     ha='right', xycoords='axes fraction', textcoords='offset points',
                     bbox=dict(boxstyle="round", fc="w"))
        plt.title(figname)
        plt.show()

    def plot_other_results():

        figname = 'Displacement Field (u)'
        plt.figure(figname)
        plot(u)
        plt.title(figname)
        plt.axis('equal')

        figname = 'Body Force (m)'
        plt.figure(figname)
        plot(m)
        plt.title(figname)
        plt.axis('equal')

        figname = 'Penalty (p)'
        plt.figure(figname)
        plot(p)
        plt.title(figname)
        plt.axis('equal')

        figname = 'Magnitude Body Force (|m|)'
        plt.figure(figname)
        plot(project(sqrt(m**2), V_s))
        plt.title(figname)
        plt.axis('equal')

        figname = 'Magnitude Penalty (|p|)'
        plt.figure(figname)
        plot(project(sqrt(p**2), V_s))
        plt.title(figname)
        plt.axis('equal')

        figname = 'Body Force Increment (dm_hat)'
        plt.figure(figname)
        dm_hat = Function(V_m)
        dm_hat.vector()[:] = solver.compute_dm_hat()
        plot(dm_hat)
        plt.title(figname)
        plt.axis('equal')

        figname = 'Penalty Increment (dp_hat)'
        plt.figure(figname)
        dp_hat = Function(V_p)
        dp_hat.vector()[:] = solver.compute_dp_hat()
        plot(dp_hat)
        plt.title(figname)
        plt.axis('equal')


    if PLOT_RESULTS:

        if mesh.geometric_dimension() == 2:
            plot_other_results()

        plot_objective_evolution()

    plt.show()

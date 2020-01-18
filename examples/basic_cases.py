
from dolfin import *

import os
import sys
import math
import dolfin
import logging
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout)

from critload import CriticalLoadSolver


def remove_existing_files(dirname, file_ext):

    if not isinstance(file_ext, (list, tuple)):
        file_ext = (file_ext,)
        
    filenames = [filename for filename in os.listdir(dirname) if
                 any(filename.endswith(ext) for ext in file_ext)]

    for filename in filenames:
        os.remove(os.path.join(dirname, filename))


if __name__ == "__main__":

    ### Results output

    PLOT_RESULTS = True

    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", SCRIPT_NAME)

    if not os.path.isdir(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    remove_existing_files(RESULTS_DIR, (".pvd", ".vtu"))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    plt.interactive(True)
    plt.close('all')


    ### Problem setup

    # PROBLEM_NAME = "clamped_one_side"
    PROBLEM_NAME = "clamped_two_sides"
    # PROBLEM_NAME = "clamped_four_sides"

    problem_dimension = 2
    # problem_dimension = 3

    # large_deformations = True
    large_deformations = False

    # Force integration domain
    # dx_m = dx # Computed in volume
    dx_m = ds # Computed on surface

    force_magnitude = 1.0

    L = 10.0
    W = 10.0
    H = 1.0

    nz = 8
    nx = nz * round(L/H)
    ny = nz * round(W/H)

    if problem_dimension == 2:
        mesh = RectangleMesh(Point(0,0), Point(L, H), nx, nz, 'left/right')
    elif problem_dimension == 3:
        mesh = BoxMesh(Point(0, 0, 0), Point(L, W, H), nx, ny, nz)
    else:
        raise ValueError('Parameter `problem_dimension` can either equal 2 or 3')


    ### Functions

    fe_u = VectorElement('CG', cell=mesh.cell_name(), degree=1)
    fe_m = VectorElement('CG', cell=mesh.cell_name(), degree=1)
    fe_s = FiniteElement('CG', cell=mesh.cell_name(), degree=1) # For plotting scalars

    V_u = FunctionSpace(mesh, fe_u)
    V_m = FunctionSpace(mesh, fe_m)
    V_s = FunctionSpace(mesh, fe_s)

    u  = Function(V_u)
    m  = Function(V_m)

    u.rename('u','unnamed label')
    m.rename('m','unnamed label')

    outfile_u = File(os.path.join(RESULTS_DIR, 'u.pvd'))
    outfile_m = File(os.path.join(RESULTS_DIR, 'm.pvd'))

    solution_writing_cycle = 20

    def write_solution_snapshot(iteration_step):
        if not iteration_step % solution_writing_cycle:
            outfile_u << u
            outfile_m << m


    ### BC's

    if PROBLEM_NAME == "clamped_one_side":

        def fixed_boundary(x, on_boundary): return x[0]<1e-6
        bcs = [DirichletBC(V_u, Constant((0.0,)*len(u)), fixed_boundary),]

    elif PROBLEM_NAME == "clamped_two_sides":

        fixed_boundary = CompiledSubDomain(f'x[0]<1e-6 || x[0]>{L}-1e-6')
        bcs = [DirichletBC(V_u, Constant((0.0,)*len(u)), fixed_boundary),]

    elif PROBLEM_NAME == "clamped_four_sides" and problem_dimension == 3:
        fixed_boundary = CompiledSubDomain(f'x[0] < 1e-6 || x[0] > {L} - 1e-6 || '
                                           f'x[1] < 1e-6 || x[1] > {W} - 1e-6')
        bcs = [DirichletBC(V_u, Constant((0.0,)*len(u)), fixed_boundary),]
    else:
        raise ValueError('Parameter `PROBLEM_NAME`')


    ### Material model

    E = Constant(1000.0) # Young's modulus
    nu = Constant(0.3) # Poisson's ratio

    # Lame material parameters
    lm = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    mu = E/(2.0 + 2.0*nu)

    if large_deformations:

        I = dolfin.Identity(len(u))
        F = dolfin.variable(I + dolfin.grad(u)) # Deformation gradient
        C = F.T*F # Right Cauchy-Green deformation tensor

        I1 = dolfin.tr(C)
        det_F = dolfin.det(F)

        # Strain energy density of a Neo-Hookean material
        psi = (mu/2.0)*(I1 - len(u) - 2.0*dolfin.ln(det_F)) + (lm/2.0)*dolfin.ln(det_F)**2

        # First Piola-Kirchhoff stress tensor
        pk1 = dolfin.diff(psi, F)

    else:

        I = dolfin.Identity(len(u))
        e = sym(grad(u))

        # Cauchy stress tensor
        s = 2*mu*e + lm*dolfin.tr(e)*I

        # Strain energy density of a linear-elastic material
        psi = 0.5*inner(s, e)

    # Strain energy
    U = psi*dx


    ### Objective to maximize

    J = U
    # J = dot(u, m)*dx


    ### Load Solver

    stepsize = 0.1
    cutoffs = tuple(0.5e-1*i for i in range(20))

    tolerance = 0.5e-2
    maximum_steps = 1000

    m0 = [0.0,] * len(u)
    dims_m = [i for i in range(len(u))]
    for dim_i in dims_m: m0[dim_i] = -1.0e-1
    m.assign(interpolate(Constant(m0), V_m))

    force_magnitude_ = assemble(sqrt(m**2)*dx_m)
    assert force_magnitude_ > 0.0, "Total force is zero"
    m.vector()[:] *= force_magnitude / force_magnitude_


    # Normalize force

    solver = CriticalLoadSolver(J, U, u, m, bcs, dx_m, dims_m,
        use_nonlinear_solver=True if large_deformations else False)

    # solver._equilibrium_solver.parameters['nonlinear_solver'] = "snes"
    # solver._equilibrium_solver.parameters['nonlinear_solver'] = "newton"

    is_converged, J_vals = solver.optimize(stepsize, cutoffs,
        tolerance, maximum_steps, write_solution_snapshot)


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

        figname = 'Magnitude Body Force (|m|)'
        plt.figure(figname)
        plot(project(sqrt(m**2), V_s))
        plt.title(figname)
        plt.axis('equal')


    if PLOT_RESULTS:

        if mesh.geometric_dimension() == 2:
            plot_other_results()

        plot_objective_evolution()

    plt.show()

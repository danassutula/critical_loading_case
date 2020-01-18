
import math
import dolfin
import logging
import numpy as np

from dolfin import DirichletBC
from dolfin import Constant
from dolfin import Function
from dolfin import solve
from dolfin import action
from dolfin import adjoint
from dolfin import assemble
from dolfin import assemble_system
from dolfin import derivative
from dolfin import inner
from dolfin import grad
from dolfin import dot
from dolfin import dx

from ufl.form import Form as ufl_form_t
from ufl.core.expr import Expr as ufl_expr_t
from ufl.constantvalue import zero as ufl_zero

from .config import logger


SMOOTHING_SOLVER_KAPPA = 0.1


class LinearEquilibriumSolver:
    def __init__(self, a, L, u, bcs):
        '''
        Solve a problem of this form:
        a u = L

        '''

        if bcs is None:
            bcs = ()
        elif not isinstance(bcs, (list, tuple)):
            bcs = (bcs,)
        if not all(isinstance(bc, DirichletBC) for bc in bcs):
            raise TypeError('Parameter `bcs` must contain '
                            'homogenized `DirichletBC`(\'s)')

        V = u.function_space()
        v0 = dolfin.TestFunction(V)

        L_dummy = dot(v0, Constant((0.0,)*len(u)))*dx
        K, self._rhs_bcs = assemble_system(a, L_dummy, bcs)
        self._solver = dolfin.LUSolver(K, "mumps")

        dof_bcs = []
        for bc in bcs:
            dof_bcs.extend(bc.get_boundary_values().keys())
        self._dof_bcs = tuple(sorted(dof_bcs))

        self._x = u.vector()
        self._L = L

    def solve(self):
        rhs = assemble(self._L); rhs[self._dof_bcs] = 0.0
        return self._solver.solve(self._x, rhs+self._rhs_bcs)

    @property
    def solver(self):
        return self._solver

    @property
    def parameters(self):
        return self._solver.parameters


class LinearAdjointSolver:
    def __init__(self, adjoint_a, L, z, bcs):
        '''
        Solve a problem of this form:
        adjoint_a z = L

        Note, "adjoint" means that the Dirichlet boundary conditions are zero.

        '''

        if bcs is None:
            bcs = ()
        elif not isinstance(bcs, (list, tuple)):
            bcs = (bcs,)
        if not all(isinstance(bc, DirichletBC) for bc in bcs):
            raise TypeError('Parameter `bcs` must contain '
                             'homogenized `DirichletBC`(\'s)')
        if any(any(bc.get_boundary_values().values()) for bc in bcs):
            raise ValueError('Parameter `bcs` must contain '
                             'homogenized `DirichletBC`(\'s)')

        V = z.function_space()
        v0 = dolfin.TestFunction(V)

        if adjoint_a is not None:
            L_dummy = dot(v0, Constant((0.0,)*len(z)))*dx
            K, _ = assemble_system(adjoint_a, L_dummy, bcs)
            self._solver = dolfin.LUSolver(K, "mumps")

        dof_bcs = []
        for bc in bcs:
            dof_bcs.extend(bc.get_boundary_values().keys())
        self._dof_bcs = tuple(sorted(dof_bcs))

        self._x = z.vector()
        self._L = L

    def solve(self):
        rhs = assemble(self._L)
        rhs[self._dof_bcs] = 0.0
        return self._solver.solve(self._x, rhs)

    @property
    def solver(self):
        return self._solver

    @property
    def parameters(self):
        return self._solver.parameters


class SmoothingSolver:
    def __init__(self, V, kappa=None):

        v0 = dolfin.TestFunction(V)
        v1 = dolfin.TrialFunction(V)

        a = dot(v0, v1)*dx

        if kappa is not None:
            a += kappa*inner(grad(v0), grad(v1))*dx

        self._solver = dolfin.LUSolver(assemble(a), "mumps")
        self._solver.parameters["symmetric"] = True

    def solve(self, x):
        return self._solver.solve(x, x)

    @property
    def solver(self):
        return self._solver

    @property
    def parameters(self):
        return self._solver.parameters


class CriticalLoadSolver:
    def __init__(self, J, U, u, m, bcs, dx_m=None,
                 dims_m=None, use_nonlinear_solver=False):
        '''
        Parameters
        ----------
        J : ufl.Form
            Objective functional to maximize, e.g. linear-elastic strain energy.
        U : ufl.Form
            Linear-elastic strain energy.
        u : dolfin.Function
            Displacement field.
        m : dolfin.Function
            Vector-valued "body force" like traction field.
        bcs : (list of) dolfin.DirichletBC('s)
            Dirichlet boundary conditions.
        dx_m : dolfin.Measure or None (optional)
            "Body force" integration measure. Can be of cell type or exterior
            facet type; the default is cell type. Note, the integration measure
            can be defined on a subdomain (not necessarily on the whole domain).
        kappa : float or None (optional)
            Diffusion-like constant for smoothing the solution (`m`) increment.

        Notes
        -----
        If the "body force" integration measure `dx_m` concerns cells (as
        opposed to exterior facets), the gradient smoothing constant `kappa`
        can be `None`. Usually, `kappa` will need to be some small positive
        value if `dx_m` concerns exterior facets.

        '''

        if not isinstance(J, ufl_form_t):
            raise TypeError('Parameter `J`')

        if not isinstance(U, ufl_form_t):
            raise TypeError('Parameter `U`')

        if not isinstance(u, Function):
            raise TypeError('Parameter `u`')

        if not isinstance(m, Function):
            raise TypeError('Parameter `m`')

        if len(u) != len(m):
            raise ValueError('Functions `u` and `m` must live in the same dimension space')

        if bcs is None:
            bcs = ()
        elif not isinstance(bcs, (list, tuple)):
            bcs = (bcs,)
        if not all(isinstance(bc, DirichletBC) for bc in bcs):
            raise TypeError('Parameter `bcs` must contain '
                            'homogenized `DirichletBC`(\'s)')

        if dims_m is not None:
            if not isinstance(bcs, (list, tuple, range)):
                dims_m = (dims_m,)
            if not all(isinstance(dim_i, int) for dim_i in dims_m):
                raise TypeError('Parameter `dims_m`')
            if not all(0 <= dim_i < len(m) for dim_i in dims_m):
                raise ValueError('Parameter `dims_m`')

        if dx_m is None:
            dx_m = dx
        elif not isinstance(dx_m, dolfin.Measure):
            raise TypeError('Parameter `dx_m`')

        bcs_zro = []
        for bc in bcs:
            bc_zro = DirichletBC(bc)
            bc_zro.homogenize()
            bcs_zro.append(bc_zro)

        V_u = u.function_space()
        V_m = m.function_space()

        v0_u = dolfin.TestFunction(V_u)
        v0_m = dolfin.TestFunction(V_m)

        v1_u = dolfin.TrialFunction(V_u)
        v1_m = dolfin.TrialFunction(V_m)

        self._u = u
        self._m = m
        self._z = z = Function(V_u) # Adjoint solution

        self._J = J
        dJ_u = derivative(J, u, v0_u)
        self._dJ_m = derivative(J, m, v0_m)

        dU_u = derivative(U, u, v0_u) # Internal force
        d2U_uu = a = derivative(dU_u, u, v1_u) # Stiffness

        if dims_m is None: dW_u = L = dot(v0_u, m)*dx_m
        else: dW_u = L = sum(v0_u[i]*m[i] for i in dims_m)*dx_m

        self._ufl_norm_m = dolfin.sqrt(m**2)*dx_m # equiv. L1-norm
        self._assembled_adj_dW_um = assemble(dot(v0_m, v1_u)*dx_m)
        self._dimdofs_V_m = tuple(np.array(V_m_sub_i.dofmap().dofs())
                                  for V_m_sub_i in V_m.split())

        if use_nonlinear_solver:

            self._equilibrium_solver = dolfin.NonlinearVariationalSolver(
                dolfin.NonlinearVariationalProblem(dU_u-dW_u, u, bcs, d2U_uu))

            self._adjoint_solver = dolfin.LinearVariationalSolver(
                dolfin.LinearVariationalProblem(d2U_uu, dJ_u, z, bcs_zro))
            # NOTE: `d2U_uu` is equivalent to `adjoint(d2U_uu)` due to symmetry

            self._equilibrium_solver.parameters["symmetric"] = True
            self._adjoint_solver.parameters["symmetric"] = True

        else:

            self._equilibrium_solver = LinearEquilibriumSolver(a, L, u, bcs)
            self._adjoint_solver = LinearAdjointSolver(None, dJ_u, z, bcs_zro)
            self._adjoint_solver._solver = self._equilibrium_solver.solver
            self._equilibrium_solver.parameters['symmetric'] = True
            # NOTE: `a` is equivalent to `adjoint(a)` due to symmetry; however,
            #       the LU factorization can be reused in the adjoint solver.

        if dx_m.integral_type() == 'cell':
            kappa = None
        else:
            # Compute scale factor for `kappa` based on domain size
            mesh = V_m.mesh()
            xs = mesh.coordinates()
            domain_volume = assemble(1.0*dx(mesh))
            domain_length = (xs.max(0)-xs.min(0)).max()
            scale_factor = domain_volume / domain_length
            kappa = Constant(scale_factor*SMOOTHING_SOLVER_KAPPA)

        self._smoothing_solver = SmoothingSolver(V_m, kappa)


    def solve_equilibrium_problem(self):
        return self._equilibrium_solver.solve()

    def solve_adjoint_problem(self):
        return self._adjoint_solver.solve()

    def solve_smoothing_problem(self, dm_vec):
        return self._smoothing_solver.solve(dm_vec)


    def assemble_DJDm(self):
        return assemble(self._dJ_m) + self._assembled_adj_dW_um*self._z.vector()

    def assemble_dm_hat(self):
        v = self.assemble_DJDm()
        self.solve_smoothing_problem(v)
        return v / math.sqrt(v.inner(v))

    def compute_DJDm(self):
        self.solve_equilibrium_problem()
        self.solve_adjoint_problem()
        return self.assemble_DJDm()

    def compute_dm_hat(self):
        self.solve_equilibrium_problem()
        self.solve_adjoint_problem()
        return self.assemble_dm_hat()

    def compute_D2JDm2_hat(self, dm_vec, h=1e-3):
        '''Compute directional curvature (scalar) in the direction
        of `dm_vec` using finite differencing with stepsize `h`.'''

        if not isinstance(dm_vec, dolfin.GenericVector):
            raise TypeError('Parameter `dm_vec`')

        dm_hat = dm_vec / math.sqrt(dm_vec.inner(dm_vec))

        # Backup current state
        u0_vec = self._u.vector().copy()
        z0_vec = self._z.vector().copy()
        m0_vec = self._m.vector().copy()

        self._m.vector()[:] -= dm_hat * h
        dJdm_0 = self.compute_DJDm()

        self._m.vector()[:] += dm_hat*(2*h)
        dJdm_1 = self.compute_DJDm()

        # Restore backup state
        self._u.vector()[:] = u0_vec
        self._z.vector()[:] = z0_vec
        self._m.vector()[:] = m0_vec

        return dm_hat.inner((dJdm_1-dJdm_0)) / (2.0*h)

    def optimize(self, stepsize, cutoffs=tuple(1e-1*i for i in range(10)),
                 tolerance=1e-2, maxsteps=1000, callback_function=None):

        if not (isinstance(stepsize, float) and stepsize > 0.0):
            raise TypeError('Parameter `stepsize` must be a positive `float`')

        if not hasattr(cutoffs, '__len__'):
            cutoffs = (float(cutoffs),)
        elif not isinstance(cutoffs, tuple):
            cutoffs = tuple(cutoffs)

        if min(cutoffs) < 0.0: raise ValueError('Require `min(cutoffs) >= 0`')
        if max(cutoffs) >= 1.0: raise ValueError('Require `max(cutoffs) < 1`')

        if callback_function is None:
            isdef_callback_function = False
        elif callable(callback_function):
            isdef_callback_function = True
        else:
            raise TypeError('Parameter `callback_function`')

        m_vec = self._m.vector()
        norm_m = assemble(self._ufl_norm_m)

        self.solve_equilibrium_problem()
        self.solve_adjoint_problem()

        if isdef_callback_function:
            callback_function(0)

        istep, nstep = 1, maxsteps+1
        list_J = [assemble(self._J),]
        rtol_J = tolerance * stepsize

        for cutoff in cutoffs:
            logger.info(f'*** Load cutoff: {cutoff:.2g} ***')

            while istep < nstep:

                dm_vec = self.assemble_DJDm()
                self.solve_smoothing_problem(dm_vec)

                nodal_norms_dm = np.sqrt(sum(dm_vec[dofs]**2
                    for dofs in self._dimdofs_V_m))

                mask = nodal_norms_dm < nodal_norms_dm.max() * cutoff
                for dofs in self._dimdofs_V_m: dm_vec[dofs[mask]] = 0.0

                m_vec += dm_vec * (stepsize * math.sqrt(
                    m_vec.inner(m_vec)/dm_vec.inner(dm_vec)))

                m_vec *= norm_m / assemble(self._ufl_norm_m)

                self.solve_equilibrium_problem()
                self.solve_adjoint_problem()

                if isdef_callback_function:
                    callback_function(istep)

                list_J.append(assemble(self._J))

                logger.info(f'i:{istep: 3d}, '
                            f'J:{list_J[istep]: .3e}')

                istep += 1

                if (list_J[-1]-list_J[-2]) < list_J[-1]*rtol_J:
                    logger.info('Negligable increase in `J`')
                    is_converged = True
                    break

            else:
                logger.error('Iterations did not converge (BREAK)')
                is_converged = False
                break

        return is_converged, list_J[:istep]

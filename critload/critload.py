
import math
import dolfin
import logging
import numpy as np

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


class CriticalLoadSolver:
    def __init__(self, J, U, u, m, p, bcs, dx_m=None, dx_p=None, kappa_m=None, kappa_p=None):
        '''
        Parameters
        ----------
        J : ufl.Form
            Objective functional to maximize.
        U : ufl.Form
            Linea-elastic strain energy of solid.
        u : dolfin.Function
            Displacement field.
        m : dolfin.Function
            "Body force" field.
        bcs : (list of) dolfin.DirichletBC('s)
            Dirichlet boundary conditions.
        dx_m : dolfin.Measure or None (optional)
            "Body force" integration measure. Can be of cell type or exterior
            facet type; the default is cell type. Note, the integration measure
            can be defined on a subdomain (not necessarily on the whole domain).
        kappa : float or None (optional)
            Diffusion-like constant for smoothing the gradient of the objective.

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

        if not isinstance(p, Function):
            raise TypeError('Parameter `p`')

        if not isinstance(bcs, (list, tuple)): bcs = (bcs,)
        if not all(isinstance(bc, dolfin.DirichletBC) for bc in bcs):
            raise TypeError('Parameter `bcs`')

        if dx_m is None:
            dx_m = dx
        elif not isinstance(dx_m, dolfin.Measure):
            raise TypeError('Parameter `dx_m`')

        if dx_p is None:
            dx_p = dx
        elif not isinstance(dx_p, dolfin.Measure):
            raise TypeError('Parameter `dx_p`')

        bcs_zro = []
        bcs_dof = []

        for bc in bcs:
            bcs_dof.extend(bc.get_boundary_values().keys())
            bc_zro = dolfin.DirichletBC(bc); bc_zro.homogenize()
            bcs_zro.append(bc_zro)

        self._bcs = bcs if isinstance(bcs, tuple) else tuple(bcs)
        self._bcs_zro, self._bcs_dof = tuple(bcs_zro), tuple(bcs_dof)

        self._V_u = V_u = u.function_space()
        self._V_m = V_m = m.function_space()
        self._V_p = V_p = p.function_space()

        self._dimdofs_V_m = tuple(
            np.array(V_m_sub_i.dofmap().dofs())
            for V_m_sub_i in V_m.split())

        self._dimdofs_V_p = tuple(
            np.array(V_p_sub_i.dofmap().dofs())
            for V_p_sub_i in V_p.split())

        self._J = J
        self._U = U

        self._u = u
        self._m = m
        self._p = p

        # Adjoint solution
        self._z = Function(V_u)

        v0_u = dolfin.TestFunction(V_u)
        v0_m = dolfin.TestFunction(V_m)
        v0_p = dolfin.TestFunction(V_p)

        v1_u = dolfin.TrialFunction(V_u)
        v1_m = dolfin.TrialFunction(V_m)
        v1_p = dolfin.TrialFunction(V_p)

        self._dJ_u = derivative(J, u, v0_u)
        self._dJ_m = derivative(J, m, v0_m)

        self._ufl_norm_m = dolfin.sqrt(m**2)*dx_m
        self._ufl_norm_p = tuple(p_i*dx_p for p_i in p)

        dU_u = derivative(U, u, v0_u)
        d2U_uu = derivative(dU_u, u, v1_u)

        d2P_uu = sum(v0_u_i * v1_u_i * p_i * dx_p
            for v0_u_i, v1_u_i, p_i in zip(v0_u, v1_u, p))

        self._a = d2U_uu + d2P_uu
        self._L = dot(v0_u, m)*dx_m

        self._adj_F = assemble(dot(v0_m, v1_u)*dx_m)

        self._adj_d2P_pu = sum(v0_p_i * v1_u_i * u_i * dx_p
            for v0_p_i, v1_u_i, u_i in zip(v0_p, v1_u, u))

        a_m = dot(v0_m, v1_m)*dx
        a_p = dot(v0_p, v1_p)*dx

        if kappa_m is not None:
            if not isinstance(kappa_m, Constant):
                kappa_m = Constant(kappa_m)
            a_m += kappa_m*inner(grad(v0_m), grad(v1_m))*dx

        if kappa_p is not None:
            if not isinstance(kappa_p, Constant):
                kappa_p = Constant(kappa_p)
            a_p += kappa_p*inner(grad(v0_p), grad(v1_p))*dx

        self.smoothing_solver_m = dolfin.LUSolver(assemble(a_m), "mumps")
        self.smoothing_solver_p = dolfin.LUSolver(assemble(a_p), "mumps")

        self.smoothing_solver_m.parameters["symmetric"] = True
        self.smoothing_solver_p.parameters["symmetric"] = True

        self._A = None

    def solve_static_problem(self):
        '''Solve statics problem.'''

        self._A, b = assemble_system(self._a, self._L, self._bcs)

        return solve(self._A, self._u.vector(), b)

    def solve_adjoint_problem(self):
        '''Solve adjoint problem.'''

        rhs = assemble(self._dJ_u)
        rhs[self._bcs_dof] = 0.0

        return solve(self._A, self._z.vector(), rhs)

    def solve_smoothing_problem_m(self, x):
        return self.smoothing_solver_m.solve(x, x)

    def solve_smoothing_problem_p(self, x):
        return self.smoothing_solver_p.solve(x, x)

    def assemble_DJDm(self):
        return assemble(self._dJ_m) + self._adj_F*self._z.vector()

    def assemble_DJDp(self):
        return -(assemble(self._adj_d2P_pu)*self._z.vector())

    def assemble_dm_hat(self):
        dm = self.assemble_DJDm()
        self.solve_smoothing_problem_m(dm)
        return dm / math.sqrt(dm.inner(dm))

    def assemble_dp_hat(self):
        dp = self.assemble_DJDp()
        self.solve_smoothing_problem_p(dp)
        return dp / math.sqrt(dp.inner(dp))

    def compute_DJDm(self):
        self.solve_static_problem()
        self.solve_adjoint_problem()
        return self.assemble_DJDm()

    def compute_DJDp(self):
        self.solve_static_problem()
        self.solve_adjoint_problem()
        return self.assemble_DJDp()

    def compute_dm_hat(self):
        self.solve_static_problem()
        self.solve_adjoint_problem()
        return self.assemble_dm_hat()

    def compute_dp_hat(self):
        self.solve_static_problem()
        self.solve_adjoint_problem()
        return self.assemble_dp_hat()

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

    def optimize(self, maxsteps, stepsizes_m, relative_stepsize_p, cutoffs_m,
            norms_p, tolerance=1e-4, num_edge_items=3, external_callable=None):
        '''
        Parameters
        ----------
        num_edge_items : int
            Compate the last `num_edge_items` of the objective function values
            with the previous `num_edge_items` values for determining solution
            convergence. Roughly speaking, the convergence criterion is:

            `sum(list_J[-num_edge_items*2:-num_edge_items]) \
                > sum(list_J[-num_edge_items:])*(1.0-tolerance)`

        '''

        if not hasattr(stepsizes_m, '__len__'):
            stepsizes_m = (float(stepsizes_m),)
        elif not isinstance(stepsizes_m, tuple):
            stepsizes_m = tuple(stepsizes_m)

        if not hasattr(cutoffs_m, '__len__'):
            cutoffs_m = (float(cutoffs_m),)
        elif not isinstance(cutoffs_m, tuple):
            cutoffs_m = tuple(cutoffs_m)

        if not hasattr(norms_p, '__len__'):
            norms_p = (float(norms_p),)
        elif not isinstance(norms_p, tuple):
            norms_p = tuple(norms_p)

        n = max(len(stepsizes_m), len(cutoffs_m), len(norms_p))

        if len(stepsizes_m) < n:
            stepsizes_m += stepsizes_m[-1:] * (n-len(stepsizes_m))

        if len(cutoffs_m) < n:
            cutoffs_m += cutoffs_m[-1:] * (n-len(cutoffs_m))

        if len(norms_p) < n:
            norms_p += norms_p[-1:] * (n-len(norms_p))

        if external_callable is None:
            isdef_external_callable = False
        elif callable(external_callable):
            isdef_external_callable = True
        else:
            raise TypeError('Parameter `external_callable`')

        m_vec = self._m.vector()
        p_vec = self._p.vector()

        norm_m = assemble(self._ufl_norm_m)

        for dofs_p_i, ufl_norm_p_i in zip(self._dimdofs_V_p, self._ufl_norm_p):
            p_vec[dofs_p_i] *= norms_p[0] / assemble(ufl_norm_p_i)

        self.solve_static_problem()
        self.solve_adjoint_problem()

        if isdef_external_callable:
            external_callable()

        list_J = [None,]*(maxsteps+1)
        list_J[0] = assemble(self._J)

        istep, nstep = 1, maxsteps+1

        for stepsize_m, cutoff_m, norm_p in \
                zip(stepsizes_m, cutoffs_m, norms_p):

            stepsize_p = relative_stepsize_p * stepsize_m

            logger.info('*** '
                        f'stepsize_m: {stepsize_m:.2g}, '
                        f'stepsize_p: {stepsize_p:.2g}, '
                        f'cutoff_m: {cutoff_m:.2g}, '
                        f'norm_p: {norm_p:.2g} '
                        '***')

            while istep < nstep:

                dm_vec = self.assemble_DJDm()
                self.solve_smoothing_problem_m(dm_vec)

                nodal_norms_dm = np.sqrt(sum(dm_vec[dofs]**2
                    for dofs in self._dimdofs_V_m))

                mask = nodal_norms_dm < nodal_norms_dm.max() * cutoff_m
                for dofs in self._dimdofs_V_m: dm_vec[dofs[mask]] = 0.0

                m_vec += dm_vec * (stepsize_m * math.sqrt(
                    m_vec.inner(m_vec)/dm_vec.inner(dm_vec)))

                m_vec *= norm_m / assemble(self._ufl_norm_m)


                dp_vec = self.assemble_DJDp()
                self.solve_smoothing_problem_p(dp_vec)

                dp_vec[np.flatnonzero((p_vec==0.0)*(dp_vec<0.0))] = 0.0

                p_vec += dp_vec * (stepsize_p * math.sqrt(
                    p_vec.inner(p_vec)/dp_vec.inner(dp_vec)))

                p_vec[np.flatnonzero(p_vec < 0.0)] = 0.0

                for dofs_p_i, ufl_norm_p_i in zip(self._dimdofs_V_p, self._ufl_norm_p):
                    p_vec[dofs_p_i] *= norm_p / assemble(ufl_norm_p_i)


                self.solve_static_problem()
                self.solve_adjoint_problem()

                if isdef_external_callable:
                    external_callable()

                list_J[istep] = assemble(self._J)

                logger.info(f'i:{istep: 3d}, '
                            f'J:{list_J[istep]: .3e}')

                istep += 1

                j = istep - num_edge_items
                if j >= num_edge_items and \
                   sum(list_J[j-num_edge_items:j]) > \
                   sum(list_J[j:istep])*(1.0-tolerance):
                    logger.info('Negligable increase in `J` (BREAK)')
                    is_converged = True
                    break

            else:
                logger.error('Iterations did not converge (BREAK)')
                is_converged = False
                break

        return is_converged, list_J[:istep]

# Copyright (C) 2025-2026 by Andrew Hoffman <hoffmaao@uw.edu>
#
# This file is part of multilayer.
#
# multilayer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# multilayer source directory or at <http://www.gnu.org/licenses/>.

r"""First-order (Blatter-Pattyn) flowline reference for a flow law that
may change with height, independent of the layered model: the solution
the layered column converges to as layers are added
(``test/first_order_test.py``).

The full first-order stress balance in x-z,

.. math::
    \partial_x(4\eta\,\partial_x u) + \partial_z(\eta\,\partial_z u)
    = \rho g\,\partial_x s, \qquad
    \eta = \tfrac12 A^{-1/n}\dot\varepsilon_e^{1/n - 1}, \qquad
    \dot\varepsilon_e^2 = (\partial_x u)^2 + \tfrac14(\partial_z u)^2,

solved as the minimum of its action on a terrain-following mesh, periodic
in x, with Q2 elements graded toward the bed and toward every change of
flow law (each change sits on a mesh level).  As in the multilayer
ISMIP-HOM scripts the surface is flat and the slope enters as a uniform
body force.  At ISMIP-HOM's density it reproduces the intercomparison
median of experiments B and D to within 0.1-1.4 % wherever the full-Stokes
and first-order submissions agree (B at 20-160 km, D at 5-160 km).

Units are icepack2's: MPa, m, yr.
"""
import numpy as np
import firedrake
from firedrake import (Constant, Function, SpatialCoordinate, as_vector,
                       conditional, derivative, ds_b, dx, lt)
from icepack2.constants import ice_density, gravity

#: Stress [MPa] at which the continuation in n holds each law's strain rate.
TAU_REF = 0.1

SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 200,
    "snes_rtol": 1e-11,
    "snes_atol": 1e-10,
    "snes_divergence_tolerance": -1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def _level_heights(interfaces, cells_per_unit, min_cells=6, ratio=1.08):
    """Mesh levels as fractions of the thickness: cells growing by ``ratio``
    upward from the bed and from every flow-law change, a level on each."""
    cuts = [0.0, *sorted(interfaces), 1.0]
    levels = [0.0]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        n = max(min_cells, int(round(cells_per_unit * (hi - lo))))
        sizes = ratio ** np.arange(n)
        levels.extend(lo + (hi - lo) * np.cumsum(sizes / sizes.sum()))
    levels[-1] = 1.0
    return np.array(levels)


class Solution:
    """Velocity of a first-order solve, sampled by height fraction."""

    def __init__(self, u, Lx, thickness, bed):
        self.u, self.Lx = u, Lx
        self._thickness, self._bed = thickness, bed

    def at(self, x_hat, zeta):
        """Along-flow velocity at ``x_hat`` (fractions of the period) and
        height fraction ``zeta`` above the bed (scalar or one per point)."""
        x = np.asarray(x_hat, dtype=float) * self.Lx
        H = np.array([float(self._thickness(xi)) for xi in x])
        b = np.array([float(self._bed(xi)) for xi in x])
        zeta = np.broadcast_to(np.asarray(zeta, dtype=float), x.shape)
        # Keep points a hair inside the domain for point location.
        x = np.clip(x, 1e-9 * self.Lx, (1 - 1e-9) * self.Lx)
        zeta = np.clip(zeta, 1e-9, 1 - 1e-9)
        return np.array(self.u.at(np.column_stack([x, b + zeta * H]).tolist()))

    def layer_midpoints(self, x_hat, fractions):
        """Velocity at the mid-height of each layer, bottom to top: the
        points the multilayer layer velocities stand for."""
        edges = np.concatenate([[0.0], np.cumsum(fractions)])
        return np.array([self.at(x_hat, 0.5 * (a + b))
                         for a, b in zip(edges[:-1], edges[1:])])


def solve(Lx, thickness, bed, tan_alpha, laws, beta2=None, nx=80,
          cells_per_unit=60, steps=10, eps0=1e-6, rho=ice_density):
    r"""Solve the first-order flowline problem.

    Parameters
    ----------
    Lx : float
        Period [m].
    thickness, bed : callable
        ``f(x)``, called with the UFL x coordinate to build the mesh and
        with floats to sample it, e.g. ``lambda x: 1000 - 500 * sin(w * x)``
        with ``sin`` from :mod:`firedrake`, which takes both.
    tan_alpha : float
        Surface slope, applied as a uniform body force.
    laws : list of (zeta_top, n, A)
        Flow laws from the bed up; each holds up to height fraction
        ``zeta_top``; the last has ``zeta_top = 1``.
    beta2 : callable, optional
        Linear sliding coefficient ``f(x)`` [MPa yr/m]; frozen bed if None.
    rho : float
        Ice density, icepack2 units.
    """
    levels = _level_heights([zt for zt, _, _ in laws[:-1]], cells_per_unit)
    base = firedrake.PeriodicIntervalMesh(nx, Lx)
    mesh = firedrake.ExtrudedMesh(base, layers=len(levels) - 1,
                                  layer_height=np.diff(levels))

    # Flow laws, cellwise on the reference (x, zeta) mesh.
    X = SpatialCoordinate(mesh)
    DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
    n_expr, A_expr = Constant(laws[-1][1]), Constant(laws[-1][2])
    for zt, n_l, A_l in reversed(laws[:-1]):
        n_expr = conditional(lt(X[1], zt), Constant(n_l), n_expr)
        A_expr = conditional(lt(X[1], zt), Constant(A_l), A_expr)
    n_target = Function(DG0).interpolate(n_expr)
    A_target = Function(DG0).interpolate(A_expr)

    # Terrain-following coordinates: z = b(x) + zeta H(x).
    Vc = mesh.coordinates.function_space()
    mesh.coordinates.assign(Function(Vc).interpolate(
        as_vector([X[0], bed(X[0]) + X[1] * thickness(X[0])])))
    x, _ = SpatialCoordinate(mesh)

    V = firedrake.FunctionSpace(mesh, "CG", 2)
    u = Function(V)
    n, A = Function(DG0), Function(DG0)
    eps_e2 = u.dx(0) ** 2 + 0.25 * u.dx(1) ** 2 + Constant(eps0) ** 2
    action = (2 * n / (n + 1) * A ** (-1 / n) * eps_e2 ** ((n + 1) / (2 * n))
              - Constant(rho) * gravity * Constant(tan_alpha) * u) * dx
    bcs = []
    if beta2 is None:
        bcs = [firedrake.DirichletBC(V, 0, "bottom")]
    else:
        action += 0.5 * beta2(x) * u ** 2 * ds_b
    problem = firedrake.NonlinearVariationalProblem(
        derivative(action, u), u, bcs,
        form_compiler_parameters={"quadrature_degree": 8})
    solver = firedrake.NonlinearVariationalSolver(
        problem, solver_parameters=SOLVER_PARAMETERS)

    n_vals, A_vals = n_target.dat.data_ro.copy(), A_target.dat.data_ro.copy()
    for lam in np.linspace(0.0, 1.0, steps):
        n_l = 1.0 + lam * (n_vals - 1.0)
        n.dat.data[:] = n_l
        A.dat.data[:] = A_vals * TAU_REF ** (n_vals - n_l)
        solver.solve()

    return Solution(u, Lx, thickness, bed)

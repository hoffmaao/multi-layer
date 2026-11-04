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

r"""The layered column converges to the first-order model.

Each layer velocity stands for the velocity at the layer's mid-height, so
as layers are added the column should converge to the first-order
(Blatter-Pattyn) solution there, here computed independently by
``multilayer.first_order``.  Three things decide whether it does, and each
case below fails without one of them:

* a sloping bed (ISMIP-HOM B geometry, frozen bed, n = 3): the layers
  slope with the bed, and ``slope_coupling`` restores the strain rate at
  constant depth;
* a sliding bed under nonlinear ice (ISMIP-HOM D friction, the poster
  slab of multi-layer-AGU, n = 1.8): ``layer_power`` shears the bottom
  half-layer over the sliding bed, and gives membrane and shear stress one
  effective stress, so ice that both stretches and shears is not too stiff;
* a change of flow law with height (200 m of n = 4 basal ice under
  n = 1.8 ice): each half-layer shears under its own law.

Every case halves the layer thickness and asks for the error to fall by
at least a factor of three (second order gives four).  The flow is
uniform across the domain, so the mesh is a strip two cells wide.

    python -u first_order_test.py      (or pytest; a few minutes)
"""
import math

import numpy as np
import firedrake
from firedrake import (Constant, Function, derivative, dx, sin,
                       NonlinearVariationalProblem, NonlinearVariationalSolver)
from icepack2.constants import ice_density, gravity
from multilayer import first_order
from multilayer.model import minimization, utilities

TAU_REF = 0.1
X_HAT = np.linspace(0.0, 1.0, 101)

SOLVER_PARAMETERS = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_linesearch_type": "bt",
    "snes_divergence_tolerance": -1,
    "snes_rtol": 1e-10,
    "snes_atol": 1e-12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def layered(Lx, thickness, tan_alpha, fractions, laws, beta2=None, nx=80,
            steps=12):
    r"""Layer velocities [layer, x] of the multilayer column: ``laws`` is
    one (n, A) per layer, bottom to top; frozen bed if ``beta2`` is None."""
    L = len(fractions)
    mesh = firedrake.PeriodicRectangleMesh(nx, 2, Lx, 2 * Lx / nx, direction="x")
    x, _ = firedrake.SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    H = Function(Q).interpolate(thickness(x))
    h = utilities.layer_thicknesses(H, L, fractions=list(fractions))
    s = Function(Q)                     # flat: the slope is a body force

    n = [Constant(1.0) for _ in laws]
    A = [Constant(a * TAU_REF ** (nn - 1.0)) for nn, a in laws]
    Z = utilities.create_function_space(mesh, L)
    z = Function(Z)
    for l in range(L):
        z.sub(3 * l).interpolate(firedrake.as_vector([Constant(10.0), Constant(0.0)]))
    f = utilities.split_fields(firedrake.split(z), L)

    J = Constant(0.0) * dx(mesh)
    for l in range(L):
        S_above = f[l + 1]["interlayer_stress"] if l < L - 1 else None
        J += minimization.layer_power(
            membrane_stress=f[l]["membrane_stress"],
            stress_below=f[l]["interlayer_stress"], stress_above=S_above,
            thickness=h[l], flow_law_coefficient=A[l], flow_law_exponent=n[l])
        J += minimization.momentum_balance(
            velocity=f[l]["velocity"], membrane_stress=f[l]["membrane_stress"],
            thickness=h[l], surface=s,
            basal_stress=f[l]["interlayer_stress"] if l == 0 else None,
            stress_above=S_above,
            stress_below=f[l]["interlayer_stress"] if l > 0 else None)
        J += ice_density * gravity * h[l] * Constant(tan_alpha) * f[l]["velocity"][0] * dx
    J += minimization.slope_coupling(
        velocities=[fl["velocity"] for fl in f],
        membrane_stresses=[fl["membrane_stress"] for fl in f],
        thicknesses=h, frozen_bed=beta2 is None)
    if beta2 is not None:
        K = Constant(1.0) / (Function(Q).interpolate(beta2(x)) + Constant(1e-10))
        J += minimization.friction_power(
            basal_stress=f[0]["interlayer_stress"], sliding_coefficient=K,
            sliding_exponent=Constant(1.0))

    bcs = [firedrake.DirichletBC(Z.sub(3 * l).sub(1), 0, (1, 2)) for l in range(L)]
    problem = NonlinearVariationalProblem(
        derivative(J, z), z, bcs, form_compiler_parameters={"quadrature_degree": 8})
    solver = NonlinearVariationalSolver(problem, solver_parameters=SOLVER_PARAMETERS)
    for lam in np.linspace(0.0, 1.0, steps):
        for l, (nn, a) in enumerate(laws):
            n_l = 1.0 + lam * (nn - 1.0)
            n[l].assign(n_l)
            A[l].assign(a * TAU_REF ** (nn - n_l))
        solver.solve()

    pts = [[xi * Lx, Lx / nx] for xi in X_HAT]
    return np.array([[u[0] for u in z.subfunctions[3 * l].at(pts)] for l in range(L)])


def error(reference, velocities, fractions):
    """RMS difference from the first-order solution at the layer
    mid-heights, relative to its largest speed there."""
    target = reference.layer_midpoints(X_HAT, fractions)
    return np.sqrt(np.mean((velocities - target) ** 2)) / np.abs(target).max()


def converges(name, Lx, thickness, bed, tan_alpha, columns, laws_of, beta2,
              reference_laws, finest):
    reference = first_order.solve(Lx, thickness, bed, tan_alpha, reference_laws,
                                  beta2=beta2)
    errors = []
    for fractions in columns:
        u = layered(Lx, thickness, tan_alpha, fractions, laws_of(fractions), beta2=beta2)
        errors.append(error(reference, u, fractions))
        print(f"{name}: {len(fractions):2d} layers, error {errors[-1]:.2%}", flush=True)
    assert errors[-1] < finest, f"{name}: {errors[-1]:.2%} at the finest column"
    assert errors[0] / errors[1] > 3, f"{name}: error fell only {errors[0] / errors[1]:.1f}x"


def test_sloping_bed():
    Lx = 80e3
    w = 2 * math.pi / Lx
    converges("ISMIP-HOM B 80 km", Lx,
              thickness=lambda x: 1000 - 500 * sin(w * x),
              bed=lambda x: -1000 + 500 * sin(w * x),
              tan_alpha=math.tan(math.radians(0.5)),
              columns=[[1 / 5] * 5, [1 / 10] * 10],
              laws_of=lambda fr: [(3.0, 100.0)] * len(fr),
              beta2=None, reference_laws=[(1.0, 3.0, 100.0)], finest=0.01)


SLAB_L = 6000.0
SLAB_H = 600.0


def _slab_beta2(x):
    return (1000 + 1000 * sin(2 * math.pi * x / SLAB_L)) * 1e-6


def test_sliding_bed():
    converges("sliding slab, n = 1.8", SLAB_L,
              thickness=lambda x: Constant(SLAB_H) + 0 * x,
              bed=lambda x: Constant(-SLAB_H) + 0 * x,
              tan_alpha=math.tan(math.radians(2.3)),
              columns=[[1 / 4] * 4, [1 / 8] * 8],
              laws_of=lambda fr: [(1.8, 100.0)] * len(fr),
              beta2=_slab_beta2, reference_laws=[(1.0, 1.8, 100.0)], finest=0.01)


def test_rheology_change():
    zb = 200.0 / SLAB_H                     # 200 m of basal ice

    def column(k):
        return [zb / k] * k + [(1 - zb) / k] * k

    def laws_of(fr):
        k = len(fr) // 2
        return [(4.0, 1000.0)] * k + [(1.8, 100.0)] * k

    converges("200 m of basal ice", SLAB_L,
              thickness=lambda x: Constant(SLAB_H) + 0 * x,
              bed=lambda x: Constant(-SLAB_H) + 0 * x,
              tan_alpha=math.tan(math.radians(2.3)),
              columns=[column(2), column(4)], laws_of=laws_of,
              beta2=_slab_beta2,
              reference_laws=[(zb, 4.0, 1000.0), (1.0, 1.8, 100.0)], finest=0.02)


if __name__ == "__main__":
    test_sloping_bed()
    test_sliding_bed()
    test_rheology_change()

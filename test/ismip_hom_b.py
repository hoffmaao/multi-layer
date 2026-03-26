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

r"""ISMIP-HOM Experiment B: ice flow over a rippled bed.

Pattyn et al. (2008), Section 3.2. Also run by Jouvet (2015), Section 5.

Geometry:
  - Surface: z_s = -x tan(alpha), alpha = 0.5 deg
  - Bed:     z_b = z_s - 1000 + 500 sin(2 pi x / L)
  - Thickness: h = 1000 - 500 sin(2 pi x / L) [periodic]
  - Domain: L x L, periodic in x, side walls in y
  - Frozen base (no-slip): v(z_b) = 0

Note on periodic surface:
  The surface z_s = -x tan(alpha) is NOT periodic, so we pass surface = 0
  to momentum_balance (zeroing the driving-stress and DG jump terms)
  and add the constant driving stress rho*g*h^l*tan(alpha) separately.
"""
import numpy as np
import math
import firedrake
from firedrake import (
    Constant,
    Function,
    sin,
    dx,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
from icepack2.constants import ice_density as ρ_I, gravity as g, glen_flow_law
from multilayer.model import minimization, utilities


alpha_deg = 0.5
tan_alpha = math.tan(math.radians(alpha_deg))

sparams = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_linesearch_type": "nleqerr",
    "ksp_type": "gmres",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def solve_ismip_hom_b(L_domain, num_layers, nx=40):
    L = num_layers
    Lx = L_domain

    mesh = firedrake.PeriodicRectangleMesh(nx, nx, Lx, Lx, direction="x")
    x, y = firedrake.SpatialCoordinate(mesh)

    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    omega = Constant(2 * np.pi / Lx)
    h_total = Function(Q).interpolate(Constant(1000.0) - Constant(500.0) * sin(omega * x))
    s_zero = Function(Q).assign(0.0)
    h_layers = utilities.layer_thicknesses(h_total, L)

    n = Constant(glen_flow_law)
    A = Constant(100.0)
    rheology = {"flow_law_coefficient": A, "flow_law_exponent": n}

    Z = utilities.create_function_space(mesh, L)
    z = Function(Z)
    fields = utilities.split_fields(firedrake.split(z), L)

    Lagrangian = Constant(0) * dx(mesh)

    for l in range(L):
        h_l = h_layers[l]
        fl = fields[l]

        Lagrangian += minimization.viscous_power(
            membrane_stress=fl["membrane_stress"], thickness=h_l, **rheology,
        )

        S_above = fields[l + 1]["interlayer_stress"] if l < L - 1 else None
        Lagrangian += minimization.momentum_balance(
            velocity=fl["velocity"],
            membrane_stress=fl["membrane_stress"],
            thickness=h_l,
            surface=s_zero,
            basal_stress=fl["interlayer_stress"] if l == 0 else None,
            stress_above=S_above,
            stress_below=fl["interlayer_stress"] if l > 0 else None,
        )

        # Constant driving stress (replaces grad(s) for periodic domain)
        Lagrangian += ρ_I * g * h_l * Constant(tan_alpha) * fl["velocity"][0] * dx

    # Basal stress power (frozen base)
    Lagrangian += minimization.basal_stress_power(
        basal_stress=fields[0]["interlayer_stress"],
        thickness=h_layers[0],
        **rheology,
    )

    # Interlayer stress power (interior interfaces)
    for l in range(1, L):
        Lagrangian += minimization.interlayer_power(
            interlayer_stress=fields[l]["interlayer_stress"],
            thickness_above=h_layers[l],
            thickness_below=h_layers[l - 1],
            **rheology,
        )

    F = derivative(Lagrangian, z)
    bcs = [firedrake.DirichletBC(Z.sub(3 * l).sub(1), 0, (1, 2)) for l in range(L)]

    params = {"form_compiler_parameters": {"quadrature_degree": 8}}
    problem = NonlinearVariationalProblem(F, z, bcs, **params)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparams)

    for lam in np.linspace(0.0, 1.0, 5):
        n.assign((1 - lam) + lam * glen_flow_law)
        solver.solve()

    # Extract surface velocity along centreline
    u_top = z.subfunctions[3 * (L - 1)]
    n_sample = 101
    x_hat = np.linspace(0, 1, n_sample)
    y_mid = Lx / 2
    u_surface = np.array([u_top.at([xi * Lx, y_mid])[0] for xi in x_hat])
    return u_surface, x_hat, z


def main():
    wavelengths = [160e3, 80e3, 40e3, 20e3, 10e3]
    layer_counts = [1, 2, 3]

    print("ISMIP-HOM Experiment B: Multilayer dual form")
    print("=" * 60)
    print(f"{'L (km)':>8s}", end="")
    for nl in layer_counts:
        print(f"  {'L=' + str(nl) + ' max u_x':>14s}", end="")
    print()
    print("-" * 60)

    for Lx in wavelengths:
        print(f"{Lx/1e3:8.0f}", end="", flush=True)
        for nl in layer_counts:
            nx = min(max(20, int(40 * 80e3 / Lx)), 80)
            u_surf, x_hat, _ = solve_ismip_hom_b(Lx, nl, nx=nx)
            print(f"  {np.max(u_surf):14.4f}", end="", flush=True)
        print()


if __name__ == "__main__":
    main()

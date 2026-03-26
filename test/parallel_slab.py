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

r"""Parallel slab verification for the multilayer model.

Verifies the multilayer dual-form model against the analytical solution
for an infinite parallel-sided slab on a constant slope with a frozen
(no-slip) base (Jouvet 2015, Section 3).

In this configuration:
- Membrane stresses are zero (no horizontal gradients)
- All motion comes from interlayer shear
- The velocity profile is a power-law in the vertical
- As L -> infinity, the surface velocity converges to the SIA solution:
    u_s = 2A/(n+1) * (rho*g*sin(alpha))^n * H^{n+1}
"""
import numpy as np
import firedrake
from firedrake import (
    Constant,
    Function,
    derivative,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
from multilayer.model import minimization, utilities
from multilayer.constants import (
    ice_density as ρ_I,
    gravity as g,
    glen_flow_law,
)


def analytical_velocities(H, α, A, n, L):
    r"""Compute the analytical per-layer velocities for the parallel slab.

    Parameters
    ----------
    H : float, ice thickness (m)
    α : float, surface slope (radians)
    A : float, rate factor (MPa^{-n} yr^{-1})
    n : float, Glen exponent
    L : int, number of layers

    Returns
    -------
    u : ndarray of shape (L,), velocity of each layer (m/yr)
    """
    τ_d = float(ρ_I) * float(g) * float(H) * np.sin(float(α))
    h_l = float(H) / L

    u = np.zeros(L)
    # Bottom layer: u^1 = h^1 * A * |S^0|^{n-1} * S^0
    # where S^0 = rho*g*H*sin(alpha) (total driving stress at base)
    S_0 = τ_d
    u[0] = h_l * float(A) * S_0 ** float(n)

    # Interior layers: u^{l+1} = u^l + 2*h_l * A * |S^l|^n
    # where S^l = tau_d * (L - l) / L
    for l in range(1, L):
        S_l = τ_d * (L - l) / L
        u[l] = u[l - 1] + 2 * h_l * float(A) * S_l ** float(n)

    return u


def sia_surface_velocity(H, α, A, n):
    r"""Exact SIA surface velocity for a parallel slab.

    .. math::
        u_s = \frac{2A}{n+1} (\rho g \sin\alpha)^n H^{n+1}
    """
    τ_d = float(ρ_I) * float(g) * float(H) * np.sin(float(α))
    return 2 * float(A) / (float(n) + 1) * τ_d ** float(n) * float(H)


def solve_parallel_slab(num_layers, nx=16):
    r"""Solve the parallel slab problem with L layers.

    Returns the computed per-layer x-velocities (evaluated at the domain centre).
    """
    L = num_layers

    # Domain: square, tilted surface
    Lx = Constant(20e3)
    Ly = Constant(20e3)
    mesh = firedrake.RectangleMesh(nx, nx, float(Lx), float(Ly), diagonal="crossed")
    x, y = firedrake.SpatialCoordinate(mesh)

    # Physical parameters
    H = Constant(1000.0)
    α = Constant(0.01)  # ~ 0.57 degrees
    n = Constant(glen_flow_law)

    # Rate factor: choose so velocities are O(100) m/yr
    τ_c = Constant(0.1)
    ε_c = Constant(0.01)
    A = ε_c / τ_c ** n

    # Surface elevation: tilted plane s(x) = const - x*sin(alpha)
    # We set the surface so that grad(s) = (-sin(alpha), 0)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    s = Function(Q).interpolate(H - x * Constant(np.sin(float(α))))

    # Create multilayer function space
    Z = utilities.create_function_space(mesh, L)
    z = Function(Z)

    # Per-layer thickness
    h_layers = utilities.layer_thicknesses(H, L)

    # Split fields
    fields = utilities.split_fields(firedrake.split(z), L)

    # Assemble the action functional
    rheology = {
        "flow_law_coefficient": A,
        "flow_law_exponent": n,
    }

    Lagrangian = Constant(0) * firedrake.dx(mesh)

    for l in range(L):
        h_l = h_layers[l]
        fl = fields[l]

        # Viscous power per layer
        Lagrangian += minimization.viscous_power(
            membrane_stress=fl["membrane_stress"],
            thickness=h_l,
            **rheology,
        )

        # Momentum balance per layer
        S_above = fields[l + 1]["interlayer_stress"] if l < L - 1 else None
        S_below = fl["interlayer_stress"]

        Lagrangian += minimization.momentum_balance(
            velocity=fl["velocity"],
            membrane_stress=fl["membrane_stress"],
            thickness=h_l,
            surface=s,
            stress_above=S_above,
            stress_below=S_below,
        )

    # Basal stress power (frozen base, no-slip)
    Lagrangian += minimization.basal_stress_power(
        basal_stress=fields[0]["interlayer_stress"],
        thickness=h_layers[0],
        **rheology,
    )

    # Interlayer stress power for interior interfaces
    for l in range(1, L):
        Lagrangian += minimization.interlayer_power(
            interlayer_stress=fields[l]["interlayer_stress"],
            thickness_above=h_layers[l],
            thickness_below=h_layers[l - 1],
            **rheology,
        )

    # Take derivative to get the residual
    F = derivative(Lagrangian, z)

    # Boundary conditions: no Dirichlet BCs needed for the parallel slab.
    # The solution is uniform in x and y; the driving stress comes from
    # grad(s) and is balanced by interlayer/basal shear.
    # We pin one velocity node to remove the translational nullspace
    # from the membrane stress equation.
    bcs = [firedrake.DirichletBC(Z.sub(3 * l), 0, "on_boundary") for l in range(L)]

    params = {"form_compiler_parameters": {"quadrature_degree": 8}}
    problem = NonlinearVariationalProblem(F, z, bcs, **params)
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_max_it": 200,
            "snes_linesearch_type": "nleqerr",
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    # Continuation: ramp n from 1 to glen_flow_law
    num_continuation_steps = 5
    for λ in np.linspace(0.0, 1.0, num_continuation_steps):
        n.assign((1 - λ) + λ * glen_flow_law)
        solver.solve()

    # Extract the x-component of velocity for each layer
    # Evaluate at the domain centre
    subfunctions = z.subfunctions
    u_computed = np.zeros(L)
    xc, yc = float(Lx) / 2, float(Ly) / 2
    for l in range(L):
        u_l = subfunctions[3 * l]
        u_computed[l] = u_l.at([xc, yc])[0]

    return u_computed


def main():
    H = 1000.0
    α = 0.01
    τ_c = 0.1
    ε_c = 0.01
    n = glen_flow_law
    A = float(ε_c) / float(τ_c) ** n

    u_sia = sia_surface_velocity(H, α, A, n)
    print(f"SIA analytical surface velocity: {u_sia:.4f} m/yr")
    print()

    layer_counts = [1, 2, 4, 8]
    print(f"{'L':>4s}  {'u_surface (computed)':>20s}  {'u_surface (analytical)':>22s}  {'relative error':>14s}")
    print("-" * 70)

    for L in layer_counts:
        u_exact = analytical_velocities(H, α, A, n, L)
        u_computed = solve_parallel_slab(L, nx=8)
        u_s_exact = u_exact[-1]
        u_s_comp = u_computed[-1]
        rel_err = abs(u_s_comp - u_s_exact) / abs(u_s_exact) if u_s_exact != 0 else 0
        print(f"{L:4d}  {u_s_comp:20.4f}  {u_s_exact:22.4f}  {rel_err:14.2e}")

    print()
    print(f"SIA limit (L -> inf): {u_sia:.4f} m/yr")


if __name__ == "__main__":
    main()

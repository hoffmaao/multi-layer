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

r"""ISMIP-HOM Experiment B with composite rheology: n=4 bottom, n=1.8 top.

Same geometry as ismip_hom_b.py (frozen base, sinusoidal bed bumps) but
with per-layer rheology following Goldsby & Kohlstedt / Ranganathan & Minchew:
  - Layer 1 (bottom, 200/2500 = 8%):  n=4 dislocation creep, A=40 MPa^{-4} yr^{-1}
  - Layer 2 (top,    2300/2500 = 92%): n=1.8 grain boundary sliding, A=1.0 MPa^{-1.8} yr^{-1}
"""
import numpy as np
import math
import firedrake
from firedrake import (
    Constant, Function, sin, dx, dS, FacetNormal,
    inner, sym, grad, avg, jump, derivative,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)
from icepack2.model.minimization import viscous_power
from icepack2.constants import ice_density as ρ_I, gravity as g
from multilayer.model.minimization import basal_stress_power, interlayer_power
from multilayer.model.utilities import create_function_space, split_fields, layer_thicknesses


alpha_deg = 0.5
tan_alpha = math.tan(math.radians(alpha_deg))

LAYER_FRACTIONS = [0.15, 0.85]
LAYER_EXPONENTS = [4.0, 1.8]
# Calibrated to match Glen n=3, A=100 at tau_ref = 0.1 MPa:
#   A_new = A_ref * tau_ref^(n_ref - n_new)
LAYER_PREFACTORS = [1000.0, 6.3]  # MPa^{-n} yr^{-1}
NUM_LAYERS = 2

sparams = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_linesearch_type": "bt",
    "snes_divergence_tolerance": -1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def solve_ismip_hom_b_composite(L_domain, nx=40):
    L = NUM_LAYERS
    Lx = L_domain

    mesh = firedrake.PeriodicRectangleMesh(nx, nx, Lx, Lx, direction="x")
    x, y = firedrake.SpatialCoordinate(mesh)

    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    omega = Constant(2 * np.pi / Lx)
    h_total = Function(Q).interpolate(Constant(1000.0) - Constant(500.0) * sin(omega * x))
    h_layers = layer_thicknesses(h_total, L, fractions=LAYER_FRACTIONS)

    # Per-layer rheology
    n_consts = [Constant(e) for e in LAYER_EXPONENTS]
    A_layers = [Constant(a) for a in LAYER_PREFACTORS]

    Z = create_function_space(mesh, L)
    z = Function(Z)
    fields = split_fields(firedrake.split(z), L)
    ν = FacetNormal(mesh)

    # Flat surface for momentum (constant slope added separately)
    s_zero = Function(Q).assign(0.0)

    Lagrangian = Constant(0) * dx(mesh)

    for l in range(L):
        h_l = h_layers[l]
        fl = fields[l]
        u_l, M_l = fl["velocity"], fl["membrane_stress"]
        S_below = fl["interlayer_stress"]
        S_above = fields[l + 1]["interlayer_stress"] if l < L - 1 else None

        # Per-layer viscous power
        Lagrangian += viscous_power(
            membrane_stress=M_l, thickness=h_l,
            flow_law_coefficient=A_layers[l], flow_law_exponent=n_consts[l],
        )

        # Momentum balance
        ε_u = sym(grad(u_l))
        Lagrangian += -h_l * inner(M_l, ε_u) * dx
        Lagrangian += -inner(S_below, u_l) * dx
        if S_above is not None:
            Lagrangian += inner(S_above, u_l) * dx
        # Constant driving stress
        Lagrangian += ρ_I * g * h_l * Constant(tan_alpha) * u_l[0] * dx

    # Basal stress power (frozen base, uses bottom layer rheology)
    Lagrangian += basal_stress_power(
        basal_stress=fields[0]["interlayer_stress"],
        thickness=h_layers[0],
        flow_law_coefficient=A_layers[0],
        flow_law_exponent=n_consts[0],
    )

    # Interlayer stress power (uses bottom layer rheology at interface)
    for l in range(1, L):
        Lagrangian += interlayer_power(
            interlayer_stress=fields[l]["interlayer_stress"],
            thickness_above=h_layers[l], thickness_below=h_layers[l - 1],
            flow_law_coefficient=A_layers[l - 1],
            flow_law_exponent=n_consts[l - 1],
        )

    F = derivative(Lagrangian, z)
    bcs = [firedrake.DirichletBC(Z.sub(3 * l).sub(1), 0, (1, 2)) for l in range(L)]

    params = {"form_compiler_parameters": {"quadrature_degree": 8}}
    problem = NonlinearVariationalProblem(F, z, bcs, **params)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparams)

    # Continuation: ramp n linearly, A in log-space from 100 to target
    A_start = 100.0
    lambdas = np.concatenate([np.linspace(0, 0.8, 8), np.linspace(0.8, 1.0, 8)])
    for λ in lambdas:
        for l in range(L):
            n_consts[l].assign((1 - λ) + λ * LAYER_EXPONENTS[l])
            log_A = (1 - λ) * np.log(A_start) + λ * np.log(LAYER_PREFACTORS[l])
            A_layers[l].assign(np.exp(log_A))
        solver.solve()

    # Extract per-layer velocities along centreline
    subfunctions = z.subfunctions
    n_sample = 101
    x_hat = np.linspace(0, 1, n_sample)
    y_mid = Lx / 2

    u_bottom = np.array([subfunctions[0].at([xi * Lx, y_mid])[0] for xi in x_hat])
    u_top = np.array([subfunctions[3].at([xi * Lx, y_mid])[0] for xi in x_hat])

    return u_top, u_bottom, x_hat, z


def main():
    # Also import uniform-n solver for comparison
    from ismip_hom_b import solve_ismip_hom_b

    wavelengths = [160e3, 80e3, 40e3, 20e3, 10e3]

    print("ISMIP-HOM B: composite rheology (n=4 bottom, n=2 top) vs uniform n=3")
    print("=" * 75)
    print(f"{'L (km)':>8s}  {'composite top':>14s}  {'composite bot':>14s}  "
          f"{'uniform L=1':>12s}  {'uniform L=2':>12s}  {'uniform L=3':>12s}")
    print("-" * 75)

    for Lx in wavelengths:
        nx = min(max(20, int(40 * 80e3 / Lx)), 80)

        # Composite solve
        u_top_c, u_bot_c, x_hat, _ = solve_ismip_hom_b_composite(Lx, nx=nx)

        # Uniform n=3 for comparison
        u_1, _, _ = solve_ismip_hom_b(Lx, 1, nx=nx)
        u_2, _, _ = solve_ismip_hom_b(Lx, 2, nx=nx)
        u_3, _, _ = solve_ismip_hom_b(Lx, 3, nx=nx)

        print(f"{Lx/1e3:8.0f}  {np.max(u_top_c):14.4f}  {np.max(u_bot_c):14.4f}  "
              f"{np.max(u_1):12.4f}  {np.max(u_2):12.4f}  {np.max(u_3):12.4f}")

    # Detailed profile for L = 40 km
    Lx = 40e3
    print(f"\nDetailed profile for L = {Lx/1e3:.0f} km:")
    u_top_c, u_bot_c, x_hat, _ = solve_ismip_hom_b_composite(Lx, nx=40)
    u_uniform, _, _ = solve_ismip_hom_b(Lx, 2, nx=40)

    print(f"{'x/L':>6s}  {'composite top':>14s}  {'composite bot':>14s}  {'uniform L=2':>12s}  {'shear (top-bot)':>16s}")
    for i in range(0, len(x_hat), 10):
        shear = u_top_c[i] - u_bot_c[i]
        print(f"{x_hat[i]:6.2f}  {u_top_c[i]:14.4f}  {u_bot_c[i]:14.4f}  "
              f"{u_uniform[i]:12.4f}  {shear:16.4f}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(wavelengths), figsize=(18, 4), sharey=False)
    fig.suptitle("ISMIP-HOM B: composite (n=4/2) vs uniform (n=3)", fontsize=14)

    for ax, Lx in zip(axes, wavelengths):
        nx = min(max(20, int(40 * 80e3 / Lx)), 80)
        u_top_c, u_bot_c, x_hat, _ = solve_ismip_hom_b_composite(Lx, nx=nx)
        u_1, _, _ = solve_ismip_hom_b(Lx, 1, nx=nx)
        u_2, _, _ = solve_ismip_hom_b(Lx, 2, nx=nx)

        ax.plot(x_hat, u_top_c, "C0-", lw=2, label="composite top (n=4/2)")
        ax.plot(x_hat, u_bot_c, "C0--", lw=1.5, label="composite bot")
        ax.plot(x_hat, u_1, "C2-", lw=1, alpha=0.7, label="uniform L=1 (n=3)")
        ax.plot(x_hat, u_2, "C1-", lw=1, alpha=0.7, label="uniform L=2 (n=3)")
        ax.set_xlabel("$\\hat{x}$")
        ax.set_title(f"L = {Lx/1e3:.0f} km")
        if ax is axes[0]:
            ax.set_ylabel("$v_x$ (m/yr)")

    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig("ismip_hom_b_composite.png", dpi=150)
    print("\nSaved ismip_hom_b_composite.png")


if __name__ == "__main__":
    main()

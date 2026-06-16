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

r"""MISMIP+ geometry with a two-layer, n=4 rheology.

This sets up the MISMIP+ marine-ice-sheet benchmark geometry (Asay-Davis
et al. 2016) with the multilayer dual model and a composite, depth-varying
rheology:

  - Layer 1 (bottom, 15% of H):  temperate ice, n = 4, A = 46 MPa^-4 yr^-1
        (Goldsby & Kohlstedt 2001 dislocation creep at the pressure-melting
        point -- the softest, most deformable ice, sitting on the bed).
  - Layer 2 (top, 85% of H):     n = 1.8, A = 0.24 MPa^-1.8 yr^-1
        (grain-boundary sliding at -20 C: the Cuffey & Paterson 2010 n=3
        rate factor at -20 C, A = 1.2e-25 Pa^-3 s^-1 = 3.79 MPa^-3 yr^-1,
        stress-matched to n = 1.8 at tau_ref = 0.1 MPa).

This composite (n = 4 base / n = 1.8 top) follows the same depth-varying
rheology as ``ismip_hom_b_composite.py``; the soft temperate base
concentrates vertical shear near the bed.

Initialisation strategy
-----------------------
The two-layer solve is warm-started from a *depth-averaged rheology*
simulation: a single-layer (SSA-like) diagnostic solve with one effective
Glen's law for the whole column.  n = 3 is the standard effective average
of n = 4 dislocation creep and n = 1.8 grain-boundary sliding (Goldsby &
Kohlstedt 2001), so the depth-averaged run is an ordinary n = 3 solve.  Its
velocity initialises both layers, after which a continuation relaxes the
per-layer exponents from n = 3 to the n = 4 / n = 1.8 targets and splits the
rate factor accordingly -- mirroring ``ismip_hom_b_composite.py``.

Marine boundary conditions
--------------------------
  - Bed:        MISMIP+ polynomial + Gaussian side-channel, floored at -720 m.
  - Grounding:  flotation-based; basal drag uses the regularised-Coulomb
                (Schoof) law with beta^2 = C * N, so drag -> 0 as the ice
                approaches flotation (N -> 0) at the grounding line.
  - Terminus:   ocean back-pressure (calving_terminus) at x = Lx.
  - Divide:     symmetry, u_x = 0 at x = 0.
  - Side walls: free slip, u_y = 0 at y = 0, Ly.

Note
----
The thickness here is a prescribed *initial* geometry (a smooth grounded
profile thinning to a floating shelf), adequate for a diagnostic velocity
solve.  The true MISMIP+ steady state is obtained by evolving the thickness with
the depth-averaged velocity (icepack2.model.mass_balance); see the
prognostic spin-up in mismip_plus_spinup.py.
"""
import numpy as np
import firedrake
from firedrake import (
    Constant,
    Function,
    SpatialCoordinate,
    RectangleMesh,
    exp,
    dx,
    ds,
    max_value,
    as_vector,
    derivative,
    DirichletBC,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
)
from icepack2.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
)
from multilayer.model import minimization, utilities


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------
Lx, Ly = 640e3, 80e3                 # MISMIP+ domain (m)

# Boundary ids for firedrake.RectangleMesh:
#   1 = x = 0   (ice divide, symmetry)
#   2 = x = Lx  (calving front / terminus)
#   3 = y = 0   (side wall)
#   4 = y = Ly  (side wall)
DIVIDE_ID = 1
TERMINUS_ID = 2
WALL_IDS = (3, 4)

# ---------------------------------------------------------------------------
# Composite rheology: n=4 temperate base + n=1.8 grain-boundary-sliding top
# ---------------------------------------------------------------------------
LAYER_FRACTIONS = [0.15, 0.85]       # bottom, top
LAYER_EXPONENTS = [4.0, 1.8]         # base: dislocation creep; top: GBS
# base: temperate n=4 (Goldsby & Kohlstedt).
# top:  Cuffey & Paterson 2010 n=3 rate factor at -20 C
#       (1.2e-25 Pa^-3 s^-1 = 3.79 MPa^-3 yr^-1), stress-matched to n=1.8 at
#       tau_ref = 0.1 MPa:  3.79 * 0.1**(3 - 1.8) = 0.24 MPa^-1.8 yr^-1.
LAYER_PREFACTORS = [46.0, 0.24]      # MPa^-n yr^-1
NUM_LAYERS = 2

# Depth-averaged ("effective Glen") rheology used to warm-start the two-layer
# solve.  n = 3 is the standard effective average of n = 4 dislocation creep
# and n = 1.8 grain-boundary sliding; A = 100 MPa^-3 yr^-1 is the n = 1.8 top
# stress-matched to n = 3 at tau_ref = 0.1 MPa.
N_AVG = 3.0
A_AVG = 100.0

# ---------------------------------------------------------------------------
# Basal sliding (regularised Coulomb / Schoof): beta^2 = C * N
# ---------------------------------------------------------------------------
COULOMB_C = 0.5                      # Coulomb coefficient (tan of bed friction angle)
U_0 = 300.0                          # m/yr, Weertman -> Coulomb transition speed
SLIDING_M = 3.0                      # sliding exponent

# Flotation density ratio
R_FLOT = float(ρ_I / ρ_W)

sparams = {
    "snes_type": "newtonls",
    "snes_max_it": 200,
    "snes_linesearch_type": "bt",
    "snes_divergence_tolerance": -1,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
fcp = {"form_compiler_parameters": {"quadrature_degree": 6}}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def mismip_bed(mesh):
    r"""MISMIP+ bed topography (Asay-Davis et al. 2016, Eqs. 1-4)."""
    x, y = SpatialCoordinate(mesh)

    x_c = Constant(300e3)
    X = x / x_c
    B_0, B_2, B_4, B_6 = -150.0, -728.8, 343.91, -50.57
    B_x = B_0 + B_2 * X ** 2 + B_4 * X ** 4 + B_6 * X ** 6

    f_c, d_c, w_c = 4e3, 500.0, 24e3
    B_y = d_c * (
        1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c))
        + 1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
    )

    z_deep = -720.0
    return max_value(B_x + B_y, Constant(z_deep))


def initial_geometry(mesh, Q):
    r"""Return (bed, thickness, surface, effective_pressure) Functions.

    The thickness is a smooth grounded profile that thins seaward into a
    floating shelf -- a reasonable starting geometry for the diagnostic
    momentum solve.  Surface and effective pressure are derived from
    flotation.
    """
    x, y = SpatialCoordinate(mesh)

    bed = Function(Q, name="bed").interpolate(mismip_bed(mesh))

    # Smooth grounded-sheet profile: thick at the divide, thinning seaward.
    H_div, H_min = 2000.0, 50.0
    # max_value(0, .) guards against a tiny negative base at the x = Lx node.
    H_expr = max_value(
        Constant(H_min),
        Constant(H_div) * max_value(Constant(0.0), 1 - x / Lx) ** 0.5,
    )
    thickness = Function(Q, name="thickness").interpolate(H_expr)

    # Surface from flotation: grounded -> bed + H; floating -> (1 - r) H.
    surface = Function(Q, name="surface").interpolate(
        max_value(bed + thickness, (1 - Constant(R_FLOT)) * thickness)
    )

    # Effective pressure N = rho_I g (H - H_f), H_f = max(0, -bed) rho_W/rho_I,
    # clamped at 0 (zero on floating ice -> no basal drag there).
    H_f = max_value(Constant(0.0), -bed) * Constant(ρ_W / ρ_I)
    N = Function(Q, name="effective_pressure").interpolate(
        max_value(ρ_I * g * (thickness - H_f), Constant(0.0))
    )

    return bed, thickness, surface, N


# ---------------------------------------------------------------------------
# Depth-averaged warm start (single effective rheology)
# ---------------------------------------------------------------------------
def solve_depth_averaged(mesh, Q, thickness, surface, N):
    r"""Single-layer diagnostic solve with one depth-averaged rate factor.

    Returns the velocity Function used to warm-start the two-layer model.
    """
    Z = utilities.create_function_space(mesh, 1)
    z = Function(Z)
    # Sensible non-zero initial guess to keep the n-continuation well posed.
    z.sub(0).interpolate(as_vector([Constant(10.0), Constant(0.0)]))
    fields = utilities.split_fields(firedrake.split(z), 1)
    f0 = fields[0]

    n = Constant(1.0)
    A = Constant(A_AVG)
    β2 = COULOMB_C * N

    Lagrangian = (
        minimization.viscous_power(
            membrane_stress=f0["membrane_stress"], thickness=thickness,
            flow_law_coefficient=A, flow_law_exponent=n,
        )
        + minimization.momentum_balance(
            velocity=f0["velocity"], membrane_stress=f0["membrane_stress"],
            thickness=thickness, surface=surface,
            basal_stress=f0["interlayer_stress"],
        )
        + minimization.schoof_friction_power(
            basal_stress=f0["interlayer_stress"],
            friction_coefficient=β2,
            transition_speed=Constant(U_0),
            sliding_exponent=Constant(SLIDING_M),
        )
        + minimization.calving_terminus(
            velocity=f0["velocity"], thickness=thickness, surface=surface,
            outflow_ids=(TERMINUS_ID,), layer_fraction=Constant(1.0),
        )
    )

    F = derivative(Lagrangian, z)
    bcs = [
        DirichletBC(Z.sub(0).sub(0), 0, (DIVIDE_ID,)),
        DirichletBC(Z.sub(0).sub(1), 0, WALL_IDS),
    ]
    problem = NonlinearVariationalProblem(F, z, bcs, **fcp)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparams)

    # Continuation on the stress exponent 1 -> 3 (effective Glen's law).
    for λ in np.linspace(0.0, 1.0, 9):
        n.assign((1 - λ) + λ * N_AVG)
        solver.solve()

    return z.subfunctions[0].copy(deepcopy=True)


# ---------------------------------------------------------------------------
# Two-layer solve (n=4, temperate base + cold top)
# ---------------------------------------------------------------------------
def build_two_layer(mesh, Q, thickness, surface, N):
    r"""Build the two-layer momentum problem around (mutable) geometry fields.

    ``thickness``, ``surface`` and ``N`` are Functions; the returned solver
    can be re-solved after they are updated in place (used by the prognostic
    spin-up).  Returns a state dict with the mixed function ``z``, the
    ``solver``, the per-layer ``fields``/``h_layers``, and the rheology
    ``Constant``s used for continuation.
    """
    L = NUM_LAYERS
    h_layers = utilities.layer_thicknesses(thickness, L, fractions=LAYER_FRACTIONS)

    # Per-layer rheology, initialised at the effective n=3 / A=100 state from
    # the depth-averaged warm start; relaxed to the composite targets below.
    n_consts = [Constant(N_AVG) for _ in range(L)]
    A_consts = [Constant(A_AVG) for _ in range(L)]

    Z = utilities.create_function_space(mesh, L)
    z = Function(Z)
    fields = utilities.split_fields(firedrake.split(z), L)

    β2 = COULOMB_C * N

    Lagrangian = Constant(0) * dx(mesh)
    for l in range(L):
        h_l = h_layers[l]
        fl = fields[l]
        S_above = fields[l + 1]["interlayer_stress"] if l < L - 1 else None

        Lagrangian += minimization.viscous_power(
            membrane_stress=fl["membrane_stress"], thickness=h_l,
            flow_law_coefficient=A_consts[l], flow_law_exponent=n_consts[l],
        )
        Lagrangian += minimization.momentum_balance(
            velocity=fl["velocity"], membrane_stress=fl["membrane_stress"],
            thickness=h_l, surface=surface,
            basal_stress=fl["interlayer_stress"] if l == 0 else None,
            stress_above=S_above,
            stress_below=fl["interlayer_stress"] if l > 0 else None,
        )
        # Ocean back-pressure at the terminus, weighted by this layer's share.
        Lagrangian += minimization.calving_terminus(
            velocity=fl["velocity"], thickness=thickness, surface=surface,
            outflow_ids=(TERMINUS_ID,), layer_fraction=h_l / thickness,
        )

    # Basal drag (regularised Coulomb) on the bottom layer's basal stress.
    Lagrangian += minimization.schoof_friction_power(
        basal_stress=fields[0]["interlayer_stress"],
        friction_coefficient=β2,
        transition_speed=Constant(U_0),
        sliding_exponent=Constant(SLIDING_M),
    )

    # Interlayer shear power at the internal interface (base-layer rheology).
    for l in range(1, L):
        Lagrangian += minimization.interlayer_power(
            interlayer_stress=fields[l]["interlayer_stress"],
            thickness_above=h_layers[l], thickness_below=h_layers[l - 1],
            flow_law_coefficient=A_consts[l - 1], flow_law_exponent=n_consts[l - 1],
        )

    F = derivative(Lagrangian, z)
    bcs = []
    for l in range(L):
        bcs.append(DirichletBC(Z.sub(3 * l).sub(0), 0, (DIVIDE_ID,)))
        bcs.append(DirichletBC(Z.sub(3 * l).sub(1), 0, WALL_IDS))

    problem = NonlinearVariationalProblem(F, z, bcs, **fcp)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sparams)

    return {
        "z": z, "solver": solver, "fields": fields, "h_layers": h_layers,
        "n_consts": n_consts, "A_consts": A_consts,
    }


def continue_rheology(state):
    r"""Relax the per-layer (n, A) from the effective n=3/A=100 warm state to
    the composite targets (n=4/A=46 base, n=1.8/A=0.24 top), ramping A in
    log-space.  Mirrors the ``ismip_hom_b_composite.py`` continuation."""
    n_consts, A_consts, solver = state["n_consts"], state["A_consts"], state["solver"]
    lambdas = np.concatenate([
        np.linspace(0.0, 0.7, 8),
        np.linspace(0.7, 0.92, 8)[1:],
        np.linspace(0.92, 1.0, 7)[1:],
    ])
    for λ in lambdas:
        for l in range(NUM_LAYERS):
            n_consts[l].assign((1 - λ) * N_AVG + λ * LAYER_EXPONENTS[l])
            log_A = (1 - λ) * np.log(A_AVG) + λ * np.log(LAYER_PREFACTORS[l])
            A_consts[l].assign(np.exp(log_A))
        solver.solve()


def solve_two_layer(mesh, Q, thickness, surface, N, warm_u):
    r"""Two-layer diagnostic momentum solve on the MISMIP+ geometry."""
    state = build_two_layer(mesh, Q, thickness, surface, N)
    for l in range(NUM_LAYERS):
        state["z"].sub(3 * l).interpolate(warm_u)
    continue_rheology(state)
    return state["z"]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def grounding_line_x(mesh, Q, thickness, bed, n_sample=321):
    r"""Return the centreline grounding-line position (m), or None."""
    haf = Function(Q).interpolate(
        thickness - max_value(Constant(0.0), -bed) * Constant(ρ_W / ρ_I)
    )
    xs = np.linspace(0.0, Lx, n_sample)
    vals = np.array([haf.at([xi, Ly / 2]) for xi in xs])
    grounded = vals > 0.0
    flip = np.where(grounded[:-1] & ~grounded[1:])[0]
    return float(xs[flip[0]]) if len(flip) else None


def centreline_speeds(z, n_sample=321):
    u_bot = z.subfunctions[0]
    u_top = z.subfunctions[3 * (NUM_LAYERS - 1)]
    xs = np.linspace(0.0, Lx, n_sample)
    s_bot = np.array([u_bot.at([xi, Ly / 2])[0] for xi in xs])
    s_top = np.array([u_top.at([xi, Ly / 2])[0] for xi in xs])
    return xs, s_top, s_bot


def main(nx=128, ny=16):
    print("MISMIP+ : two-layer composite (n=4 temperate base / n=1.8 GBS top)")
    print("=" * 68)
    print(f"  domain        : {Lx/1e3:.0f} x {Ly/1e3:.0f} km, mesh {nx} x {ny}")
    print(f"  base  ({LAYER_FRACTIONS[0]*100:.0f}%)   : n={LAYER_EXPONENTS[0]}, "
          f"A={LAYER_PREFACTORS[0]} (temperate)")
    print(f"  top   ({LAYER_FRACTIONS[1]*100:.0f}%)   : n={LAYER_EXPONENTS[1]}, "
          f"A={LAYER_PREFACTORS[1]} (grain-boundary sliding)")
    print(f"  warm-start    : effective Glen n={N_AVG:.0f}, A={A_AVG:.0f}")

    mesh = RectangleMesh(nx, ny, Lx, Ly)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)

    bed, thickness, surface, N = initial_geometry(mesh, Q)

    x_gl = grounding_line_x(mesh, Q, thickness, bed)
    print(f"  grounding line: x = {x_gl/1e3:.1f} km" if x_gl else "  grounding line: (none)")

    print("\n  depth-averaged warm start ...", flush=True)
    warm_u = solve_depth_averaged(mesh, Q, thickness, surface, N)
    print(f"    max depth-avg speed: {np.max(np.abs(warm_u.dat.data_ro[:, 0])):.2f} m/yr")

    print("  two-layer solve ...", flush=True)
    z = solve_two_layer(mesh, Q, thickness, surface, N, warm_u)

    xs, s_top, s_bot = centreline_speeds(z)
    shear = s_top - s_bot
    print(f"\n    max surface (top)  speed: {np.max(s_top):.2f} m/yr")
    print(f"    max basal  (bot)   speed: {np.max(s_bot):.2f} m/yr")
    print(f"    max vertical shear (top-bot): {np.max(shear):.2f} m/yr")

    print(f"\n  {'x (km)':>8s}  {'u_top':>10s}  {'u_bot':>10s}  {'shear':>10s}")
    for i in range(0, len(xs), len(xs) // 12):
        print(f"  {xs[i]/1e3:8.1f}  {s_top[i]:10.3f}  {s_bot[i]:10.3f}  {shear[i]:10.3f}")

    # Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs / 1e3, s_top, "C0-", lw=2, label="surface (top, n=1.8 GBS)")
        ax.plot(xs / 1e3, s_bot, "C1-", lw=2, label="base (bottom, n=4 temperate)")
        if x_gl:
            ax.axvline(x_gl / 1e3, color="0.5", ls="--", lw=1, label="grounding line")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("along-flow speed (m/yr)")
        ax.set_title("MISMIP+ centreline speeds: composite n=4 base / n=1.8 top")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("figures/mismip_plus.png", dpi=150)
        print("\n  saved figures/mismip_plus.png")
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"\n  (skipped figure: {e})")


if __name__ == "__main__":
    main()

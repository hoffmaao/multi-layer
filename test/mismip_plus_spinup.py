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

r"""Prognostic MISMIP+ spin-up / response with the two-layer composite model.

This drives the diagnostic two-layer momentum balance from ``mismip_plus.py``
forward in time, evolving the thickness with the depth-integrated continuity
equation.  The scheme is *staggered* (operator-split):

  for each step:
    1. recompute surface s and effective pressure N from the current H
    2. solve the two-layer momentum balance (warm-started from the previous
       step) -> per-layer velocities
    3. form the thickness-weighted depth-averaged velocity
       ubar = (h1 u1 + h2 u2) / H
    4. advance H one step of the continuity equation
         dH/dt + div(H ubar) = a - m
       using icepack2.model.mass_balance (DG-upwind) + an irksome
       variational-inequality solve that keeps H >= 0.

The continuity step follows the icepack2 transport pattern
(``mass_balance_test.py`` / ``dome_test.py``): a DG(1) Bernstein thickness
with a ``vinewtonrsls`` bound H >= 0.  The thickness is projected onto CG(1)
for the momentum solve so the velocity discretisation matches the diagnostic.

A MISMIP+ "Ice1r" sub-shelf melt parameterisation (Asay-Davis et al. 2016)
can be switched on after a spin-up interval to drive grounding-line retreat
-- this is the "response" of the ice sheet.

Notes
-----
The time step ``DT``, the spin-up / response durations, and the transport
polynomial degree may need tuning for a given mesh.  Velocities near the
grounding line can be large, so ``DT`` is the main stability control.
"""
import numpy as np
import firedrake
from firedrake import (
    Constant,
    Function,
    FunctionSpace,
    FiniteElement,
    BrokenElement,
    RectangleMesh,
    conditional,
    lt,
    tanh,
    max_value,
    dx,
)
import irksome
from icepack2.model import mass_balance
from icepack2.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
)
from multilayer.model import utilities

import mismip_plus as mp


# ---------------------------------------------------------------------------
# Forcing
# ---------------------------------------------------------------------------
ACCUM = 0.3                          # m/yr surface mass balance (MISMIP+)

# MISMIP+ "Ice1r" sub-shelf melt parameterisation (Asay-Davis et al. 2016):
#   m = OMEGA * tanh(H_c / Z0) * max(Z_THRESH - z_d, 0)
# with H_c the water-column thickness and z_d the ice-shelf draft.
OMEGA = 0.2                          # 1/yr
Z0 = 75.0                            # m
Z_THRESH = -100.0                    # m

# Time stepping
DT = 2.0                             # yr (stability knob)


# ---------------------------------------------------------------------------
# Geometry helpers (consistent with mismip_plus.initial_geometry)
# ---------------------------------------------------------------------------
def update_surface(surface, bed, H):
    r"""s = max(bed + H, (1 - r) H) -- flotation-aware, updated in place."""
    surface.interpolate(
        max_value(bed + H, (1 - Constant(mp.R_FLOT)) * H)
    )


def update_effective_pressure(N, bed, H):
    r"""N = max(0, rho_I g (H - H_f)), updated in place (zero on floating ice)."""
    H_f = max_value(Constant(0.0), -bed) * Constant(ρ_W / ρ_I)
    N.interpolate(max_value(ρ_I * g * (H - H_f), Constant(0.0)))


def mass_balance_expr(bed, H, melt_factor):
    r"""Net mass balance a - m: accumulation minus Ice1r sub-shelf melt.

    Melt acts only on floating ice (H < flotation thickness).
    """
    H_f = max_value(Constant(0.0), -bed) * Constant(ρ_W / ρ_I)
    floating = conditional(lt(H, H_f), Constant(1.0), Constant(0.0))

    z_d = -Constant(mp.R_FLOT) * H                    # ice-shelf draft (<= 0)
    H_c = max_value(Constant(0.0), z_d - bed)         # water-column thickness
    melt = (
        Constant(OMEGA) * tanh(H_c / Constant(Z0))
        * max_value(Constant(Z_THRESH) - z_d, Constant(0.0))
    )
    return Constant(ACCUM) - Constant(melt_factor) * floating * melt


# ---------------------------------------------------------------------------
# Spin-up / response driver
# ---------------------------------------------------------------------------
def run(nx=128, ny=16, spinup_years=400.0, response_years=200.0, dt=DT):
    print("MISMIP+ prognostic spin-up : two-layer composite (n=4 base / n=1.8 top)")
    print("=" * 72)
    print(f"  mesh {nx} x {ny}, dt = {dt} yr, "
          f"spin-up {spinup_years:.0f} yr -> response {response_years:.0f} yr")

    mesh = RectangleMesh(nx, ny, mp.Lx, mp.Ly)
    Q = FunctionSpace(mesh, "CG", 1)

    # Initial geometry (CG1 fields shared with the momentum solve).
    bed, H_cg, surface, N = mp.initial_geometry(mesh, Q)

    # Build the momentum solver around these mutable geometry fields, then
    # warm-start + run the rheology continuation once.
    state = mp.build_two_layer(mesh, Q, H_cg, surface, N)
    warm_u = mp.solve_depth_averaged(mesh, Q, H_cg, surface, N)
    for l in range(mp.NUM_LAYERS):
        state["z"].sub(3 * l).interpolate(warm_u)
    print("  rheology continuation ...", flush=True)
    mp.continue_rheology(state)

    V = state["z"].subfunctions[0].function_space()
    ubar = Function(V, name="depth_averaged_velocity")

    # --- transport (continuity) setup: DG(1) Bernstein thickness, H >= 0 ---
    dg1 = BrokenElement(FiniteElement("Bernstein", "triangle", 1))
    Q_dg = FunctionSpace(mesh, dg1)
    H_dg = Function(Q_dg, name="thickness").project(H_cg)
    a_dg = Function(Q_dg, name="mass_balance")

    F_mass = mass_balance(thickness=H_dg, velocity=ubar, accumulation=a_dg)

    t = Constant(0.0)
    dt_c = Constant(dt)
    tableau = irksome.BackwardEuler()

    lower = Function(Q_dg).assign(0.0)
    upper = Function(Q_dg).assign(1.0e4)
    bounds = ("stage", lower, upper)
    bparams = {
        "solver_parameters": {
            "snes_type": "vinewtonrsls",
            "snes_max_it": 100,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        "stage_type": "value",
        "basis_type": "Bernstein",
        "bounds": bounds,
    }
    stepper = irksome.TimeStepper(F_mass, tableau, t, dt_c, H_dg, **bparams)

    total_years = spinup_years + response_years
    num_steps = int(round(total_years / dt))
    report_every = max(1, num_steps // 20)

    print(f"\n  {'t (yr)':>8s}  {'x_gl (km)':>10s}  {'max u (m/yr)':>12s}  "
          f"{'V (1e3 km^3)':>12s}  {'phase':>8s}")
    history = []
    for step in range(num_steps + 1):
        time = step * dt
        phase_melt = 1.0 if time >= spinup_years else 0.0

        # 1. geometry from current thickness
        H_cg.project(H_dg)
        update_surface(surface, bed, H_cg)
        update_effective_pressure(N, bed, H_cg)

        # 2. momentum solve (warm-started from previous step)
        state["solver"].solve()

        # 3. depth-averaged velocity
        ubar.assign(utilities.depth_averaged_velocity(
            state["z"], state["h_layers"], H_cg, V
        ))

        # 4. net mass balance (melt only after spin-up)
        a_dg.interpolate(mass_balance_expr(bed, H_dg, melt_factor=phase_melt))

        if step % report_every == 0 or step == num_steps:
            x_gl = mp.grounding_line_x(mesh, Q, H_cg, bed)
            u_max = float(np.max(np.abs(ubar.dat.data_ro[:, 0])))
            vol = firedrake.assemble(H_cg * dx) / 1e12  # 1e3 km^3
            gl_km = x_gl / 1e3 if x_gl else float("nan")
            tag = "response" if phase_melt else "spin-up"
            print(f"  {time:8.1f}  {gl_km:10.1f}  {u_max:12.1f}  {vol:12.3f}  {tag:>8s}",
                  flush=True)
            history.append((time, gl_km, u_max, vol))

        # 5. advance thickness one step
        if step < num_steps:
            stepper.advance()
            t.assign(float(t) + dt)

    _plot(history, spinup_years)
    return history


def _plot(history, spinup_years):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = np.array(history)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(h[:, 0], h[:, 1], "C0-o", ms=3, label="grounding line")
        ax.axvline(spinup_years, color="0.5", ls="--", lw=1, label="melt on")
        ax.set_xlabel("time (yr)")
        ax.set_ylabel("centreline grounding line x (km)")
        ax.set_title("MISMIP+ two-layer composite: GL spin-up + Ice1r response")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("figures/mismip_plus_spinup.png", dpi=150)
        print("\n  saved figures/mismip_plus_spinup.png")
    except Exception as e:  # pragma: no cover - plotting optional
        print(f"\n  (skipped figure: {e})")


if __name__ == "__main__":
    run()

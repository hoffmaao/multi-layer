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

r"""Action functional (minimization form) for the multilayer model

Each function returns a UFL Form contributing to the Lagrangian whose
Gateaux derivative recovers the variational-form equations.  Functions
that are identical to their icepack2 counterparts are re-exported;
multilayer-specific terms (interlayer power, basal stress power, and the
extended momentum balance) are new.
"""

import ufl
from firedrake import (
    eq,
    conditional,
    Constant,
    inner,
    tr,
    sym,
    grad,
    dx,
    ds,
    dS,
    avg,
    jump,
    FacetNormal,
    min_value,
)
from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Re-export from icepack2 -- identical for per-layer use
from icepack2.model.minimization import viscous_power  # noqa: F401
from icepack2.model.minimization import friction_power  # noqa: F401


def interlayer_power(**kwargs):
    r"""Return the interlayer shear dissipation potential

    .. math::
        \int (h^{l+1} + h^l)\,\frac{A}{n+1}\,\lvert S^l \rvert^{n+1}\,dx

    The factor :math:`h^{l+1} + h^l` arises from the normalisation of the
    velocity jump in the interlayer constitutive law.
    """
    S = kwargs["interlayer_stress"]
    h_above = kwargs["thickness_above"]
    h_below = kwargs["thickness_below"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    S_2 = inner(S, S)
    S_n = conditional(eq(n, 1), S_2, S_2 ** ((n + 1) / 2))
    return (h_above + h_below) * A / (n + 1) * S_n * dx


def basal_stress_power(**kwargs):
    r"""Return the basal shear dissipation potential for a frozen base

    .. math::
        \int h^1\,\frac{A}{n+1}\,\lvert \tau \rvert^{n+1}\,dx
    """
    τ = kwargs["basal_stress"]
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    τ_2 = inner(τ, τ)
    τ_n = conditional(eq(n, 1), τ_2, τ_2 ** ((n + 1) / 2))
    return h * A / (n + 1) * τ_n * dx


def momentum_balance(**kwargs):
    r"""Return the per-layer momentum balance constraint

    In the minimization form the velocity :math:`u^l` acts as a Lagrange
    multiplier enforcing force balance:

    .. math::
        -h^l M^l : \varepsilon(u^l) + \tau \cdot u^l
        + S_{\mathrm{above}} \cdot u^l - S_{\mathrm{below}} \cdot u^l
        - \rho_I g\, h^l \nabla s \cdot u^l

    Parameters
    ----------
    velocity, membrane_stress, thickness, surface : per-layer fields
    basal_stress : icepack2-convention basal drag (bottom layer only)
    stress_above : interlayer stress from the layer above, or ``None``
    stress_below : interlayer stress from the layer below, or ``None``
    """
    u = kwargs["velocity"]
    M = kwargs["membrane_stress"]
    h = kwargs["thickness"]
    s = kwargs["surface"]

    τ = kwargs.get("basal_stress")
    S_above = kwargs.get("stress_above")
    S_below = kwargs.get("stress_below")

    ε = sym(grad(u))
    F = (-h * inner(M, ε) - ρ_I * g * h * inner(grad(s), u)) * dx

    if τ is not None:
        F += inner(τ, u) * dx

    if S_above is not None:
        F += inner(S_above, u) * dx
    if S_below is not None:
        F -= inner(S_below, u) * dx

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)
    F += ρ_I * g * avg(h) * inner(jump(s, ν), avg(u)) * dS

    return F


def schoof_friction_power(**kwargs):
    r"""Friction power for the regularized Coulomb (Schoof/RCF) law.

    This is the dual (minimization) form whose derivative with respect to
    :math:`\tau` recovers the Schoof friction constitutive relation.

    The RCF law :math:`|\tau| = \beta^2 (|u|/(|u|+u_0))^{1/m}` inverts to
    :math:`|u| = u_0 (|\tau|/\beta^2)^m / (1 - (|\tau|/\beta^2)^m)`,
    which has the same structure as Weertman with a stress-dependent
    compliance :math:`K_{\rm eff} = u_0 / (\beta^{2m} - |\tau|^m)`.

    The friction power is:

    .. math::
        P = \int u_0 \int_0^{|\tau|} \frac{t^m}{\beta^{2m} - t^m}\,dt\,dx

    For :math:`m = 3`, the integral has a closed form involving logarithms
    and arctangent.

    Parameters
    ----------
    basal_stress : UFL split variable
    friction_coefficient : Function
        :math:`\beta^2` in MPa.
    transition_speed : Constant
        :math:`u_0` in m/yr.
    sliding_exponent : Constant
        m (= 3 for Glen's law).
    """
    τ = kwargs["basal_stress"]
    β2 = kwargs["friction_coefficient"]
    u_0 = kwargs["transition_speed"]
    m = kwargs["sliding_exponent"]

    τ_2 = inner(τ, τ)
    τ_mag = ufl.sqrt(τ_2 + Constant(1e-20))

    # r = |τ| / (β² + ε), smooth regularization (no conditional)
    eps = Constant(1e-4)
    r = τ_mag / (β2 + eps)

    # For m=3: ∫₀^r s³/(1-s³) ds  (closed form)
    # = -r + (1/6)ln((1+r+r²)/(1-r)²) + (1/√3)(atan((2r+1)/√3) - π/6)
    # Regularize (1-r) → (1-r+δ) to keep ln finite near Coulomb limit
    sqrt3 = Constant(1.7320508075688772)
    pi_over_6 = Constant(0.5235987755982988)
    delta = Constant(1e-6)
    one_minus_r = Constant(1.0) - r + delta

    log_term = (Constant(1.0) / Constant(6.0)) * ufl.ln(
        (Constant(1.0) + r + r**2 + delta) / (one_minus_r**2)
    )
    atan_term = (Constant(1.0) / sqrt3) * (
        ufl.atan2(Constant(2.0) * r + Constant(1.0), sqrt3)
        - pi_over_6
    )

    integral = -r + log_term + atan_term

    return u_0 * (β2 + eps) * integral * dx


def calving_terminus(**kwargs):
    r"""Return the ocean back-pressure at the terminus for one layer"""
    u, h, s = map(kwargs.get, ("velocity", "thickness", "surface"))
    outflow_ids = kwargs["outflow_ids"]
    layer_fraction = kwargs.get("layer_fraction", Constant(1.0))

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)

    f_I = 0.5 * ρ_I * g * h ** 2
    d = min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d ** 2

    return layer_fraction * (f_I - f_W) * inner(u, ν) * ds(outflow_ids)

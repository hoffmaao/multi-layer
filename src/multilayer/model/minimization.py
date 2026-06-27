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


def composite_viscous_power(**kwargs):
    r"""Composite (dislocation + linear) membrane viscous power.

    A composite flow law adds the strain rates of a nonlinear creep mechanism
    and a linear (:math:`n=1`) one; in the dual form their complementary
    energies add.  Following icepack2's ``dome_test`` and the ismip7 /
    peninsula composite rheology, the linear term is a small regulariser
    evaluated at a *constant* reference thickness :math:`H_{\rm ref}`:

    .. math::
        P = \underbrace{2 h\,\frac{A}{n+1}\,|M_{\rm dev}|^{n+1}}_{\text{creep, exponent }n}
          + \alpha\,\underbrace{2 H_{\rm ref}\,\frac{A_{\rm lin}}{2}\,|M_{\rm dev}|^{2}}_{\text{linear regulariser}}

    with :math:`A_{\rm lin} = A\,\tau_c^{\,n-1}`, so the two mechanisms agree at
    :math:`|M_{\rm dev}| = \tau_c`.  The :math:`n=1` term is positive-definite
    in :math:`M` for any :math:`A>0`, so it keeps the membrane-stress block of
    the Jacobian non-singular where the pure-creep Hessian vanishes: at zero
    stress (the first Newton step of an :math:`n>1` solve) and -- because it
    uses :math:`H_{\rm ref}` rather than :math:`h` -- where the ice thins to
    zero (calving fronts, nunataks).

    Parameters
    ----------
    membrane_stress, thickness, flow_law_coefficient, flow_law_exponent
        As in :func:`viscous_power`.
    regularization : Constant, optional
        :math:`\alpha`, weight of the linear term (default ``1e-4``).
    reference_thickness : Constant, optional
        :math:`H_{\rm ref}` for the linear term (default ``100`` m).
    reference_stress : Constant, optional
        :math:`\tau_c`, the mechanism cross-over stress (default ``0.1`` MPa).
    """
    M = kwargs["membrane_stress"]
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    α = kwargs.get("regularization", Constant(1e-4))
    H_ref = kwargs.get("reference_thickness", Constant(100.0))
    τ_c = kwargs.get("reference_stress", Constant(0.1))

    A_lin = A * τ_c ** (n - Constant(1.0))
    return (
        viscous_power(
            membrane_stress=M, thickness=h,
            flow_law_coefficient=A, flow_law_exponent=n,
        )
        + α * viscous_power(
            membrane_stress=M, thickness=H_ref,
            flow_law_coefficient=A_lin, flow_law_exponent=Constant(1.0),
        )
    )


def composite_interlayer_power(**kwargs):
    r"""Composite interlayer shear power (dislocation + linear regulariser).

    The :func:`interlayer_power` analogue of :func:`composite_viscous_power`.
    The interlayer (vertical-shear) Hessian also scales like :math:`|S|^{n-1}`
    and vanishes at :math:`S=0`, which makes the two-layer dual system singular
    on the first Newton step when the stresses are initialised to zero.  The
    linear (:math:`n=1`) term restores a positive-definite block.

    Parameters
    ----------
    interlayer_stress, thickness_above, thickness_below, flow_law_coefficient,
    flow_law_exponent
        As in :func:`interlayer_power`.
    regularization : Constant, optional
        :math:`\alpha` (default ``1e-4``).
    reference_stress : Constant, optional
        :math:`\tau_c` (default ``0.1`` MPa).
    """
    S = kwargs["interlayer_stress"]
    h_above = kwargs["thickness_above"]
    h_below = kwargs["thickness_below"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    α = kwargs.get("regularization", Constant(1e-4))
    τ_c = kwargs.get("reference_stress", Constant(0.1))

    A_lin = A * τ_c ** (n - Constant(1.0))
    return (
        interlayer_power(
            interlayer_stress=S, thickness_above=h_above, thickness_below=h_below,
            flow_law_coefficient=A, flow_law_exponent=n,
        )
        + α * interlayer_power(
            interlayer_stress=S, thickness_above=h_above, thickness_below=h_below,
            flow_law_coefficient=A_lin, flow_law_exponent=Constant(1.0),
        )
    )


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

    Dual (minimization) form whose derivative with respect to :math:`\tau`
    recovers a regularized-Coulomb sliding relation that is valid for *any*
    sliding exponent :math:`m` (the earlier implementation was hard-wired to
    :math:`m = 3`).  With :math:`r = |\tau| / \beta^2` the constitutive law is

    .. math::
        |u| = u_0\,\frac{r^{m}}{1 - r^{\,m+1}},

    a Weertman power law :math:`|u|\approx u_0\,r^{m}` at low driving stress
    that diverges as :math:`|\tau|\to\beta^2` -- the Coulomb yield limit.  Its
    complementary energy is elementary for every :math:`m`,

    .. math::
        P = \int \frac{u_0\,\beta^2}{m+1}\,\bigl[-\ln\!\bigl(1 - r^{\,m+1}\bigr)\bigr]\,dx,

    which mirrors the Weertman :func:`friction_power`
    :math:`\tfrac{K}{m+1}|\tau|^{m+1}` (its small-stress limit, with
    :math:`K = u_0/\beta^{2m}`) and reduces drag to zero at the grounding line
    as :math:`\beta^2 = C N \to 0`.

    Because the basal-stress Hessian scales like :math:`|\tau|^{m-1}`, it is
    *constant and non-zero at* :math:`\tau = 0` only when :math:`m = 1`.  Ramping
    the sliding exponent :math:`m: 1 \to m` alongside the flow-law exponent
    therefore keeps the dual system non-singular through a continuation -- the
    same trick icepack2's own tests use for Weertman drag.

    Parameters
    ----------
    basal_stress : UFL split variable
    friction_coefficient : Function
        :math:`\beta^2 = C N` in MPa (vanishes on floating ice).
    transition_speed : Constant
        :math:`u_0` in m/yr.
    sliding_exponent : Constant
        m (= 3 for Glen's law); may be a continuation parameter.
    """
    τ = kwargs["basal_stress"]
    β2 = kwargs["friction_coefficient"]
    u_0 = kwargs["transition_speed"]
    m = kwargs["sliding_exponent"]

    eps = Constant(1e-4)    # keep r finite where β² = C N → 0 (floating ice)
    delta = Constant(1e-6)  # keep the log finite at the Coulomb limit r → 1

    # r^(m+1) built from |τ|², mirroring the Weertman friction_power idiom.
    β2e = β2 + eps
    r_2 = (inner(τ, τ) + Constant(1e-20)) / (β2e * β2e)         # = r²
    r_mp1 = conditional(eq(m, 1), r_2, r_2 ** ((m + 1) / 2))    # = r^(m+1)

    return u_0 * β2e / (m + 1) * (-ufl.ln(Constant(1.0) - r_mp1 + delta)) * dx


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

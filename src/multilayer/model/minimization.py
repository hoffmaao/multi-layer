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
multilayer-specific terms are new.

The column action that converges to the first-order (Blatter-Pattyn)
model as layers are added is, for layers :math:`l = 1 \ldots L` from the
bed up,

.. code::

    sum_l layer_power(l)          membrane and shear in one flow law
    + sum_l momentum_balance(l)   with the driving stress
    + slope_coupling              layers that follow the bed
    + friction_power              sliding beds only

:func:`layer_power` splits each layer into two half-layers under its own
law, each with one effective stress from the membrane stress and the
shear on its face, so the interfaces are in series and the bottom
half-layer shears under the basal stress on a sliding bed as well as on a
frozen one.  :func:`slope_coupling` restores the strain rate at constant
height on layers that slope.  Against a first-order reference both
together converge at second order in the layer thickness (ISMIP-HOM B and
D, rheology changing with height).

The older terms -- :func:`viscous_power` per layer with
:func:`interlayer_power` and :func:`basal_stress_power` -- give each stress
its own flow law.  They agree with :func:`layer_power` for linear ice and
for a laterally uniform slab, and are kept for existing models, but for
:math:`n > 1` they make ice that both stretches and shears too stiff, and
with :func:`friction_power` alone they drop the shear of the bottom
half-layer.
"""

import ufl
from firedrake import (
    eq,
    conditional,
    Constant,
    inner,
    sym,
    grad,
    dx,
    ds,
    dS,
    avg,
    jump,
    FacetNormal,
    min_value,
    max_value,
    tr,
)
from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Re-export from icepack2 -- identical for per-layer use
from icepack2.model.minimization import viscous_power  # noqa: F401
from icepack2.model.minimization import friction_power  # noqa: F401


def _shear_power(S_2, h, A, n):
    r"""One half-layer's share of the shear dissipation,
    :math:`h\,\frac{A}{n+1}|S|^{n+1}`."""
    S_n = conditional(eq(n, 1), S_2, S_2 ** ((n + 1) / 2))
    return h * A / (n + 1) * S_n


def _membrane_invariant(M):
    r""":math:`|M|^2_{\rm eff}`: the membrane part of the effective stress
    squared, :math:`\tfrac12 \tau'_{ij}\tau'_{ij}` over the horizontal
    deviatoric stresses and :math:`\tau_{zz} = -\operatorname{tr}\tau'`
    (icepack2's convention)."""
    mesh = ufl.domain.extract_unique_domain(M)
    d = mesh.geometric_dimension          # FD2026: attribute; FD<=2025: method
    d = d() if callable(d) else d
    return (inner(M, M) - tr(M) ** 2 / (d + 1)) / 2


def _half_layer_power(M_2, S, h, A, n):
    r"""One half-layer of a layer of thickness ``h``:
    :math:`\tfrac{h}{2}\,\tfrac{2A}{n+1}\,\tau_e^{n+1}` with
    :math:`\tau_e^2 = |M|^2_{\rm eff} + |S|^2` (``S`` None at the surface)."""
    t_2 = M_2 if S is None else M_2 + inner(S, S)
    t_n = conditional(eq(n, 1), t_2, t_2 ** ((n + 1) / 2))
    return h * A / (n + 1) * t_n


def layer_power(**kwargs):
    r"""Return one layer's dissipation, membrane and shear in one flow law

    Glen's law has one effective stress for every component,
    :math:`\tau_e^2 = \tfrac12\tau'_{ij}\tau'_{ij}`, here the membrane part
    plus the shear :math:`|S|^2`.  The shear in a layer runs from the stress
    on its lower face to the one on its upper face, so the layer is taken as
    two half-layers, each with the membrane stress and the shear of its
    face:

    .. math::
        \int \frac{h^l}{2}\,\frac{2A_l}{n_l+1}\left[
        \left(|M^l|^2_{\rm eff} + |S^{l-1}|^2\right)^{\frac{n_l+1}{2}}
        + \left(|M^l|^2_{\rm eff} + |S^l|^2\right)^{\frac{n_l+1}{2}}
        \right] dx

    with :math:`S^0` the basal stress and :math:`S^L = 0` at the surface.
    Summed over the column it replaces :func:`viscous_power`,
    :func:`interlayer_power` and :func:`basal_stress_power`, and equals
    them for linear ice.  Each half-layer shears under its own layer's law,
    so every interface is in series, and the bottom half-layer shears on a
    sliding bed too (add :func:`friction_power` for the sliding itself).
    It also keeps the membrane block of the Jacobian invertible at
    :math:`M = 0` wherever the ice shears.

    Parameters
    ----------
    membrane_stress : this layer's membrane stress
    stress_below : the stress on its lower face (the basal stress for the
        bottom layer)
    stress_above : the stress on its upper face, ``None`` for the top layer
    thickness, flow_law_coefficient, flow_law_exponent : this layer's
    measure : optional
        ``dx`` (the default); the vertex measure for nodal stresses.
    """
    M = kwargs["membrane_stress"]
    S_lo = kwargs["stress_below"]
    S_hi = kwargs.get("stress_above")
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    measure = kwargs.get("measure", dx)

    M_2 = _membrane_invariant(M)
    return (_half_layer_power(M_2, S_lo, h, A, n)
            + _half_layer_power(M_2, S_hi, h, A, n)) * measure


def slope_coupling(**kwargs):
    r"""Return the coupling of membrane stress and vertical shear on
    sloping layers

    The layers follow the bed, so a layer's velocity gradient is taken
    along a surface that slopes, relative to the ice surface, by
    :math:`\nabla z_l`, while the first-order strain rate is the one at
    constant depth,

    .. math::
        \dot\varepsilon\big|_z = \dot\varepsilon(u^l)
        - \operatorname{sym}\left(\nabla z_l \otimes \partial_z u\right).

    :func:`momentum_balance` pairs the membrane stress with the first
    term; this adds the second,

    .. math::
        \sum_l \int h^l\, M^l : \operatorname{sym}\left(\nabla z_l
        \otimes \partial_z u\big|_l\right) dx,

    with :math:`z_l = -\left(\sum_{k>l} h^k + h^l/2\right)` the depth of
    the middle of layer :math:`l` below the surface and
    :math:`\partial_z u|_l` the mean of the velocity gradients on its two
    faces: the jump between layer midpoints at an interior face, zero at
    the surface, and on a frozen bed the bottom layer's velocity over its
    half thickness.  On a sliding bed the bottom layer takes the gradient
    of its upper face.  It is bilinear in stress and velocity, so the
    system stays the derivative of one action.  It vanishes where the
    layers parallel the surface -- a uniform slab on any slope stays exact
    -- and without it a column on a sloping bed does not converge to the
    first-order solution as layers are added (ISMIP-HOM B levels off 16 %
    from it at 10 km).  Measuring the slope from the surface rather than
    the horizontal changes the strain rate only at second order in the
    surface slope, the order the first-order model itself neglects.

    Parameters
    ----------
    velocities, membrane_stresses, thicknesses : lists, bottom to top
    frozen_bed : bool
        No slip at the bed (default ``False``).
    """
    u = kwargs["velocities"]
    M = kwargs["membrane_stresses"]
    h = kwargs["thicknesses"]
    frozen = kwargs.get("frozen_bed", False)
    L = len(h)
    if not (len(u) == len(M) == L):
        raise ValueError("need one velocity, membrane stress and thickness per layer")

    rates = [(u[l + 1] - u[l]) / ((h[l] + h[l + 1]) / 2) for l in range(L - 1)]
    form = 0
    for l in range(L):
        above = sum(h[k] for k in range(l + 1, L)) if l < L - 1 else 0
        z_l = -(above + h[l] / 2)
        if l > 0:
            below_rate = rates[l - 1]
        elif frozen:
            below_rate = u[0] / (h[0] / 2)
        else:
            below_rate = None
        above_rate = rates[l] if l < L - 1 else 0 * u[l]
        if below_rate is None:
            if L == 1:
                continue
            dudz = above_rate
        else:
            dudz = (below_rate + above_rate) / 2
        form = form + h[l] * inner(M[l], sym(ufl.outer(grad(z_l), dudz)))
    if form == 0:
        return Constant(0.0) * dx(ufl.domain.extract_unique_domain(u[0]))
    return form * dx


def interlayer_power(**kwargs):
    r"""Return the interlayer shear dissipation potential

    .. math::
        \int (h^{l+1} + h^l)\,\frac{A}{n+1}\,\lvert S^l \rvert^{n+1}\,dx

    The factor :math:`h^{l+1} + h^l` arises from the normalisation of the
    velocity jump in the interlayer constitutive law.

    With the law of the layer above given too (``flow_law_coefficient_above``,
    ``flow_law_exponent_above``) the two half-layers are in series and each
    dissipates under its own law,

    .. math::
        \int \left(h^l\,\frac{A_l}{n_l+1}\,|S^l|^{n_l+1}
        + h^{l+1}\,\frac{A_{l+1}}{n_{l+1}+1}\,|S^l|^{n_{l+1}+1}\right) dx

    whose derivative is the series closure of
    :func:`multilayer.model.variational.interlayer_stress_law`.  ``measure``
    is the integral it is taken in (``dx``; the vertex measure for nodal
    stresses, together with the coupling term of the momentum balance).
    """
    S = kwargs["interlayer_stress"]
    h_above = kwargs["thickness_above"]
    h_below = kwargs["thickness_below"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    A_above, n_above = map(kwargs.get, ("flow_law_coefficient_above",
                                        "flow_law_exponent_above"))
    measure = kwargs.get("measure", dx)

    S_2 = inner(S, S)
    if A_above is None:
        S_n = conditional(eq(n, 1), S_2, S_2 ** ((n + 1) / 2))
        return (h_above + h_below) * A / (n + 1) * S_n * measure
    if n_above is None:
        raise ValueError("a series interface needs flow_law_exponent_above "
                         "with flow_law_coefficient_above")
    return (_shear_power(S_2, h_below, A, n)
            + _shear_power(S_2, h_above, A_above, n_above)) * measure


def basal_stress_power(**kwargs):
    r"""Return the basal shear dissipation potential for a frozen base

    .. math::
        \int h^1\,\frac{A}{n+1}\,\lvert \tau \rvert^{n+1}\,dx
    """
    τ = kwargs["basal_stress"]
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    measure = kwargs.get("measure", dx)

    τ_2 = inner(τ, τ)
    τ_n = conditional(eq(n, 1), τ_2, τ_2 ** ((n + 1) / 2))
    return h * A / (n + 1) * τ_n * measure


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
    A_above, n_above = map(kwargs.get, ("flow_law_coefficient_above",
                                        "flow_law_exponent_above"))
    α = kwargs.get("regularization", Constant(1e-4))
    τ_c = kwargs.get("reference_stress", Constant(0.1))
    measure = kwargs.get("measure", dx)

    A_lin = A * τ_c ** (n - Constant(1.0))
    series, series_lin = {}, {}
    if A_above is not None:
        if n_above is None:
            raise ValueError("a series interface needs flow_law_exponent_above "
                             "with flow_law_coefficient_above")
        series = dict(flow_law_coefficient_above=A_above,
                      flow_law_exponent_above=n_above)
        series_lin = dict(flow_law_coefficient_above=A_above * τ_c ** (n_above - Constant(1.0)),
                          flow_law_exponent_above=Constant(1.0))
    return (
        interlayer_power(
            interlayer_stress=S, thickness_above=h_above, thickness_below=h_below,
            flow_law_coefficient=A, flow_law_exponent=n, measure=measure, **series,
        )
        + α * interlayer_power(
            interlayer_stress=S, thickness_above=h_above, thickness_below=h_below,
            flow_law_coefficient=A_lin, flow_law_exponent=Constant(1.0),
            measure=measure, **series_lin,
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
    membrane_thickness, stress_measure, ice_density, gravity : optional
        As in :func:`multilayer.model.variational.momentum_balance`; the
        stress measure must match the one the shear powers are taken in.
    """
    u = kwargs["velocity"]
    M = kwargs["membrane_stress"]
    h = kwargs["thickness"]
    h_m = kwargs.get("membrane_thickness")
    if h_m is None:
        h_m = h
    s = kwargs["surface"]
    ρ = kwargs.get("ice_density", ρ_I)
    g_ = kwargs.get("gravity", g)
    dxs = kwargs.get("stress_measure", dx)

    τ = kwargs.get("basal_stress")
    S_above = kwargs.get("stress_above")
    S_below = kwargs.get("stress_below")

    ε = sym(grad(u))
    F = (-h_m * inner(M, ε) - ρ * g_ * h * inner(grad(s), u)) * dx

    if τ is not None:
        F += inner(τ, u) * dxs

    if S_above is not None:
        F += inner(S_above, u) * dxs
    if S_below is not None:
        F -= inner(S_below, u) * dxs

    mesh = ufl.domain.extract_unique_domain(u)
    ν = FacetNormal(mesh)
    F += ρ * g_ * avg(h) * inner(jump(s, ν), avg(u)) * dS

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

    # Clamp the log argument: a Newton iterate can overshoot the Coulomb
    # limit (r > 1), where 1 - r^(m+1) is negative and ln would be NaN.
    log_arg = max_value(Constant(1.0) - r_mp1, delta)
    return u_0 * β2e / (m + 1) * (-ufl.ln(log_arg)) * dx


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

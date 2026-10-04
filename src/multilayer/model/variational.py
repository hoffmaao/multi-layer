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

r"""Variational (weak-form) equations for the multilayer model

Each function returns a UFL Form.  Functions that are identical to their
icepack2 counterparts are re-exported; multilayer-specific functions
(interlayer stress law, basal stress law, and the extended momentum
balance) are new.

The model is derived from the depth integration of the first-order
approximation (FOA) over *L* vertical layers, following Jouvet (2015),
*J. Fluid Mech.*, 764, 26--51.
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
    sqrt,
)
from icepack2.model.utilities import get_test_function
from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Re-export from icepack2 -- identical for per-layer use
from icepack2.model.variational import flow_law  # noqa: F401

#: Column thickness (m) below which the series weights fade from the true
#: thickness fractions to an equal share.  It is far below any thickness
#: floor, so an empty layer has weight exactly zero wherever there is ice,
#: and it keeps the weights summing to one at ice-free nodes, where the
#: coefficient on the stress would otherwise vanish.
SERIES_WEIGHT_THICKNESS = 1e-3


def _test(field, kwargs):
    """The test function paired with ``field``: the caller's, or the one
    of the mixed space ``field`` was split from."""
    σ = kwargs.get("test_function")
    return get_test_function(field) if σ is None else σ


def _shear_compliance(S_2, A, n, linear=None):
    r"""The scalar :math:`c(S)` of a shear closure :math:`c(S)\,S = \ldots`:
    :math:`A|S|^{n-1}`, plus an optional :math:`n = 1` term in parallel
    (a regulariser, diffusion creep)."""
    S_n = conditional(eq(n, 1), Constant(1.0), S_2 ** ((n - 1) / 2))
    c = A * S_n
    if linear is not None:
        c = c + linear
    return c


def interlayer_stress_law(**kwargs):
    r"""Return the constitutive relation for interlayer shear stress

    The dual form of the interlayer traction (Jouvet 2015, eq. 2.57) is

    .. math::
        A \lvert S^l \rvert^{n-1} S^l
        = \frac{u^{l+1} - u^l}{h^{l+1} + h^l}

    This has the same structure as a friction law with coefficient *A*
    and exponent *n* applied to the normalised velocity jump between
    adjacent layers.

    With one law the whole jump is read with it -- by convention the law
    of the layer *below* the interface.  Given the law of the layer above
    as well (``flow_law_coefficient_above``, ``flow_law_exponent_above``)
    the two half-layers meeting at the interface are put **in series**:
    the shear stress is continuous across it, each half-layer shears under
    its own law, and the jump between the layer centres is the sum of the
    two,

    .. math::
        u^{l+1} - u^l = \left(h^l c_l(S) + h^{l+1} c_{l+1}(S)\right) S,
        \qquad c(S) = A|S|^{n-1}

    so, divided by :math:`h^l + h^{l+1}`, the closure above with the
    compliance replaced by its thickness-weighted mean.  Two layers of one
    law give exactly the single-law form; a layer that has thinned to
    nothing has weight zero, so its rheology leaves no trace at the
    interface, and the limit is reached continuously.  The weights are
    the true thickness fractions wherever the column is at least
    :data:`SERIES_WEIGHT_THICKNESS` thick, and fade to an equal share only
    as the column vanishes, so they sum to one everywhere.

    Parameters
    ----------
    interlayer_stress, velocity_above, velocity_below,
    thickness_above, thickness_below, flow_law_coefficient, flow_law_exponent
        As before.
    flow_law_coefficient_above, flow_law_exponent_above : optional
        The law of the layer above, for the series closure.
    linear_coefficient, linear_coefficient_above : optional
        An :math:`n = 1` term in parallel with each law -- a regulariser,
        or diffusion creep -- added to :math:`A|S|^{n-1}`.
    stress_regularization : float, optional
        Added to :math:`|S|^2` inside the power, so the Jacobian of a law
        with :math:`n < 3` stays finite at :math:`S = 0`.
    thickness_floor : float, optional
        Floor on :math:`h^l + h^{l+1}` in the velocity-jump normalisation,
        which vanishes with the column and is zero at ice-free nodes.  The
        series weights do not use it: they come from the true thicknesses,
        so an empty layer has weight zero under the floor too.  The floor
        is a dual-form device that the primal :func:`interlayer_power` does
        not carry, so the two forms agree wherever it does not engage.
    measure : optional
        The integral the closure is taken in: ``dx`` (the default) for a
        cellwise stress; :func:`multilayer.model.utilities.vertex_measure`
        collocates a nodal one at the vertices.
    test_function : optional
        Supplied by a caller that holds the test functions itself.
    """
    S = kwargs["interlayer_stress"]
    σ = _test(S, kwargs)
    u_above = kwargs["velocity_above"]
    u_below = kwargs["velocity_below"]
    h_above = kwargs["thickness_above"]
    h_below = kwargs["thickness_below"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    A_above, n_above = map(kwargs.get, ("flow_law_coefficient_above",
                                        "flow_law_exponent_above"))
    linear, linear_above = map(kwargs.get, ("linear_coefficient",
                                            "linear_coefficient_above"))
    eps = kwargs.get("stress_regularization", 0.0)
    floor = kwargs.get("thickness_floor", 0.0)
    measure = kwargs.get("measure", dx)

    h_total = h_above + h_below
    h_sum = max_value(h_total, Constant(floor)) if floor else h_total
    Δu = (u_above - u_below) / h_sum
    S_2 = inner(S, S)
    if eps:
        S_2 = S_2 + Constant(eps)
    c = _shear_compliance(S_2, A, n, linear)
    if A_above is not None:
        if n_above is None:
            raise ValueError("a series interface needs flow_law_exponent_above "
                             "with flow_law_coefficient_above")
        δ = Constant(SERIES_WEIGHT_THICKNESS)
        w_below = (h_below / max_value(h_total, δ)
                   + Constant(0.5) * (Constant(1.0) - min_value(h_total, δ) / δ))
        c = w_below * c + (Constant(1.0) - w_below) * _shear_compliance(
            S_2, A_above, n_above, linear_above)
    return inner(c * S - Δu, σ) * measure


def basal_stress_law(**kwargs):
    r"""Return the constitutive relation for basal stress on a frozen base

    For a no-slip (frozen) base the basal traction follows the same power
    law as the interlayer stress with :math:`u^0 = 0` and :math:`h^0 = 0`:

    .. math::
        A \lvert \tau \rvert^{n-1} \tau + u^1 / h^1 = 0

    The sign convention matches icepack2: :math:`\tau` opposes the velocity.
    """
    τ = kwargs["basal_stress"]
    σ = _test(τ, kwargs)
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))
    measure = kwargs.get("measure", dx)

    S_2 = inner(τ, τ)
    S_n = conditional(eq(n, 1), Constant(1.0), S_2 ** ((n - 1) / 2))
    return inner(A * S_n * τ + u / h, σ) * measure


def friction_law(**kwargs):
    r"""Return the Weertman sliding law for basal stress

    .. math::
        K \lvert \tau \rvert^{m-1} \tau + u = 0

    The sign convention matches icepack2: :math:`\tau` opposes the velocity.
    """
    τ, u = map(kwargs.get, ("basal_stress", "velocity"))
    σ = _test(τ, kwargs)
    K, m = map(kwargs.get, ("sliding_coefficient", "sliding_exponent"))
    measure = kwargs.get("measure", dx)
    τ_2 = inner(τ, τ)
    τ_m = conditional(eq(m, 1), Constant(1.0), τ_2 ** ((m - 1) / 2))
    return inner(K * τ_m * τ + u, σ) * measure


def sliding_law(**kwargs):
    r"""Close the basal stress on a sliding relation given as a traction
    magnitude: :math:`\tau = -\tau_b(u)\,u / |u|_{\rm reg}`, with
    :math:`|u|_{\rm reg} = \sqrt{|u|^2 + u_{\min}^2}`.

    Linear in :math:`\tau`, so the basal-stress block of the Jacobian is
    the identity, non-singular at :math:`\tau = 0`.  Any law written as a
    magnitude of the sliding velocity (Weertman, Budd, regularised
    Coulomb) closes this way; :func:`schoof_friction` is one.  The sign
    follows icepack2: :math:`\tau` opposes the velocity.

    Parameters
    ----------
    basal_stress, velocity : UFL split variables
    traction : UFL expression
        :math:`\tau_b(u) \ge 0` in MPa.
    speed_floor : float, optional
        :math:`u_{\min}` in m/yr (default 1).
    measure, test_function : optional
        As in :func:`interlayer_stress_law`.
    """
    τ, u = map(kwargs.get, ("basal_stress", "velocity"))
    σ = _test(τ, kwargs)
    τ_b = kwargs["traction"]
    u_min = kwargs.get("speed_floor", 1.0)
    measure = kwargs.get("measure", dx)
    u_reg = sqrt(inner(u, u) + Constant(u_min) ** 2)
    return inner(τ + τ_b * u / u_reg, σ) * measure


def momentum_balance(**kwargs):
    r"""Return the per-layer momentum balance for the multilayer model

    .. math::
        -h^l M^l : \varepsilon(v) + \tau \cdot v
        + S_{\mathrm{above}} \cdot v - S_{\mathrm{below}} \cdot v
        - \rho_I g\, h^l \nabla s \cdot v = 0

    Parameters
    ----------
    membrane_stress, velocity, thickness, surface : per-layer fields
    basal_stress : icepack2-convention basal drag (bottom layer only)
    stress_above : interlayer stress from the layer above, or ``None``
    stress_below : interlayer stress from the layer below, or ``None``
    membrane_thickness : optional
        The thickness the membrane coupling uses when it is not the
        layer's own -- a floored one that keeps the velocity coercive
        where the ice runs out.  The driving stress always sees the true
        thickness.
    stress_measure : optional
        The integral the basal and interlayer stress terms are taken in:
        ``dx`` by default, the vertex measure for nodal stresses, so that
        the balance and the closures are the derivative of one action.
    ice_density, gravity : optional
        Defaults to icepack2's constants.
    test_function : optional
        Supplied by a caller that holds the test functions itself.
    """
    M = kwargs["membrane_stress"]
    h = kwargs["thickness"]
    h_m = kwargs.get("membrane_thickness")
    if h_m is None:
        h_m = h
    s = kwargs["surface"]
    u = kwargs["velocity"]
    v = _test(u, kwargs)
    ρ = kwargs.get("ice_density", ρ_I)
    g_ = kwargs.get("gravity", g)
    dxs = kwargs.get("stress_measure", dx)

    τ = kwargs.get("basal_stress")
    S_above = kwargs.get("stress_above")
    S_below = kwargs.get("stress_below")

    ε = sym(grad(v))
    F = (-h_m * inner(M, ε) - ρ * g_ * h * inner(grad(s), v)) * dx

    if τ is not None:
        F += inner(τ, v) * dxs

    if S_above is not None:
        F += inner(S_above, v) * dxs
    if S_below is not None:
        F -= inner(S_below, v) * dxs

    mesh = ufl.domain.extract_unique_domain(v)
    ν = FacetNormal(mesh)
    F += ρ * g_ * avg(h) * inner(jump(s, ν), avg(v)) * dS

    return F


def schoof_friction(**kwargs):
    r"""Regularized Coulomb friction (RCF, Joughin et al. 2019/2024).

    .. math::
        |\tau| = \beta^2 \left(\frac{|u|}{|u| + u_0}\right)^{1/m}

    At low velocity (Weertman): :math:`|\tau| \sim \beta^2 (|u|/u_0)^{1/m}`

    At high velocity (Coulomb): :math:`|\tau| \to \beta^2`

    Structurally identical to Zoet & Iverson (2020) with
    :math:`\beta^2 = C \cdot N`.  Sign convention matches icepack2:
    :math:`\tau` opposes velocity.

    Parameters
    ----------
    basal_stress : UFL split variable
    velocity : UFL split variable
    friction_coefficient : Function or Constant
        :math:`\beta^2` in MPa. For explicit N dependence, pass
        :math:`C \cdot N` where C is the Coulomb coefficient and N
        is the effective pressure.
    transition_speed : Constant
        :math:`u_0` in m/yr. Default ~300.
    sliding_exponent : Constant
        m, typically Glen's n = 3.
    """
    τ = kwargs["basal_stress"]
    u = kwargs["velocity"]
    σ = _test(τ, kwargs)
    measure = kwargs.get("measure", dx)

    β2 = kwargs["friction_coefficient"]
    u_0 = kwargs["transition_speed"]
    m = kwargs["sliding_exponent"]

    # Primal RCF (Joughin et al. 2024, Eq. 7)
    eps = Constant(1e-4)  # ~0.01 m/yr speed floor (prevents NaN on fine meshes)
    u_mag = ufl.sqrt(inner(u, u) + eps)
    ratio = u_mag / (u_mag + u_0)
    τ_mag = β2 * ratio ** (Constant(1.0) / m)

    return inner(τ + τ_mag * u / u_mag, σ) * measure


def calving_terminus(**kwargs):
    r"""Return the ocean back-pressure at the terminus for one layer

    The total ice-minus-ocean pressure is split equally among layers
    via the ``layer_fraction`` parameter (default 1).
    """
    h, s, u = map(kwargs.get, ("thickness", "surface", "velocity"))
    v = get_test_function(u)
    outflow_ids = kwargs["outflow_ids"]
    layer_fraction = kwargs.get("layer_fraction", Constant(1.0))

    mesh = ufl.domain.extract_unique_domain(v)
    ν = FacetNormal(mesh)

    f_I = 0.5 * ρ_I * g * h ** 2
    d = min_value(0, s - h)
    f_W = 0.5 * ρ_W * g * d ** 2

    return layer_fraction * (f_I - f_W) * inner(v, ν) * ds(outflow_ids)

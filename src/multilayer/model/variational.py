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
from icepack2.model.utilities import get_test_function
from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Re-export from icepack2 -- identical for per-layer use
from icepack2.model.variational import flow_law  # noqa: F401


def interlayer_stress_law(**kwargs):
    r"""Return the constitutive relation for interlayer shear stress

    The dual form of the interlayer traction (Jouvet 2015, eq. 2.57) is

    .. math::
        A \lvert S^l \rvert^{n-1} S^l
        = \frac{u^{l+1} - u^l}{h^{l+1} + h^l}

    This has the same structure as a friction law with coefficient *A*
    and exponent *n* applied to the normalised velocity jump between
    adjacent layers.
    """
    S = kwargs["interlayer_stress"]
    σ = get_test_function(S)
    u_above = kwargs["velocity_above"]
    u_below = kwargs["velocity_below"]
    h_above = kwargs["thickness_above"]
    h_below = kwargs["thickness_below"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    Δu = (u_above - u_below) / (h_above + h_below)
    S_2 = inner(S, S)
    S_n = conditional(eq(n, 1), Constant(1.0), S_2 ** ((n - 1) / 2))
    return inner(A * S_n * S - Δu, σ) * dx


def basal_stress_law(**kwargs):
    r"""Return the constitutive relation for basal stress on a frozen base

    For a no-slip (frozen) base the basal traction follows the same power
    law as the interlayer stress with :math:`u^0 = 0` and :math:`h^0 = 0`:

    .. math::
        A \lvert \tau \rvert^{n-1} \tau + u^1 / h^1 = 0

    The sign convention matches icepack2: :math:`\tau` opposes the velocity.
    """
    τ = kwargs["basal_stress"]
    σ = get_test_function(τ)
    u = kwargs["velocity"]
    h = kwargs["thickness"]
    A, n = map(kwargs.get, ("flow_law_coefficient", "flow_law_exponent"))

    S_2 = inner(τ, τ)
    S_n = conditional(eq(n, 1), Constant(1.0), S_2 ** ((n - 1) / 2))
    return inner(A * S_n * τ + u / h, σ) * dx


def friction_law(**kwargs):
    r"""Return the Weertman sliding law for basal stress

    .. math::
        K \lvert \tau \rvert^{m-1} \tau + u = 0

    The sign convention matches icepack2: :math:`\tau` opposes the velocity.
    """
    τ, u = map(kwargs.get, ("basal_stress", "velocity"))
    σ = get_test_function(τ)
    K, m = map(kwargs.get, ("sliding_coefficient", "sliding_exponent"))
    τ_2 = inner(τ, τ)
    τ_m = conditional(eq(m, 1), Constant(1.0), τ_2 ** ((m - 1) / 2))
    return inner(K * τ_m * τ + u, σ) * dx


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
    """
    M = kwargs["membrane_stress"]
    h = kwargs["thickness"]
    s = kwargs["surface"]
    u = kwargs["velocity"]
    v = get_test_function(u)

    τ = kwargs.get("basal_stress")
    S_above = kwargs.get("stress_above")
    S_below = kwargs.get("stress_below")

    ε = sym(grad(v))
    F = (-h * inner(M, ε) - ρ_I * g * h * inner(grad(s), v)) * dx

    if τ is not None:
        F += inner(τ, v) * dx

    if S_above is not None:
        F += inner(S_above, v) * dx
    if S_below is not None:
        F -= inner(S_below, v) * dx

    mesh = ufl.domain.extract_unique_domain(v)
    ν = FacetNormal(mesh)
    F += ρ_I * g * avg(h) * inner(jump(s, ν), avg(v)) * dS

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
    σ = get_test_function(τ)

    β2 = kwargs["friction_coefficient"]
    u_0 = kwargs["transition_speed"]
    m = kwargs["sliding_exponent"]

    # Primal RCF (Joughin et al. 2024, Eq. 7)
    eps = Constant(1e-4)  # ~0.01 m/yr speed floor (prevents NaN on fine meshes)
    u_mag = ufl.sqrt(inner(u, u) + eps)
    ratio = u_mag / (u_mag + u_0)
    τ_mag = β2 * ratio ** (Constant(1.0) / m)

    return inner(τ + τ_mag * u / u_mag, σ) * dx


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

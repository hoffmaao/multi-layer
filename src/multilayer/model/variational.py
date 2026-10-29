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
icepack2 counterparts are re-exported.  The interlayer stress law, the
sliding law and the per-layer momentum balance live in icepack_tools,
which builds the dual model for one layer or several, and are re-exported
here; the frozen-base, Weertman, Schoof and terminus laws are this
package's.

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
    dx,
    ds,
    FacetNormal,
    min_value,
)
from icepack2.model.utilities import get_test_function
from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g

# Re-export from icepack2 -- identical for per-layer use
from icepack2.model.variational import flow_law  # noqa: F401

# Re-export from icepack_tools, which owns the layered dual building blocks
from icepack_tools.viscosity import (  # noqa: F401
    SERIES_WEIGHT_THICKNESS,
    interlayer_stress_law,
)
from icepack_tools.friction import sliding_law  # noqa: F401
from icepack_tools.momentum import momentum_balance  # noqa: F401


def _test(field, kwargs):
    """The test function paired with ``field``: the caller's, or the one
    of the mixed space ``field`` was split from."""
    σ = kwargs.get("test_function")
    return get_test_function(field) if σ is None else σ


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

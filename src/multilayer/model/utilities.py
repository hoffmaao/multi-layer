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

r"""Utilities for constructing multilayer function spaces and extracting
per-layer fields from mixed functions."""

import firedrake
from icepack2.model.utilities import get_test_function  # noqa: F401


def create_function_space(mesh, num_layers, degree=1):
    r"""Create the mixed function space for the multilayer model

    For *L* layers the space has 3L subspaces ordered as

    .. math::
       [u^1, M^1, S^0,\; u^2, M^2, S^1,\; \ldots,\; u^L, M^L, S^{L-1}]

    where :math:`u^l` is the velocity (CG vector), :math:`M^l` is the
    membrane stress (DG symmetric tensor), and :math:`S^l` is the
    interlayer or basal stress (DG vector).

    Parameters
    ----------
    mesh : firedrake.Mesh
    num_layers : int
    degree : int, optional
        Polynomial degree for velocity (CG). Stresses use degree - 1 (DG).
    """
    cg = firedrake.FiniteElement("CG", "triangle", degree)
    dg = firedrake.FiniteElement("DG", "triangle", degree - 1)
    V = firedrake.VectorFunctionSpace(mesh, cg)
    Σ = firedrake.TensorFunctionSpace(mesh, dg, symmetry=True)
    T = firedrake.VectorFunctionSpace(mesh, dg)

    spaces = []
    for l in range(num_layers):
        spaces.extend([V, Σ, T])

    return firedrake.MixedFunctionSpace(spaces)


def split_fields(z, num_layers):
    r"""Extract per-layer fields from a split mixed function

    Parameters
    ----------
    z : tuple
        Result of ``firedrake.split(z_func)`` or ``z_func.subfunctions``.
    num_layers : int

    Returns
    -------
    list of dict
        ``layers[l]`` has keys ``"velocity"``, ``"membrane_stress"``, and
        ``"interlayer_stress"``.  ``layers[0]["interlayer_stress"]`` is the
        basal stress :math:`S^0`.
    """
    layers = []
    for l in range(num_layers):
        layers.append({
            "velocity": z[3 * l],
            "membrane_stress": z[3 * l + 1],
            "interlayer_stress": z[3 * l + 2],
        })
    return layers


def effective_pressure(thickness, surface, bed, Q):
    r"""Compute effective pressure assuming perfect hydrological connectivity.

    Following Joughin et al. (2024, The Cryosphere), Eq. 5:

    .. math::
        N = \rho_I g (h - h_f)

    where :math:`h_f = \max(0, -b \cdot \rho_W / \rho_I)` is the flotation
    height. N is clamped to be non-negative (zero on floating ice).

    Parameters
    ----------
    thickness : Function
        Ice thickness H.
    surface : Function
        Ice surface elevation s.
    bed : Function
        Bed elevation b.
    Q : FunctionSpace
        Scalar CG1 space for the result.

    Returns
    -------
    Function
        Effective pressure N in MPa (icepack2 units).
    """
    from icepack2.constants import ice_density as ρ_I, water_density as ρ_W, gravity as g
    from firedrake import Function, Constant, max_value

    # Height above flotation
    h_f = max_value(Constant(0.0), -bed * Constant(ρ_W / ρ_I))
    haf = surface - h_f

    N = Function(Q, name="effective_pressure")
    N.interpolate(max_value(ρ_I * g * haf, Constant(0.0)))
    return N


def grounding_line_weakening(thickness, surface, bed, Q, h_T=41.0):
    r"""Compute grounding line friction weakening factor.

    Following Joughin et al. (2024, The Cryosphere), Eq. 8:

    .. math::
        \lambda = \begin{cases}
            1 & \text{if } h - h_f > h_T \\
            \frac{h - h_f}{\min(h_T,\, h_0 - h_f)} & \text{if } 0 < h - h_f \le h_T \\
            0 & \text{if } h - h_f \le 0
        \end{cases}

    For a diagnostic (snapshot) inversion :math:`h_0 = h`, so the
    denominator simplifies to :math:`\min(h_T, h - h_f)` and
    :math:`\lambda = 1` everywhere that is grounded.  The weakening
    only activates during prognostic runs when the ice thins toward
    flotation below its initial state.

    For the diagnostic case the effective result is:

    .. math::
        \lambda = \begin{cases}
            1 & \text{if } h - h_f > 0 \\
            0 & \text{otherwise}
        \end{cases}

    To get a smooth transition for the diagnostic inversion, we use the
    continuous form :math:`\lambda = \min((h-h_f)/h_T, 1)` which tapers
    friction linearly over the last :math:`h_T` metres above flotation.

    Parameters
    ----------
    thickness, surface, bed : Function
    Q : FunctionSpace
    h_T : float
        Weakening threshold in meters above flotation.
        Default 41 m (Joughin et al. 2024, best fit for Schoof friction).

    Returns
    -------
    Function
        Weakening factor lambda in [0, 1].
    """
    from icepack2.constants import water_density as ρ_W, ice_density as ρ_I
    from firedrake import Function, Constant, max_value, min_value

    h_f = max_value(Constant(0.0), -bed * Constant(ρ_W / ρ_I))
    haf = max_value(surface - h_f, Constant(0.0))

    lam = Function(Q, name="gl_weakening")
    lam.interpolate(min_value(haf / Constant(h_T), Constant(1.0)))
    return lam


def depth_averaged_velocity(z, h_layers, thickness, V=None):
    r"""Return the thickness-weighted depth-averaged velocity.

    .. math::
        \bar{u} = \frac{1}{H} \sum_{l=1}^{L} h^l\, u^l

    This is the velocity that drives the depth-integrated continuity
    (mass-balance) equation for the multilayer column.

    Parameters
    ----------
    z : firedrake.Function
        The multilayer mixed function (so ``z.subfunctions[3*l]`` is
        :math:`u^l`).
    h_layers : list
        Per-layer thicknesses (UFL expressions), as from
        :func:`layer_thicknesses`.
    thickness : UFL expression or Function
        Total thickness :math:`H`.
    V : firedrake.FunctionSpace, optional
        Vector space for the result.  Defaults to the velocity subspace.

    Returns
    -------
    firedrake.Function
        :math:`\bar{u}` projected into ``V``.
    """
    subs = z.subfunctions
    num_layers = len(h_layers)
    u_bar = sum(h_layers[l] * subs[3 * l] for l in range(num_layers)) / thickness
    if V is None:
        V = subs[0].function_space()
    return firedrake.Function(V).project(u_bar)


def layer_thicknesses(h, num_layers, fractions=None):
    r"""Return per-layer thicknesses

    Parameters
    ----------
    h : UFL expression or Function
        Total ice thickness.
    num_layers : int
    fractions : list of float, optional
        Fraction of total thickness for each layer, ordered bottom to top.
        Must sum to 1.  If ``None``, layers are uniform.

    Returns
    -------
    list of UFL expressions
        :math:`h^1, \ldots, h^L` (bottom to top).
    """
    if fractions is None:
        return [h / num_layers for _ in range(num_layers)]
    assert len(fractions) == num_layers
    return [h * f for f in fractions]

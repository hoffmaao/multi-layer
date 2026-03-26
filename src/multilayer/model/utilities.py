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

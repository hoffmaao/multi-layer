r"""The series interface and nodal stresses: the algebra, without solves.

A layer of zero thickness should leave no trace: the model with it must be
the model without it.  Two things in the discretisation decide whether it
does.  Which law closes an interface -- by convention the layer below's,
so an empty basal layer's rheology still governed the shear above it; the
series closure of ``interlayer_stress_law`` weights the two laws by the
thicknesses on either side, so an empty layer has weight zero.  And where
the interface and basal stresses live -- cellwise, closed on cell means,
an empty layer's momentum balance equates its two interface stresses only
in projection; at the vertices (``create_function_space(stress_family=
"CG")``) with every stress term in the vertex measure, it equates them
node by node.  These checks assemble the forms at a fixed state and
compare:

  * the series closure with one law on both sides is the single-law form,
  * with the layer below empty it is the law above's form, with the layer
    above empty the law below's,
  * the series ``interlayer_power`` is the potential of the series closure
    (its derivative, with the momentum coupling, assembles to the same
    residual), so the primal and dual forms stay one action,
  * in the vertex measure a nodal closure is collocation: each node's
    residual is the lumped mass times the pointwise relation,
  * the default spaces are unchanged.

The solved equivalences -- an empty top layer is the one-layer model, an
empty middle layer is the two-layer model, the sliding velocity converges
as a layer thins -- are in ``icepack_tools/test/empty_layer_test.py``,
which solves with this package's closures.

    python -u series_interface_test.py     (or pytest)
"""
import numpy as np
import firedrake
from firedrake import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, SpatialCoordinate,
    TestFunction, as_vector, assemble, derivative, dx, inner, split,
)

from multilayer.model.utilities import (create_function_space, split_fields,
                                        stress_family, vertex_measure)
from multilayer.model.variational import interlayer_stress_law
from multilayer.model.minimization import interlayer_power

TEMPERATE, GBS = (46.0, 4.0), (0.45, 1.8)
TOL = 1e-13
#: one quadrature degree for every form, so two spellings of one integrand
#: are not told apart by the degree UFL estimates for each
FC = {"quadrature_degree": 4}


def state(stress_family="DG", seed=0):
    """A two-layer mixed function at a fixed, non-zero state on a slab,
    with the geometry its closures read."""
    mesh = firedrake.RectangleMesh(8, 4, 40e3, 12e3, diagonal="crossed")
    x, y = SpatialCoordinate(mesh)
    Q = FunctionSpace(mesh, "CG", 1)
    H = Function(Q).interpolate(Constant(900.0) - Constant(840.0) * x / Constant(40e3))
    Z = create_function_space(mesh, 2, stress_family=stress_family)
    z = Function(Z)
    rng = np.random.default_rng(seed)
    for l in range(2):
        z.subfunctions[3 * l].interpolate(
            as_vector([Constant(100.0 + 40.0 * l) + Constant(300.0) * x / Constant(40e3),
                       Constant(5.0) * y / Constant(12e3)]))
        for i in (3 * l + 1, 3 * l + 2):
            dat = z.subfunctions[i].dat
            dat.data[:] = 0.05 + 0.02 * rng.standard_normal(dat.data.shape)
    return mesh, Q, H, Z, z


def closure(z, H, f_below, law_below, law_above=None, measure=dx, **extra):
    """The assembled interlayer closure (block S1) for a basal layer of
    ``f_below * H``."""
    f = split_fields(split(z), 2)
    kw = dict(interlayer_stress=f[1]["interlayer_stress"],
              velocity_above=f[1]["velocity"], velocity_below=f[0]["velocity"],
              thickness_above=Constant(1.0 - f_below) * H,
              thickness_below=Constant(f_below) * H,
              flow_law_coefficient=Constant(law_below[0]),
              flow_law_exponent=Constant(law_below[1]), measure=measure, **extra)
    if law_above is not None:
        kw.update(flow_law_coefficient_above=Constant(law_above[0]),
                  flow_law_exponent_above=Constant(law_above[1]))
    return assemble(interlayer_stress_law(**kw),
                    form_compiler_parameters=FC).subfunctions[5].dat.data_ro.copy()


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def test_series_with_one_law_is_the_single_law():
    mesh, Q, H, Z, z = state()
    single = closure(z, H, 0.3, TEMPERATE)
    series = closure(z, H, 0.3, TEMPERATE, TEMPERATE)
    d = rel(series, single)
    print(f"  series, one law on both sides, vs the single law: {d:.1e}")
    assert d < TOL


def test_empty_side_takes_the_other_law():
    mesh, Q, H, Z, z = state()
    for f_below, label, expect in ((0.0, "layer below empty", GBS),
                                   (1.0, "layer above empty", TEMPERATE)):
        series = closure(z, H, f_below, TEMPERATE, GBS)
        single = closure(z, H, f_below, expect)
        other = closure(z, H, f_below, TEMPERATE if expect is GBS else GBS)
        d, d_other = rel(series, single), rel(series, other)
        print(f"  {label}: vs the law that remains {d:.1e}, vs the one that left {d_other:.1e}")
        assert d < TOL and d_other > 1e-3


def test_empty_side_takes_the_other_law_under_the_floor():
    """With a thickness floor far above the column, so that it engages
    everywhere, an empty layer on either side still has weight zero."""
    mesh, Q, H, Z, z = state()
    floor = dict(thickness_floor=1e4)
    for f_below, label, expect in ((0.0, "layer below empty", GBS),
                                   (1.0, "layer above empty", TEMPERATE)):
        series = closure(z, H, f_below, TEMPERATE, GBS, **floor)
        single = closure(z, H, f_below, expect, **floor)
        d = rel(series, single)
        print(f"  {label}, floor engaged: vs the law that remains {d:.1e}")
        assert d < TOL


def test_nodal_stresses_need_degree_one():
    mesh = firedrake.UnitSquareMesh(2, 2)
    try:
        create_function_space(mesh, 2, degree=2, stress_family="CG")
    except ValueError:
        return
    raise AssertionError("degree 2 nodal stresses were accepted")


def test_series_power_is_the_potential_of_the_series_closure():
    """d/dS of the series interlayer power, with the coupling -S.(u_above
    - u_below), is the series closure times (h_above + h_below)."""
    mesh, Q, H, Z, z = state()
    f = split_fields(split(z), 2)
    S = f[1]["interlayer_stress"]
    h_b, h_a = Constant(0.3) * H, Constant(0.7) * H
    action = (interlayer_power(interlayer_stress=S, thickness_above=h_a, thickness_below=h_b,
                               flow_law_coefficient=Constant(TEMPERATE[0]),
                               flow_law_exponent=Constant(TEMPERATE[1]),
                               flow_law_coefficient_above=Constant(GBS[0]),
                               flow_law_exponent_above=Constant(GBS[1]))
              - inner(S, f[1]["velocity"] - f[0]["velocity"]) * dx)
    from_action = assemble(derivative(action, z),
                           form_compiler_parameters=FC).subfunctions[5].dat.data_ro.copy()
    # the closure, scaled back by h_sum: c S h_sum - du
    sigma = split(TestFunction(Z))[5]
    law = interlayer_stress_law(
        interlayer_stress=S, test_function=sigma,
        velocity_above=f[1]["velocity"], velocity_below=f[0]["velocity"],
        thickness_above=h_a, thickness_below=h_b,
        flow_law_coefficient=Constant(TEMPERATE[0]), flow_law_exponent=Constant(TEMPERATE[1]),
        flow_law_coefficient_above=Constant(GBS[0]), flow_law_exponent_above=Constant(GBS[1]))
    # multiply the integrand by h_sum: replace the test function
    scaled = firedrake.replace(law, {sigma: (h_a + h_b) * sigma})
    from_closure = assemble(scaled, form_compiler_parameters=FC).subfunctions[5].dat.data_ro.copy()
    d = rel(from_action, from_closure)
    print(f"  series power's derivative vs the series closure: {d:.1e}")
    assert d < 1e-10


def test_nodal_closure_is_collocation():
    """With nodal stresses and the vertex measure, node i's residual is its
    lumped mass times c(S_i) S_i - (u1_i - u0_i) / h_i."""
    mesh, Q, H, Z, z = state(stress_family="CG")
    assert stress_family(Z.sub(2)) == "Lagrange"
    dxv = vertex_measure(mesh)
    res = closure(z, H, 0.3, TEMPERATE, GBS, measure=dxv)
    # by hand, at the nodes
    V = VectorFunctionSpace(mesh, "CG", 1)
    lumped = assemble(TestFunction(Q) * dxv).dat.data_ro
    u0, u1 = (z.subfunctions[3 * l].dat.data_ro for l in range(2))
    S = z.subfunctions[5].dat.data_ro
    Hn = H.dat.data_ro
    h_b, h_a = 0.3 * Hn, 0.7 * Hn
    S2 = (S ** 2).sum(axis=1)
    A_b, n_b = TEMPERATE
    A_a, n_a = GBS
    c = (h_b * A_b * S2 ** ((n_b - 1) / 2) + h_a * A_a * S2 ** ((n_a - 1) / 2)) / (h_a + h_b)
    by_hand = lumped[:, None] * (c[:, None] * S - (u1 - u0) / (h_a + h_b)[:, None])
    d = rel(res, by_hand)
    print(f"  nodal closure vs collocation by hand: {d:.1e}")
    assert d < TOL
    assert V.dim() == Z.sub(2).dim()


def test_default_spaces_unchanged():
    mesh = firedrake.UnitSquareMesh(2, 2)
    Z = create_function_space(mesh, 2)
    fams = [stress_family(Z.sub(i)) for i in range(6)]
    print(f"  default families: {fams}")
    assert fams == ["Lagrange", "Discontinuous Lagrange", "Discontinuous Lagrange"] * 2
    Zn = create_function_space(mesh, 2, stress_family="CG")
    assert stress_family(Zn.sub(2)) == "Lagrange" and stress_family(Zn.sub(1)) == "Discontinuous Lagrange"


def main():
    print("The series interface and nodal stresses, assembled:")
    test_series_with_one_law_is_the_single_law()
    test_empty_side_takes_the_other_law()
    test_empty_side_takes_the_other_law_under_the_floor()
    test_nodal_stresses_need_degree_one()
    test_series_power_is_the_potential_of_the_series_closure()
    test_nodal_closure_is_collocation()
    test_default_spaces_unchanged()
    print("all checks passed")


if __name__ == "__main__":
    main()

# multilayer

This package implements the multilayer model described in [Jouvet (2015)](https://doi.org/10.1017/jfm.2014.689), which splits the ice column into a stack of thin layers, each with its own horizontal velocity.
Adjacent layers are coupled through interlayer shear tractions.
As the number of layers increases, the solution converges to the first-order (Blatter-Pattyn) solution, at second order in the layer thickness.

Unlike the SSA, which treats the ice as a plug, the multilayer model captures vertical shear.
This matters wherever basal drag is significant (near grounding lines, in ice streams, and in regions with strong bed topography). The model uses the **dual formulation** from [icepack2](https://github.com/icepack/icepack2), where membrane stress and interlayer shear stress are explicit unknowns alongside velocity.
This formulation is well-posed at zero thickness, which enables natural handling of calving fronts terminus evolution, and areas where layers thin to zero (over nunataks for instance).


## Multilayer momentum balance

Each layer has three fields: velocity, membrane stress, and interlayer stress.
The interlayer stress has the same mathematical structure as a friction law --it relates the shear stress at an interface to the normalised velocity between layers.

The action functional (`multilayer.model.minimization`) is:

- **Layer power** per layer (`layer_power`): the layer is taken as two
  half-layers under its own flow law, each with one effective stress made
  of the membrane stress and the shear on its face, as Glen's law has.
  Every interface is then in series (each half-layer shears under its own
  law), and the bottom half-layer shears under the basal stress on a
  sliding bed as well as a frozen one.
- **Momentum balance** per layer (`momentum_balance`, including the
  interlayer coupling)
- **Slope coupling** (`slope_coupling`): layers follow the bed, so the
  strain rate along a layer differs from the one at constant depth by the
  layer slope times the vertical shear; this restores it.
- **Friction power** at a sliding bed (`friction_power`)

Taking the Gateaux derivative of the total action recovers all the weak-form equations.

The older terms -- `viscous_power` per layer with `interlayer_power` at the
interfaces and `basal_stress_power` at a frozen bed -- give the membrane
and shear stresses separate flow laws.  They agree with `layer_power` for
linear ice and for a laterally uniform slab and are kept for existing
models, but for $n > 1$ they make ice that both stretches and shears too
stiff, `friction_power` alone with them drops the shear of the bottom
half-layer, and without `slope_coupling` a column on a sloping bed does not
converge to the first-order solution (see [ISMIP-HOM
verification](#ismip-hom-verification)).  The residual-form closures in
`multilayer.model.variational` (from `icepack_tools`) are the derivatives
of these older terms.

## Composite rheology

Ice near the bed may deform via different mechanisms than ice near the surface (i.e.by dislocation creep, $n = 4$ vs. while  grain boundary sliding, $n = 1.8$).
The standard Glen's law with $n = 3$ is an effective average of these two mechanisms (Goldsby & Kohlstedt 2001, Behn et al. 2021, Ranganathan & Minchew 2024).

The multilayer model naturally accommodates rheologies that change as a function of depth.
We can assign different stress exponents and rate factors to each layer, representing the transition from warm basal ice (large recrystallised grains, dislocation creep) to cold upper ice (smaller grains, grain boundary sliding).

## Layers that vanish

A layer whose thickness is zero somewhere -- a temperate basal layer that
pinches out up-glacier, say -- should leave no trace there: the model with
it must be the model without it.  Two choices in the discretisation decide
whether it does, and both are options of this package.

**Series interfaces.** Closing an interface with the law of the layer
below it alone lets an empty basal layer's rheology still govern the
shear above it.  `layer_power` always puts the two half-layers in series,
and so do `interlayer_stress_law` and `interlayer_power` when given the law
of the layer above as well: the shear stress is continuous across the
interface and each half-layer shears under its own law, so the closure's
compliance is the thickness-weighted mean of the two.  Two layers of one
law give exactly the single-law form; a layer of zero thickness has
weight zero.

**Nodal stresses.** The basal and interlayer stresses live one value per
cell by default, closed on the cell means of the velocities, while the
momentum balances that tie them are tested with vertex functions.  An
empty layer's balance then equates its two interface stresses only in
projection, and the sliding velocity under an empty bottom layer is
determined only through cell means and comes out rough.
`create_function_space(stress_family="CG")` puts those stresses at the
vertices with the velocities, and every term they appear in -- the
closures and the stress terms of the momentum balance -- is taken in the
vertex measure (`vertex_measure`), so it is still the derivative of one
action.  Stress continuity through an empty layer then holds node by
node, an empty middle layer drops out exactly, and the sliding velocity
is determined node by node.  The membrane stress stays cellwise.

`test/series_interface_test.py` checks the algebra: the series closure
with one law is the single-law form; with either side empty it is the
other law's form; the series power is the potential of the series
closure; a nodal closure is collocation.  The solved equivalences -- an
empty top layer is the one-layer model, an empty middle layer is the
two-layer model, a thinning layer converges to the empty one -- are in
`icepack_tools`, which solves with these closures.

## ISMIP-HOM verification

Each layer velocity stands for the velocity at the layer's mid-height, so
as layers are added the column should converge to the first-order
solution there.  `multilayer.first_order` solves the first-order
flowline problem independently of this package (Q2 elements on a
terrain-following mesh with a level on every change of flow law); at
ISMIP-HOM's density it reproduces the intercomparison median (Pattyn et
al. 2008) to 0.1-1.4 % for experiment B at 20-160 km and D at 5-160 km,
where the full-Stokes and first-order submissions agree.

Against it, the error of the layer velocities (RMS at the layer
mid-heights, relative to the largest speed) falls about fourfold each time
the layers are halved:

| case | 5 layers | 10 layers | 20 layers |
|---|---|---|---|
| ISMIP-HOM B, 80 km, $n = 3$ | 2.7 % | 0.69 % | 0.18 % |
| ISMIP-HOM B, 10 km, $n = 3$ | 4.3 % | 1.1 % | 0.26 % |
| sliding slab (ISMIP-HOM D friction, 6 km, 600 m), $n = 1.8$ | 3.2 % (4 layers) | 0.79 % (8) | 0.19 % (16) |
| the same slab with 200 m of $n = 4$ basal ice | 5.1 % (4 layers) | 1.3 % (8) | |

Without the fixes of `layer_power` and `slope_coupling` (the older terms
above, which every version up to this one used) the column does not
converge: B levels off 18 % from the first-order solution at 10 km and
10 % at the peak at 80 km with 40 layers, and the sliding slab 7.5 % low.
`test/first_order_test.py` checks the convergence.

`ismip_hom_b.py` and `ismip_hom_d.py` use ISMIP-HOM's ice density,
910 kg m$^{-3}$ (icepack2's is 917, which makes B about 2.3 % fast), and
`plot_ismip_hom.py` compares them with the intercomparison median and its
16-84 % band: a few submissions sit far from the rest (in D at 80 km one
peaks at 192 m/yr and one at 61, against a cluster near 97), and drag the
mean.  Five D submissions (rhi1-rhi5) are half a period out of phase,
peaking at $L/4$ rather than over the slippery spot at $3L/4$, and are
shifted back.  The figures plot the top layer's velocity, which sits half
a layer below the surface.


## Installation

The package depends on [Firedrake](https://firedrakeproject.org/),
[icepack2](https://github.com/icepack/icepack2) and
[icepack_tools](https://github.com/hoffmaao/icepack_tools).  The mixed
function space, the per-layer momentum balance, the interlayer stress
law, the sliding law and the nodal-stress helpers live in icepack_tools,
which builds the dual model for one layer or several; this package
re-exports them under their usual names (`create_function_space`,
`momentum_balance`, `interlayer_stress_law`, ...) and adds the layered
model around them.  With those three installed:

```
git clone https://github.com/hoffmaao/multi-layer.git
cd multi-layer
pip install -e .
```

## Tutorial

The `notebooks/` directory contains a Jupyter notebook that walks through the model step by step:

1. Single-layer SSA on a slab
2. Two-layer model with uniform Glen's law
3. Two-layer composite rheology (dislocation creep + grain boundary sliding)

## Running the ISMIP-HOM experiments

From the `test/` directory:

```python
python ismip_hom_b.py          # Experiment B, uniform n=3, L=1-5 layers
python ismip_hom_d.py          # Experiment D, uniform n=3, L=1-5 layers
python ismip_hom_b_composite.py  # Experiment B, composite n=4/1.8
python plot_ismip_hom.py       # Comparison with intercomparison data
python plot_composite.py       # Composite vs uniform comparison
python plot_schematic.py       # 3D schematic figure
python series_interface_test.py  # series interfaces and nodal stresses, assembled
python first_order_test.py     # convergence to the first-order solution (a few minutes)
```

`plot_ismip_hom.py` and `plot_composite.py` read the intercomparison
submissions from `/tmp/ismip_hom_data/ismip_all` (the Pattyn et al. 2008
supplement).

## Running the MISMIP+ experiment

The package also includes the MISMIP+ marine-ice-sheet benchmark geometry
(Asay-Davis et al. 2016) with a composite, depth-varying rheology: a soft
temperate base (15% of the column, $n = 4$, $A = 46$ MPa$^{-4}$ yr$^{-1}$;
Goldsby & Kohlstedt 2001) beneath a colder grain-boundary-sliding layer
($n = 1.8$, $A = 0.24$ MPa$^{-1.8}$ yr$^{-1}$, obtained from the Cuffey &
Paterson 2010 $n = 3$ rate factor at $-20$ °C stress-matched to $n = 1.8$).
Grounded ice slides by a regularised-Coulomb (Schoof) law whose drag vanishes
at the grounding line, and the terminus carries the ocean back-pressure.

```python
python mismip_plus.py          # diagnostic two-layer velocity solve
python mismip_plus_spinup.py   # prognostic spin-up + Ice1r melt response
```

`mismip_plus.py` solves the two-layer momentum balance on a prescribed
geometry, warm-started from a depth-averaged ($n = 3$) solve.
`mismip_plus_spinup.py` evolves the thickness with the depth-integrated
continuity equation (icepack2's mass balance driven by the depth-averaged
velocity), spins up to a steady grounding line, and then applies the MISMIP+
Ice1r sub-shelf melt parameterisation to drive grounding-line retreat.


## References

- Asay-Davis, W. K. et al. (2016). Experimental design for three interrelated marine ice sheet and ocean model intercomparison projects: MISMIP v.3 (MISMIP+), ISOMIP v.2 (ISOMIP+) and MISOMIP v.1 (MISOMIP1). *Geosci. Model Dev.*, 9, 2471-2497. doi:[10.5194/gmd-9-2471-2016](https://doi.org/10.5194/gmd-9-2471-2016)
- Cuffey, K. M. & Paterson, W. S. B. (2010). *The Physics of Glaciers* (4th ed.). Butterworth-Heinemann.
- Jouvet, G. (2015). A multilayer ice-flow model generalising the shallow shelf approximation. *J. Fluid Mech.*, 764, 26-51. doi:[10.1017/jfm.2014.689](https://doi.org/10.1017/jfm.2014.689)
- Pattyn, F. et al. (2008). Benchmark experiments for higher-order and full-Stokes ice sheet models (ISMIP-HOM). *The Cryosphere*, 2, 95-108. doi:[10.5194/tc-2-95-2008](https://doi.org/10.5194/tc-2-95-2008)
- Goldsby, D. L. & Kohlstedt, D. L. (2001). Superplastic deformation of ice. *J. Geophys. Res.*, 106, 11017-11030.
- Behn, M. D. et al. (2021). The role of grain size evolution in the rheology of ice. *The Cryosphere*, 15, 4589-4605. doi:[10.5194/tc-15-4589-2021](https://doi.org/10.5194/tc-15-4589-2021)
- Ranganathan, M. & Minchew, B. (2024). A modified viscous flow law for natural glacier ice. *PNAS*, 121(23). doi:[10.1073/pnas.2309788121](https://doi.org/10.1073/pnas.2309788121)

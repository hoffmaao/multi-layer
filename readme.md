# multilayer

A multilayer ice-flow model generalising the shallow shelf approximation, implemented in the dual form of [icepack2](https://github.com/icepack/icepack2).

This package implements the multilayer model described in [Jouvet (2015)](https://doi.org/10.1017/jfm.2014.689).
The idea is to split the ice column into a stack of thin layers, each with its own horizontal velocity.
Adjacent layers are coupled through interlayer shear tractions.
For a single layer the model reduces to the shallow shelf approximation (SSA); as the number of layers increases, the solution converges to the Stokes solution.

Unlike the SSA, which treats the ice as a plug, the multilayer model captures vertical shear.
This matters wherever basal drag is significant — near grounding lines, in ice streams, and in regions with strong bed topography.

The model uses the **dual formulation** from icepack2, where membrane stress and interlayer shear stress are explicit unknowns alongside velocity.
This formulation is well-posed at zero thickness, which enables natural handling of calving fronts and terminus evolution.


## Multilayer momentum balance

The core contribution is extending icepack2's dual-form SSA to multiple layers.
Each layer has three fields: velocity, membrane stress, and interlayer stress.
The interlayer stress has the same mathematical structure as a friction law — it relates the shear stress at an interface to the normalised velocity jump between layers.

The action functional is:

- **Viscous power** per layer (from icepack2, unchanged)
- **Interlayer power** at each interface (new: dissipation from vertical shear)
- **Basal stress power** at the bed (new: for frozen or sliding base)
- **Momentum balance** per layer (extended: includes interlayer coupling)

Taking the Gateaux derivative of the total action recovers all the weak-form equations.

## Composite rheology

Ice near the bed deforms primarily by dislocation creep ($n = 4$), while the colder upper ice deforms by grain boundary sliding ($n = 1.8$).
The standard Glen's law with $n = 3$ is an effective average of these two mechanisms (Goldsby & Kohlstedt 2001, Behn et al. 2021, Ranganathan & Minchew 2024).

The multilayer model naturally accommodates per-layer rheology.
We can assign different stress exponents and rate factors to each layer, representing the transition from warm basal ice (large recrystallised grains, dislocation creep) to cold upper ice (smaller grains, grain boundary sliding).

## ISMIP-HOM verification

The model has been tested against the ISMIP-HOM benchmark experiments (Pattyn et al. 2008):

- **Experiment B** (frozen base, sinusoidal bed bumps): surface velocities converge to the intercomparison mean as the number of layers increases from 1 to 5.
- **Experiment D** (sliding base, sinusoidal friction): all layer counts produce consistent results within the intercomparison spread.

Both uniform ($n = 3$) and composite ($n = 4$ / $n = 1.8$) rheologies have been verified across wavelengths from 10 to 160 km.


## Installation

The package depends on [Firedrake](https://firedrakeproject.org/) and [icepack2](https://github.com/icepack/icepack2).
With both installed:

```
git clone https://github.com/hoffmaao/multilayer.git
cd multilayer
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
```


## References

- Jouvet, G. (2015). A multilayer ice-flow model generalising the shallow shelf approximation. *J. Fluid Mech.*, 764, 26-51. doi:[10.1017/jfm.2014.689](https://doi.org/10.1017/jfm.2014.689)
- Pattyn, F. et al. (2008). Benchmark experiments for higher-order and full-Stokes ice sheet models (ISMIP-HOM). *The Cryosphere*, 2, 95-108. doi:[10.5194/tc-2-95-2008](https://doi.org/10.5194/tc-2-95-2008)
- Goldsby, D. L. & Kohlstedt, D. L. (2001). Superplastic deformation of ice. *J. Geophys. Res.*, 106, 11017-11030.
- Behn, M. D. et al. (2021). The role of grain size evolution in the rheology of ice. *The Cryosphere*, 15, 4589-4605. doi:[10.5194/tc-15-4589-2021](https://doi.org/10.5194/tc-15-4589-2021)
- Ranganathan, M. & Minchew, B. (2024). A modified viscous flow law for natural glacier ice. *PNAS*, 121(23). doi:[10.1073/pnas.2309788121](https://doi.org/10.1073/pnas.2309788121)

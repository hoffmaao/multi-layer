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

r"""Plot multilayer results against ISMIP-HOM intercomparison data.

Reads published model submissions from Pattyn et al. (2008) supplement
and overlays our multilayer dual-form results for experiments B and D.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from ismip_hom_b import solve_ismip_hom_b
from ismip_hom_d import solve_ismip_hom_d


DATA_DIR = Path("/tmp/ismip_hom_data/ismip_all")


def load_intercomparison_data(experiment, wavelength_km):
    r"""Load all model submissions for an ISMIP-HOM experiment.

    Returns arrays of surface vx profiles interpolated to a common x grid.

    Parameters
    ----------
    experiment : str, "b" or "d"
    wavelength_km : int, e.g. 80

    Returns
    -------
    x_common : ndarray, normalised x [0, 1]
    vx_surf_all : list of ndarrays, surface vx from each model
    vx_base_all : list of ndarrays, base vx from each model (D only)
    """
    tag = f"{experiment}{wavelength_km:03d}"
    x_common = np.linspace(0, 1, 101)
    vx_surf_all = []
    vx_base_all = []

    for model_dir in sorted(DATA_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        # Try combined file first (5+ columns: x, vx_s, vy_s, vx_b, vy_b)
        combined_files = list(model_dir.glob(f"*{tag}.txt"))
        if combined_files:
            for f in combined_files:
                try:
                    data = np.loadtxt(f)
                    if data.ndim == 1:
                        continue
                    if data.shape[1] >= 4:
                        x = data[:, 0]
                        vx_s = data[:, 1]
                        vx_b = data[:, 3] if data.shape[1] >= 5 else np.zeros_like(vx_s)
                        # Interpolate to common grid
                        vx_surf_all.append(np.interp(x_common, x, vx_s))
                        vx_base_all.append(np.interp(x_common, x, vx_b))
                except Exception:
                    continue

        # Try separate _surf/_base files (3 columns: x, vx, vy)
        surf_files = list(model_dir.glob(f"*{tag}_surf.txt"))
        base_files = list(model_dir.glob(f"*{tag}_base.txt"))
        if surf_files:
            for f in surf_files:
                try:
                    data = np.loadtxt(f)
                    if data.ndim == 1:
                        continue
                    if data.shape[1] >= 2:
                        x = data[:, 0]
                        vx_s = data[:, 1]
                        vx_surf_all.append(np.interp(x_common, x, vx_s))
                except Exception:
                    continue
        if base_files:
            for f in base_files:
                try:
                    data = np.loadtxt(f)
                    if data.ndim == 1:
                        continue
                    if data.shape[1] >= 2:
                        x = data[:, 0]
                        vx_b = data[:, 1]
                        vx_base_all.append(np.interp(x_common, x, vx_b))
                except Exception:
                    continue

    return x_common, vx_surf_all, vx_base_all


def plot_experiment_b():
    r"""Plot experiment B: surface vx for each wavelength."""
    wavelengths_km = [160, 80, 40, 20, 10]
    layer_counts = [1, 2, 3, 4, 5]
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C4", 5: "C5"}

    fig, axes = plt.subplots(1, len(wavelengths_km), figsize=(20, 4), sharey=False)
    fig.suptitle("ISMIP-HOM B: surface $v_x$ (m/yr) — uniform n=3, L=1–5", fontsize=14)

    for ax, Lkm in zip(axes, wavelengths_km):
        Lx = Lkm * 1e3

        # Load intercomparison data
        x_ref, vx_surf_all, _ = load_intercomparison_data("b", Lkm)
        if vx_surf_all:
            vx_arr = np.array(vx_surf_all)
            mean = np.mean(vx_arr, axis=0)
            std = np.std(vx_arr, axis=0)
            ax.fill_between(x_ref, mean - std, mean + std, alpha=0.3, color="gray",
                            label="Intercomp. mean $\\pm$ 1$\\sigma$")
            ax.plot(x_ref, mean, "k--", lw=1, label="Intercomp. mean")

        # Our multilayer results
        for nl in layer_counts:
            print(f"  B: L={Lkm}km, layers={nl}...", flush=True)
            u_surf, x_hat, _ = solve_ismip_hom_b(Lx, nl, nx=20)
            ax.plot(x_hat, u_surf, color=colors[nl], lw=1.5,
                    label=f"Multilayer L={nl}")

        ax.set_xlabel("$\\hat{{x}}$")
        ax.set_title(f"L = {Lkm} km")
        if ax is axes[0]:
            ax.set_ylabel("$v_x$ (m/yr)")

    axes[-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig("figures/ismip_hom_b.png", dpi=150)
    print(f"Saved figures/ismip_hom_b.png")


def plot_experiment_d():
    r"""Plot experiment D: surface and base vx for each wavelength."""
    wavelengths_km = [160, 80, 40, 20, 10]
    layer_counts = [1, 2, 3, 4, 5]
    colors = {1: "C0", 2: "C1", 3: "C2", 4: "C4", 5: "C5"}

    fig, axes = plt.subplots(2, len(wavelengths_km), figsize=(20, 7), sharey="row")
    fig.suptitle("ISMIP-HOM D: $v_x$ (m/yr) — uniform n=3, L=1–5", fontsize=14)

    for col, Lkm in enumerate(wavelengths_km):
        Lx = Lkm * 1e3
        ax_s = axes[0, col]
        ax_b = axes[1, col]

        # Load intercomparison data
        x_ref, vx_surf_all, vx_base_all = load_intercomparison_data("d", Lkm)
        if vx_surf_all:
            vx_s = np.array(vx_surf_all)
            mean_s = np.mean(vx_s, axis=0)
            std_s = np.std(vx_s, axis=0)
            ax_s.fill_between(x_ref, mean_s - std_s, mean_s + std_s,
                              alpha=0.3, color="gray")
            ax_s.plot(x_ref, mean_s, "k--", lw=1, label="Intercomp. mean")
        if vx_base_all:
            vx_b = np.array(vx_base_all)
            mean_b = np.mean(vx_b, axis=0)
            std_b = np.std(vx_b, axis=0)
            ax_b.fill_between(x_ref, mean_b - std_b, mean_b + std_b,
                              alpha=0.3, color="gray")
            ax_b.plot(x_ref, mean_b, "k--", lw=1)

        # Our multilayer results
        for nl in layer_counts:
            print(f"  D: L={Lkm}km, layers={nl}...", flush=True)
            u_surf, u_base, x_hat, _ = solve_ismip_hom_d(Lx, nl, nx=20)
            ax_s.plot(x_hat, u_surf, color=colors[nl], lw=1.5,
                      label=f"Multilayer L={nl}")
            ax_b.plot(x_hat, u_base, color=colors[nl], lw=1.5)

        ax_s.set_title(f"L = {Lkm} km")
        ax_b.set_xlabel("$\\hat{{x}}$")
        if col == 0:
            ax_s.set_ylabel("Surface $v_x$ (m/yr)")
            ax_b.set_ylabel("Base $v_x$ (m/yr)")

    axes[0, -1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig("figures/ismip_hom_d.png", dpi=150)
    print(f"Saved figures/ismip_hom_d.png")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    print("Plotting ISMIP-HOM Experiment B...")
    plot_experiment_b()

    print("\nPlotting ISMIP-HOM Experiment D...")
    plot_experiment_d()

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

r"""Plot composite rheology (n=4/1.8, A=40/1.0) vs uniform n=3 multilayer.

Compares the two-layer composite against uniform n=3 with L=1,2,3,4,5
layers, for ISMIP-HOM experiment B.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ismip_hom_b_composite import solve_ismip_hom_b_composite
from ismip_hom_b import solve_ismip_hom_b
from plot_ismip_hom import load_intercomparison_data


def main():
    wavelengths_km = [160, 80, 40, 20, 10]
    uniform_layers = [1, 2, 3, 4, 5]
    colors_uniform = {1: "C0", 2: "C1", 3: "C2", 4: "C4", 5: "C5"}

    fig, axes = plt.subplots(2, len(wavelengths_km), figsize=(20, 8))
    from ismip_hom_b_composite import LAYER_EXPONENTS, LAYER_PREFACTORS, LAYER_FRACTIONS
    n_b, n_t = LAYER_EXPONENTS
    a_b, a_t = LAYER_PREFACTORS
    pct = f"{LAYER_FRACTIONS[0]*100:.0f}%/{LAYER_FRACTIONS[1]*100:.0f}%"
    fig.suptitle(
        f"ISMIP-HOM B: composite (n={n_b} base, A={a_b:.0f} / n={n_t} top, A={a_t:.0f}, {pct}) vs uniform n=3",
        fontsize=13,
    )

    for col, Lkm in enumerate(wavelengths_km):
        Lx = Lkm * 1e3
        nx = min(max(20, int(40 * 80e3 / Lx)), 80)
        ax_vel = axes[0, col]
        ax_shear = axes[1, col]

        # Intercomparison data (gray band)
        x_ref, vx_surf_all, _ = load_intercomparison_data("b", Lkm)
        if vx_surf_all:
            vx_arr = np.array(vx_surf_all)
            mean = np.mean(vx_arr, axis=0)
            std = np.std(vx_arr, axis=0)
            ax_vel.fill_between(
                x_ref, mean - std, mean + std, alpha=0.2, color="gray",
                label="Intercomp. $\\pm 1\\sigma$",
            )
            ax_vel.plot(x_ref, mean, "k--", lw=0.8, label="Intercomp. mean")

        # Uniform n=3 multilayer: L=1 through 5
        for nl in uniform_layers:
            print(f"  B uniform L={nl}, Lx={Lkm}km...", flush=True)
            u_surf, x_hat, _ = solve_ismip_hom_b(Lx, nl, nx=nx)
            ax_vel.plot(x_hat, u_surf, color=colors_uniform[nl], lw=1.0,
                        alpha=0.8, label=f"L={nl} (n=3)")

        # Composite n=4/1.8
        print(f"  B composite, Lx={Lkm}km...", flush=True)
        u_top, u_bot, x_hat, _ = solve_ismip_hom_b_composite(Lx, nx=nx)
        ax_vel.plot(x_hat, u_top, "C3-", lw=2.5,
                    label="Composite top")
        ax_vel.plot(x_hat, u_bot, "C3--", lw=1.5,
                    label="Composite bot")

        # Shear panel
        shear = u_top - u_bot
        ax_shear.plot(x_hat, shear, "C3-", lw=2.0, label="Composite shear")
        ax_shear.axhline(0, color="gray", lw=0.5, ls=":")

        ax_vel.set_title(f"L = {Lkm} km")
        ax_shear.set_xlabel("$\\hat{x}$")
        if col == 0:
            ax_vel.set_ylabel("Surface $v_x$ (m/yr)")
            ax_shear.set_ylabel("$v_{top} - v_{bot}$ (m/yr)")

    axes[0, -1].legend(fontsize=5.5, loc="best", ncol=2)
    axes[1, -1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig("figures/ismip_hom_b_composite.png", dpi=150)
    print("Saved figures/ismip_hom_b_composite.png")


if __name__ == "__main__":
    main()

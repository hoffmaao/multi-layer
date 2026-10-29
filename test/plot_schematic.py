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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ismip_hom_b_composite import LAYER_FRACTIONS


# ISMIP-HOM B geometry
L_DOMAIN = 80e3
H_MEAN = 1000
ALPHA_DEG = 0.5


def make_geometry():
    omega = 2 * np.pi / L_DOMAIN
    nx = 80
    x = np.linspace(0, L_DOMAIN, nx)
    z_s = -x * np.tan(math.radians(ALPHA_DEG))
    z_b = z_s - H_MEAN + 500 * np.sin(omega * x)
    h = z_s - z_b
    return x, z_s, z_b, h, omega


def draw_layer(ax, x, z_bot, z_top, y0, y1, color, alpha_front=0.25, alpha_back=0.1):
    """Draw a 3D layer as front face, back face, and top surface."""
    # Front face
    layer_x = np.concatenate([x, x[::-1]]) / 1e3
    layer_z = np.concatenate([z_bot, z_top[::-1]])
    ax.add_collection3d(Poly3DCollection(
        [list(zip(layer_x, np.full_like(layer_x, y0 / 1e3), layer_z))],
        alpha=alpha_front, facecolor=color))
    # Back face
    ax.add_collection3d(Poly3DCollection(
        [list(zip(layer_x, np.full_like(layer_x, y1 / 1e3), layer_z))],
        alpha=alpha_back, facecolor=color))


def draw_top_surface(ax, x, z_top, y0, y1, color, alpha=0.12):
    """Draw the top surface connecting front and back."""
    nx = len(x)
    for i in range(0, nx - 1, 4):
        verts = [[(x[i] / 1e3, y0 / 1e3, z_top[i]),
                  (x[i + 1] / 1e3, y0 / 1e3, z_top[i + 1]),
                  (x[i + 1] / 1e3, y1 / 1e3, z_top[i + 1]),
                  (x[i] / 1e3, y1 / 1e3, z_top[i])]]
        ax.add_collection3d(Poly3DCollection(verts, alpha=alpha, facecolor=color))


def main():
    x, z_s, z_b, h, _ = make_geometry()
    nx = len(x)
    yw = L_DOMAIN * 0.12
    y0, y1 = 0, yw
    frac_bot = LAYER_FRACTIONS[0]

    configs = [
        {
            "frac_bot": None,
            "col_top": "steelblue", "col_bot": None,
        },
        {
            "frac_bot": 0.5,
            "col_top": "steelblue", "col_bot": "steelblue",
        },
        {
            "frac_bot": frac_bot,
            "col_top": "#4477AA", "col_bot": "coral",
        },
    ]

    fig = plt.figure(figsize=(20, 12))

    for idx, cfg in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        # Bed
        bed_x = np.concatenate([x, x[::-1]]) / 1e3
        bed_z = np.concatenate([z_b - 50, z_b[::-1]])
        ax.add_collection3d(Poly3DCollection(
            [list(zip(bed_x, np.full_like(bed_x, y0 / 1e3), bed_z))],
            alpha=0.4, facecolor="saddlebrown"))
        ax.plot(x / 1e3, np.full(nx, y0 / 1e3), z_b, "k-", lw=2, zorder=10)
        ax.plot(x / 1e3, np.full(nx, y1 / 1e3), z_b, color="gray", lw=1)
        ax.plot(x / 1e3, np.full(nx, y0 / 1e3), z_s, "k-", lw=1.5, zorder=10)
        ax.plot(x / 1e3, np.full(nx, y1 / 1e3), z_s, color="gray", lw=1)

        fb = cfg["frac_bot"]

        if fb is None:
            # Single layer
            draw_layer(ax, x, z_b, z_s, y0, y1, cfg["col_top"])
            draw_top_surface(ax, x, z_s, y0, y1, cfg["col_top"])
        else:
            z_int = z_b + h * fb
            ax.plot(x / 1e3, np.full(nx, y0 / 1e3), z_int, "k--", lw=1.5, zorder=10)
            ax.plot(x / 1e3, np.full(nx, y1 / 1e3), z_int, color="gray", lw=0.8, ls="--")

            draw_layer(ax, x, z_b, z_int, y0, y1, cfg["col_bot"], alpha_front=0.3)
            draw_layer(ax, x, z_int, z_s, y0, y1, cfg["col_top"])
            draw_top_surface(ax, x, z_s, y0, y1, cfg["col_top"])

        ax.set_xlabel("x (km)", fontsize=9, labelpad=5)
        ax.set_ylabel("y (km)", fontsize=9, labelpad=5)
        ax.set_zlabel("z (m)", fontsize=9, labelpad=5)
        ax.view_init(elev=22, azim=-55)
        ax.set_xlim(0, L_DOMAIN / 1e3)
        ax.set_ylim(0, yw / 1e3)
        ax.set_zlim(z_b.min() - 100, z_s.max() + 200)
        ax.set_yticks([0, 4, 8])
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(None, zoom=0.9)

    plt.subplots_adjust(wspace=0.05, right=0.96)
    fig.savefig("figures/ismip_hom_schematic.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
    print("Saved figures/ismip_hom_schematic.png")


if __name__ == "__main__":
    main()

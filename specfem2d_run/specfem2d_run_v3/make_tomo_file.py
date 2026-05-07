#!/usr/bin/env python3
"""
Generate SpecFEM2D TOMOGRAPHY_FILE ./DATA/tomo_file.xyz for src0/src1.

Format used here: x(m) z(m) rho(kg/m^3) vp(m/s) vs(m/s)
This is a common SpecFEM2D tomography xyz format for (x,z) gridded models.

We use a background vp=3000 m/s and an elliptical low-velocity anomaly vp=2000 m/s,
with vs=0 for acoustic/fluid.
"""

from __future__ import annotations

import os
import numpy as np


def generate_tomo(path: str,
                  x_max: float = 1500.0,
                  z_max: float = 500.0,
                  dx: float = 10.0,
                  dz: float = 10.0,
                  rho: float = 2700.0,
                  vp_bg: float = 3000.0,
                  vp_anom: float = 2000.0,
                  xc: float = 800.0,
                  zc: float = 250.0,
                  a: float = 180.0,
                  b: float = 100.0) -> None:
    xs = np.arange(0.0, x_max + 0.5 * dx, dx)
    zs = np.arange(0.0, z_max + 0.5 * dz, dz)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    g = ((X - xc) ** 2) / (a ** 2) + ((Z - zc) ** 2) / (b ** 2)
    vp = np.where(g <= 1.0, vp_anom, vp_bg)
    vs = np.zeros_like(vp)
    rho_arr = rho * np.ones_like(vp)
    out = np.column_stack([X.ravel(), Z.ravel(), rho_arr.ravel(), vp.ravel(), vs.ravel()])
    np.savetxt(path, out, fmt="%.6f")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    for src in ["src0", "src1"]:
        data_dir = os.path.join(root, src, "DATA")
        os.makedirs(data_dir, exist_ok=True)
        out_path = os.path.join(data_dir, "tomo_file.xyz")
        generate_tomo(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


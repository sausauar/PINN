# SpecFem2D — PINN 2src Setup  (auto-generated)

## Parameters
- Domain: 1.50 x 0.50 km
- Grid: 150 x 50  (dx=10.0m, dz=10.0m)
- dt=5e-05s,  Nstep=14000  →  T_total=0.700s
- f0=20.0 Hz
- Source 0: (400.0 m, 85.0 m)
- Source 1: (1100.0 m, 85.0 m)
- Receivers: 20  z=415.0 m  x=[100.0..1400.0] m
- PML thickness: 10 elements
- Wavefield snapshots every 100 steps

## Materials
  mat 1: rho=2700  vp=3000 m/s  (acoustic)
  mat 2: rho=2700  vp=2000 m/s  (acoustic)

## Regions (element-index, 1-based)
  ix=[1..150]  iz=[1..15]  mat=1
  ix=[1..150]  iz=[37..50]  mat=1
  ix=[1..55]  iz=[16..36]  mat=1
  ix=[56..96]  iz=[16..36]  mat=2
  ix=[97..150]  iz=[16..36]  mat=1

## Usage
```bash
export SPECFEM_BIN=/path/to/specfem2d/bin
bash run_specfem.sh
python3 convert_specfem_to_pinn.py
```

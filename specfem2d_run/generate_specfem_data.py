#!/usr/bin/env python3
"""
generate_specfem_setup.py
Генерирует полный набор файлов SpecFem2D (src0/, src1/) с правильным форматом Par_file.

Ключевые исправления относительно v2:
  - Материалы и регионы — ТОЛЬКО в Par_file, Mesh_Par_file НЕ используется
  - Правильные имена параметров: NGNOD, NTSTEP_BETWEEN_OUTPUT_SAMPLE, ROTATE_PML_ANGLE
  - interfacesfile — в Par_file с абсолютным или относительным путём
  - absorbbottom/absorbright/absorbtop/absorbleft — в Par_file
  - Убраны несуществующие параметры: subsamp_seismos, Q0, freq0, TURN_ANISOTROPY_ON,
    add_Biot_absorption, NTSSTEP_BETWEEN_OUTPUT_WAVE_DUMPS (→ NTSTEP_BETWEEN_OUTPUT_IMAGES)

Запуск:
    python3 generate_specfem_setup.py [--outdir specfem2d_run]
"""

import os
import sys
import argparse
import textwrap
import numpy as np


# ─────────────────────────────────────────────
#  КОНФИГУРАЦИЯ  (правьте здесь)
# ─────────────────────────────────────────────

CFG = dict(
    # ── Домен ────────────────────────────────
    xmin       = 0.0,       # м
    xmax       = 1500.0,    # м
    zmax       = 500.0,     # м  (zmin всегда 0)
    nx         = 150,       # элементов по X
    nz         = 50,        # элементов по Z

    # ── Время ────────────────────────────────
    NSTEP      = 14000,
    DT         = 5e-5,      # с

    # ── Источники ────────────────────────────
    sources    = [
        dict(xs=400.0,  zs=85.0,  f0=20.0, tshift=0.06, factor=1e10),
        dict(xs=1100.0, zs=85.0,  f0=20.0, tshift=0.06, factor=1e10),
    ],

    # ── Приёмники ────────────────────────────
    n_receivers   = 20,
    x_rec_start   = 100.0,   # м
    x_rec_end     = 1400.0,  # м
    z_rec          = 415.0,   # м

    # ── Материалы ────────────────────────────
    # Формат: (rho, vp, vs)  vs=0 → акустический
    materials = [
        (2700.0, 3000.0, 0.0),   # материал 1 — фон
        (2700.0, 2000.0, 0.0),   # материал 2 — аномалия
    ],

    # ── Регионы сетки ────────────────────────
    # (ix_min, ix_max, iz_min, iz_max, mat_id)  — 1-based, в единицах элементов
    regions = [
        (1,   150,  1,  15,  1),   # нижний слой
        (1,   150, 37,  50,  1),   # верхний слой
        (1,    55, 16,  36,  1),   # фон слева
        (56,   96, 16,  36,  2),   # аномалия
        (97,  150, 16,  36,  1),   # фон справа
    ],

    # ── PML ──────────────────────────────────
    NELEM_PML_THICKNESS = 10,

    # ── Снимки волнового поля ────────────────
    NTSTEP_BETWEEN_OUTPUT_IMAGES = 100,   # шаг записи wavefield dumps
    output_wavefield_dumps       = True,
    imagetype_wavefield_dumps    = 1,     # 1=displ vector

    # ── Прочее ───────────────────────────────
    NTSTEP_BETWEEN_OUTPUT_INFO   = 2000,
    NTSTEP_BETWEEN_OUTPUT_SEISMOS = 14000,
)


# ─────────────────────────────────────────────
#  ГЕНЕРАТОРЫ ФАЙЛОВ
# ─────────────────────────────────────────────

def _fortran_float(val: float) -> str:
    """Convert Python float to Fortran double-precision notation (e.g. 5e-5 → 5.d-5)."""
    s = f"{val:g}"          # '5e-05' or '0.001' etc.
    if "e" in s:
        mantissa, exp = s.split("e")
        exp_int = int(exp)
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}d{exp_int:+d}"
    else:
        return s + "d0"


def par_file(cfg: dict, src_idx: int, interfaces_path: str) -> str:
    """Генерирует Par_file точно в формате официального SPECFEM2D."""

    nbmodels = len(cfg["materials"])
    nbregions = len(cfg["regions"])

    # Строки материалов
    mat_lines = []
    for i, (rho, vp, vs) in enumerate(cfg["materials"], 1):
        mat_lines.append(
            f"{i} 1 {rho:.1f} {vp:.1f} {vs:.4f}  0 0  9999. 9999.  0 0 0 0 0 0"
        )

    # Строки регионов
    reg_lines = [
        f"{r[0]} {r[1]}  {r[2]} {r[3]}  {r[4]}"
        for r in cfg["regions"]
    ]

    wf_flag = ".true." if cfg["output_wavefield_dumps"] else ".false."
    dt_str  = _fortran_float(cfg["DT"])

    text = f"""\
#-----------------------------------------------------------
#
# Simulation input parameters
#
#-----------------------------------------------------------

# title of job
title                           = PINN_2src_{src_idx}

# forward or adjoint simulation
# 1 = forward, 2 = adjoint, 3 = both simultaneously
SIMULATION_TYPE                 = 1
NOISE_TOMOGRAPHY                = 0
SAVE_FORWARD                    = .false.

# parameters concerning partitioning
NPROC                           = 1

# time step parameters
NSTEP                           = {cfg["NSTEP"]}
DT                              = {dt_str}

# time stepping  1=Newmark  2=LDDRK4-6  3=RK4
time_stepping_scheme            = 1

# P-SV or SH/membrane waves
P_SV                            = .true.

# axisymmetric (2.5D) or Cartesian planar (2D)
AXISYM                          = .false.

#-----------------------------------------------------------
#
# Mesh
#
#-----------------------------------------------------------

PARTITIONING_TYPE               = 3
NGNOD                           = 9
setup_with_binary_database      = 0
MODEL                           = default
SAVE_MODEL                      = default
TOMOGRAPHY_FILE                 = dummy

#-----------------------------------------------------------
#
# Attenuation
#
#-----------------------------------------------------------

ATTENUATION_VISCOELASTIC        = .false.
ATTENUATION_VISCOACOUSTIC       = .false.
N_SLS                           = 3
ATTENUATION_f0_REFERENCE        = {cfg["sources"][src_idx]["f0"]:.3f}
READ_VELOCITIES_AT_f0           = .false.
USE_SOLVOPT                     = .false.
ATTENUATION_PORO_FLUID_PART     = .false.
Q0_poroelastic                  = 1
freq0_poroelastic               = 10
ATTENUATION_PERMITTIVITY        = .false.
ATTENUATION_CONDUCTIVITY        = .false.
f0_electromagnetic              = 1.d9
UNDO_ATTENUATION_AND_OR_PML     = .false.
NT_DUMP_ATTENUATION             = 500
NO_BACKWARD_RECONSTRUCTION      = .false.

#-----------------------------------------------------------
#
# Sources
#
#-----------------------------------------------------------

NSOURCES                        = 1
force_normal_to_surface         = .false.
initialfield                    = .false.
add_Bielak_conditions_bottom    = .false.
add_Bielak_conditions_right     = .false.
add_Bielak_conditions_top       = .false.
add_Bielak_conditions_left      = .false.
ACOUSTIC_FORCING                = .false.
noise_source_time_function_type = 4
write_moving_sources_database   = .false.
PRINT_SOURCE_TIME_FUNCTION      = .true.

#-----------------------------------------------------------
#
# Receivers
#
#-----------------------------------------------------------

seismotype                      = 1
NTSTEP_BETWEEN_OUTPUT_SEISMOS   = {cfg["NTSTEP_BETWEEN_OUTPUT_SEISMOS"]}
NTSTEP_BETWEEN_OUTPUT_SAMPLE    = 1
USE_TRICK_FOR_BETTER_PRESSURE   = .false.
USER_T0                         = 0.0d0
save_ASCII_seismograms          = .true.
save_binary_seismograms_single  = .false.
save_binary_seismograms_double  = .false.
SU_FORMAT                       = .false.
use_existing_STATIONS           = .true.
nreceiversets                   = 1
anglerec                        = 0.d0
rec_normal_to_surface           = .false.

#-----------------------------------------------------------
#
# adjoint kernel outputs
#
#-----------------------------------------------------------

SAVE_ASCII_KERNELS              = .true.
SAVE_KERNEL_WEIGHTS             = .false.
NTSTEP_BETWEEN_COMPUTE_KERNELS  = 1
APPROXIMATE_HESS_KL             = .false.

#-----------------------------------------------------------
#
# Boundary conditions
#
#-----------------------------------------------------------

PML_BOUNDARY_CONDITIONS         = .true.
NELEM_PML_THICKNESS             = {cfg["NELEM_PML_THICKNESS"]}
ROTATE_PML_ACTIVATE             = .false.
ROTATE_PML_ANGLE                = 0.
K_MIN_PML                       = 1.0d0
K_MAX_PML                       = 1.0d0
damping_change_factor_acoustic  = 0.5d0
damping_change_factor_elastic   = 1.0d0
PML_PARAMETER_ADJUSTMENT        = .false.
STACEY_ABSORBING_CONDITIONS     = .false.
ADD_PERIODIC_CONDITIONS         = .false.
PERIODIC_HORIZ_DIST             = {cfg["xmax"]:.1f}d0

#-----------------------------------------------------------
#
# MESHING - Velocity and density models
#
#-----------------------------------------------------------

read_external_mesh              = .false.

#-----------------------------------------------------------
#
# PARAMETERS FOR INTERNAL MESHING
#
#-----------------------------------------------------------

# material properties
nbmodels                        = {nbmodels}
# model_number 1 rho Vp Vs 0 0 QKappa Qmu 0 0 0 0 0 0
"""

    for line in mat_lines:
        text += line + "\n"

    text += f"""
# interfaces file
interfacesfile                  = {interfaces_path}

# geometry of the model
xmin                            = {cfg["xmin"]:.1f}d0
xmax                            = {cfg["xmax"]:.1f}d0
nx                              = {cfg["nx"]}

# absorbing boundary parameters
absorbbottom                    = .true.
absorbright                     = .true.
absorbtop                       = .true.
absorbleft                      = .true.

# regions
nbregions                       = {nbregions}
# ix_min  ix_max  iz_min  iz_max  material_number
"""

    for line in reg_lines:
        text += line + "\n"

    text += f"""
#-----------------------------------------------------------
#
# Display parameters
#
#-----------------------------------------------------------

NTSTEP_BETWEEN_OUTPUT_INFO      = {cfg["NTSTEP_BETWEEN_OUTPUT_INFO"]}
SAVE_MESH_FILES                 = .true.
output_grid_Gnuplot             = .false.
output_grid_ASCII               = .false.
OUTPUT_ENERGY                   = .false.
NTSTEP_BETWEEN_OUTPUT_ENERGY    = 10
COMPUTE_INTEGRATED_ENERGY_FIELD = .false.

#-----------------------------------------------------------
#
# Movies/images/snapshots visualizations
#
#-----------------------------------------------------------

NTSTEP_BETWEEN_OUTPUT_IMAGES    = {cfg["NTSTEP_BETWEEN_OUTPUT_IMAGES"]}
cutsnaps                        = 1.
output_color_image              = .false.
imagetype_JPEG                  = 2
factor_subsample_image          = 1.0d0
USE_CONSTANT_MAX_AMPLITUDE      = .false.
CONSTANT_MAX_AMPLITUDE_TO_USE   = 1.d4
POWER_DISPLAY_COLOR             = 0.30d0
DRAW_SOURCES_AND_RECEIVERS      = .true.
DRAW_WATER_IN_BLUE              = .true.
USE_SNAPSHOT_NUMBER_IN_FILENAME = .false.
output_postscript_snapshot      = .false.
imagetype_postscript            = 1
meshvect                        = .true.
modelvect                       = .false.
boundvect                       = .true.
interpol                        = .true.
pointsdisp                      = 6
subsamp_postscript              = 1
sizemax_arrows                  = 1.d0
US_LETTER                       = .false.
output_wavefield_dumps          = {wf_flag}
imagetype_wavefield_dumps       = {cfg["imagetype_wavefield_dumps"]}
use_binary_for_wavefield_dumps  = .false.

#-----------------------------------------------------------

NUMBER_OF_SIMULTANEOUS_RUNS     = 1
BROADCAST_SAME_MESH_AND_MODEL   = .true.
GPU_MODE                        = .false.
"""
    return text


def source_file(cfg: dict, src_idx: int) -> str:
    s = cfg["sources"][src_idx]
    return f"""\
source_surf                     = .false.
xs                              = {s["xs"]:.2f}
zs                              = {s["zs"]:.2f}
source_type                     = 1
time_function_type              = 1
name_of_source_file             = NONE
burst_band_width                = 0.
f0                              = {s["f0"]:.4f}
tshift                          = {s["tshift"]:.4f}
anglesource                     = 0.
Mxx                             = 1.d0
Mzz                             = 1.d0
Mxz                             = 0.d0
factor                          = {s["factor"]:.1e}
vx                              = 0.d0
vz                              = 0.d0
"""


def interfaces_file(cfg: dict) -> str:
    xmin, xmax = cfg["xmin"], cfg["xmax"]
    zmax = cfg["zmax"]
    nx, nz = cfg["nx"], cfg["nz"]
    return f"""\
# Number of interfaces
2

# Interface 1 — bottom (z=0)
2
{xmin:.1f}d0  0.d0
{xmax:.1f}d0  0.d0

# Interface 2 — top / free surface (z={zmax:.1f})
2
{xmin:.1f}d0  {zmax:.1f}d0
{xmax:.1f}d0  {zmax:.1f}d0

# nb of spectral elements in Z for the single layer between the two interfaces
{nz}

"""


def stations_file(cfg: dict) -> str:
    xs = np.linspace(cfg["x_rec_start"], cfg["x_rec_end"], cfg["n_receivers"])
    z  = cfg["z_rec"]
    lines = [
        f"S{i+1:04d}    AA    {x:.2f}    {z:.2f}    0.0    0.0"
        for i, x in enumerate(xs)
    ]
    return "\n".join(lines) + "\n"


def readme_file(cfg: dict) -> str:
    srcs = cfg["sources"]
    T_total = cfg["NSTEP"] * cfg["DT"]
    dx = (cfg["xmax"] - cfg["xmin"]) / cfg["nx"]
    dz = cfg["zmax"] / cfg["nz"]
    return f"""\
# SpecFem2D — PINN 2src Setup  (auto-generated)

## Параметры
- Домен: {cfg["xmax"]/1000:.1f} x {cfg["zmax"]/1000:.1f} km
- Сетка: {cfg["nx"]}x{cfg["nz"]}  (dx={dx:.0f}m, dz={dz:.0f}m)
- dt={cfg["DT"]:g}s,  Nstep={cfg["NSTEP"]}  →  T={T_total:.2f}s
- f0={srcs[0]["f0"]:.1f} Hz
- Источник 0: ({srcs[0]["xs"]}m, {srcs[0]["zs"]}m)
- Источник 1: ({srcs[1]["xs"]}m, {srcs[1]["zs"]}m)
- Датчики: {cfg["n_receivers"]} шт,  z={cfg["z_rec"]}m,  x=[{cfg["x_rec_start"]}..{cfg["x_rec_end"]}]m
- PML: {cfg["NELEM_PML_THICKNESS"]} элементов
- Материалов: {len(cfg["materials"])},  регионов: {len(cfg["regions"])}

## Материалы
"""
    # add material lines
    for i, (rho, vp, vs) in enumerate(cfg["materials"], 1):
        kind = "акустический" if vs == 0 else "упругий"
        result = f"- Мат.{i}: rho={rho:.0f}, vp={vp:.0f} m/s ({kind})\n"
    return readme_file.__doc__ or ""


def make_readme(cfg: dict) -> str:
    srcs = cfg["sources"]
    T_total = cfg["NSTEP"] * cfg["DT"]
    dx = (cfg["xmax"] - cfg["xmin"]) / cfg["nx"]
    dz = cfg["zmax"] / cfg["nz"]
    mat_lines = ""
    for i, (rho, vp, vs) in enumerate(cfg["materials"], 1):
        kind = "acoustic" if vs == 0 else "elastic"
        mat_lines += f"  mat {i}: rho={rho:.0f}  vp={vp:.0f} m/s  ({kind})\n"
    reg_lines = ""
    for r in cfg["regions"]:
        reg_lines += f"  ix=[{r[0]}..{r[1]}]  iz=[{r[2]}..{r[3]}]  mat={r[4]}\n"

    snap_steps = [
        cfg["NTSTEP_BETWEEN_OUTPUT_IMAGES"] * k
        for k in range(1, 4)
        if cfg["NTSTEP_BETWEEN_OUTPUT_IMAGES"] * k <= cfg["NSTEP"]
    ]
    return f"""\
# SpecFem2D — PINN 2src Setup  (auto-generated)

## Parameters
- Domain: {cfg["xmax"]/1000:.2f} x {cfg["zmax"]/1000:.2f} km
- Grid: {cfg["nx"]} x {cfg["nz"]}  (dx={dx:.1f}m, dz={dz:.1f}m)
- dt={cfg["DT"]:g}s,  Nstep={cfg["NSTEP"]}  →  T_total={T_total:.3f}s
- f0={srcs[0]["f0"]:.1f} Hz
- Source 0: ({srcs[0]["xs"]} m, {srcs[0]["zs"]} m)
- Source 1: ({srcs[1]["xs"]} m, {srcs[1]["zs"]} m)
- Receivers: {cfg["n_receivers"]}  z={cfg["z_rec"]} m  x=[{cfg["x_rec_start"]}..{cfg["x_rec_end"]}] m
- PML thickness: {cfg["NELEM_PML_THICKNESS"]} elements
- Wavefield snapshots every {cfg["NTSTEP_BETWEEN_OUTPUT_IMAGES"]} steps

## Materials
{mat_lines}
## Regions (element-index, 1-based)
{reg_lines}
## Usage
```bash
export SPECFEM_BIN=/path/to/specfem2d/bin
bash run_specfem.sh
python3 convert_specfem_to_pinn.py
```
"""


# ─────────────────────────────────────────────
#  СБОРКА ДИРЕКТОРИЙ
# ─────────────────────────────────────────────

def build(outdir: str, cfg: dict):
    os.makedirs(outdir, exist_ok=True)

    # Интерфейсы — общий файл, на который ссылается Par_file
    ifaces_content = interfaces_file(cfg)

    for src_idx in range(len(cfg["sources"])):
        src_dir  = os.path.join(outdir, f"src{src_idx}")
        data_dir = os.path.join(src_dir, "DATA")
        out_dir  = os.path.join(src_dir, "OUTPUT_FILES")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(out_dir,  exist_ok=True)

        # interfaces.dat — кладём в DATA/
        ifaces_path = os.path.join(data_dir, "interfaces.dat")
        write(ifaces_path, ifaces_content)

        # Par_file ссылается на interfaces.dat относительно DATA/
        pf = par_file(cfg, src_idx, interfaces_path="interfaces.dat")
        write(os.path.join(data_dir, "Par_file"), pf)

        write(os.path.join(data_dir, "SOURCE"),   source_file(cfg, src_idx))
        write(os.path.join(data_dir, "STATIONS"), stations_file(cfg))

    # run_specfem.sh
    run_sh = _run_script(len(cfg["sources"]))
    write(os.path.join(outdir, "run_specfem.sh"), run_sh)
    os.chmod(os.path.join(outdir, "run_specfem.sh"), 0o755)

    # README
    write(os.path.join(outdir, "README.md"), make_readme(cfg))


def write(path: str, content: str):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"  wrote: {path}")


def _run_script(n_sources: int) -> str:
    src_calls = ""
    for i in range(n_sources):
        src = CFG["sources"][i]
        src_calls += f'run_source src{i} "Source {i} — x={src["xs"]}m, z={src["zs"]}m"\n'
    return f"""\
#!/bin/bash
# run_specfem.sh  (auto-generated)
# Usage:
#   export SPECFEM_BIN=/path/to/specfem2d/bin
#   bash run_specfem.sh

SPECFEM_BIN="${{SPECFEM_BIN:-$HOME/specfem2d/bin}}"
XMESHFEM="$SPECFEM_BIN/xmeshfem2D"
XSPECFEM="$SPECFEM_BIN/xspecfem2D"

if [ ! -f "$XMESHFEM" ]; then
    echo "ERROR: xmeshfem2D not found at $SPECFEM_BIN"
    echo "Install: git clone --recursive https://github.com/SPECFEM/specfem2d.git"
    echo "         cd specfem2d && ./configure && make all"
    exit 1
fi

set -e

run_source() {{
    local SRCDIR=$1
    local LABEL=$2
    echo ""; echo "=== $LABEL ==="; echo ""
    cd "$SRCDIR"
    mkdir -p OUTPUT_FILES
    "$XMESHFEM"
    "$XSPECFEM"
    echo "  wavefields: $(ls OUTPUT_FILES/wavefield*.txt 2>/dev/null | wc -l)"
    echo "  seismograms: $(ls OUTPUT_FILES/*.semd 2>/dev/null | wc -l)"
    cd ..
}}

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTDIR"

{src_calls}
echo "=== All done. Next: python3 convert_specfem_to_pinn.py ==="
"""


# ─────────────────────────────────────────────
#  ВАЛИДАЦИЯ
# ─────────────────────────────────────────────

def validate(cfg: dict):
    errors = []
    nx, nz = cfg["nx"], cfg["nz"]
    for r in cfg["regions"]:
        if r[0] < 1 or r[1] > nx:
            errors.append(f"Region ix [{r[0]}..{r[1]}] out of range [1..{nx}]")
        if r[2] < 1 or r[3] > nz:
            errors.append(f"Region iz [{r[2]}..{r[3]}] out of range [1..{nz}]")
        if r[4] < 1 or r[4] > len(cfg["materials"]):
            errors.append(f"Region material {r[4]} not defined")
    for i, s in enumerate(cfg["sources"]):
        if not (cfg["xmin"] < s["xs"] < cfg["xmax"]):
            errors.append(f"Source {i} xs={s['xs']} outside domain [{cfg['xmin']}, {cfg['xmax']}]")
        if not (0 < s["zs"] < cfg["zmax"]):
            errors.append(f"Source {i} zs={s['zs']} outside domain [0, {cfg['zmax']}]")
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  [ERR] {e}")
        sys.exit(1)
    else:
        print("  [OK] Validation passed")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="specfem2d_run_v3",
                   help="Output directory (default: specfem2d_run_v3)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print Par_file for src0 to stdout without writing files")
    args = p.parse_args()

    print("=== SpecFem2D setup generator (v3 — correct Par_file format) ===")
    validate(CFG)

    if args.dry_run:
        print("\n--- Par_file (src0) ---")
        print(par_file(CFG, 0, "interfaces.dat"))
        print("\n--- SOURCE (src0) ---")
        print(source_file(CFG, 0))
        print("\n--- interfaces.dat ---")
        print(interfaces_file(CFG))
        return

    print(f"\nOutput directory: {args.outdir}")
    build(args.outdir, CFG)

    # Summary
    T_total = CFG["NSTEP"] * CFG["DT"]
    dx = (CFG["xmax"] - CFG["xmin"]) / CFG["nx"]
    print(f"""
=== Summary ===
  Domain:    {CFG['xmax']/1000:.1f} x {CFG['zmax']/1000:.1f} km
  Grid:      {CFG['nx']} x {CFG['nz']}  (dx={dx:.0f}m)
  Time:      {CFG['NSTEP']} steps x {CFG['DT']:g}s = {T_total:.2f}s
  Sources:   {len(CFG['sources'])}
  Receivers: {CFG['n_receivers']}
  PML:       {CFG['NELEM_PML_THICKNESS']} elements

Next steps:
  export SPECFEM_BIN=$HOME/specfem2d/bin
  cd {args.outdir}
  bash run_specfem.sh
  python3 ../convert_specfem_to_pinn.py
""")


if __name__ == "__main__":
    main()
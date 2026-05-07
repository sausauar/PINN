#!/bin/bash
# run_specfem.sh  (auto-generated)
# Usage:
#   export SPECFEM_BIN=/path/to/specfem2d/bin
#   bash run_specfem.sh

SPECFEM_BIN="${SPECFEM_BIN:-$HOME/specfem2d/bin}"
XMESHFEM="$SPECFEM_BIN/xmeshfem2D"
XSPECFEM="$SPECFEM_BIN/xspecfem2D"

if [ ! -f "$XMESHFEM" ]; then
    echo "ERROR: xmeshfem2D not found at $SPECFEM_BIN"
    echo "Install: git clone --recursive https://github.com/SPECFEM/specfem2d.git"
    echo "         cd specfem2d && ./configure && make all"
    exit 1
fi

set -e

run_source() {
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
}

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTDIR"

run_source src0 "Source 0 — x=400.0m, z=85.0m"
run_source src1 "Source 1 — x=1100.0m, z=85.0m"

echo "=== All done. Next: python3 convert_specfem_to_pinn.py ==="

#!/bin/bash
# run_specfem.sh
# Запускает SpecFem2D для двух источников (src0, src1)
# Использование:
#   export SPECFEM_BIN=/path/to/specfem2d/bin
#   bash run_specfem.sh

SPECFEM_BIN="${SPECFEM_BIN:-$HOME/specfem2d/bin}"
XMESHFEM="$SPECFEM_BIN/xmeshfem2D"
XSPECFEM="$SPECFEM_BIN/xspecfem2D"

if [ ! -f "$XMESHFEM" ]; then
    echo "=============================================="
    echo " SpecFem2D не найден в: $SPECFEM_BIN"
    echo ""
    echo " Установка:"
    echo "   git clone --recursive https://github.com/SPECFEM/specfem2d.git"
    echo "   cd specfem2d && ./configure && make all"
    echo "   export SPECFEM_BIN=\$PWD/bin"
    echo "=============================================="
    exit 1
fi

set -e

run_source() {
    local SRCDIR=$1
    local LABEL=$2
    echo ""
    echo "=========================================="
    echo " $LABEL"
    echo "=========================================="

    cd "$SRCDIR"
    mkdir -p OUTPUT_FILES

    echo " → xmeshfem2D ..."
    "$XMESHFEM"

    echo " → xspecfem2D ..."
    "$XSPECFEM"

    echo " DONE: $LABEL"
    echo " Wavefield files: $(ls OUTPUT_FILES/wavefield*.txt 2>/dev/null | wc -l)"
    echo " Seismogram files: $(ls OUTPUT_FILES/*.semd 2>/dev/null | wc -l)"
    cd ..
}

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPTDIR"

run_source src0 "Source 0 — x=400m, z=85m"
run_source src1 "Source 1 — x=1100m, z=85m"

echo ""
echo "=========================================="
echo " Обе симуляции завершены!"
echo " Следующий шаг:"
echo "   python3 convert_specfem_to_pinn.py"
echo "=========================================="

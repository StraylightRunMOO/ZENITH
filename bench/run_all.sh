#!/bin/sh
set -eu

# Run from the build directory after building zenith_bench.
# Usage: sh ../bench/run_all.sh ./zenith_bench ../results [quick|full|perf] [tag]
#
#   quick  N=4096  D=64   (default)
#   full   N=16384 D=96
#   perf   N=8192  D=256  (FFTW-friendly, OpenMP-friendly scans)
#
# Writes $OUT/zenith_${dataset}_${tag}.{csv,log}.
# tag defaults to the mode name so quick/full/perf do not overwrite each other.
# Shipped files:
#   quick + FFTW3 + OpenMP  →  tag o3_fftw_omp
#   perf  + FFTW3 + OpenMP  →  tag perf
#   perf  + dense DCT, 1 thread →  tag perf_o3_serial

BIN=${1:-./zenith_bench}
OUT=${2:-../results}
MODE=${3:-quick}
TAG=${4:-$MODE}
mkdir -p "$OUT"

case "$MODE" in
    quick) FLAG=--quick ;;
    full)  FLAG=--full ;;
    perf)  FLAG=--perf ;;
    *) echo "mode must be quick, full, or perf" >&2; exit 2 ;;
esac

export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

for dataset in clustered rough gaussian; do
    "$BIN" "$FLAG" --dataset "$dataset" --out "$OUT/zenith_${dataset}_${TAG}.csv" \
        > "$OUT/zenith_${dataset}_${TAG}.log" 2>&1
    echo "wrote $OUT/zenith_${dataset}_${TAG}.csv"
done

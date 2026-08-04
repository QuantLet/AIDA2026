#!/usr/bin/env bash
# Raised-count runs for the M5 Max, launched 2026-07-31.
#
# Two sampling stages sit on a quantile noise floor set by their draw count. On the
# laptop the counts were what fitted; here they are what the method needs:
#
#   09_llmtime.py   200 -> 1000 draws   ten draws below the 1% quantile, not two
#   02_chronos_lab  500 -> 2000 samples twenty draws below the 1% quantile, not five
#
# Both stages checkpoint every forecast and both tag their output with the draw count,
# so this script is interruptible and cannot overwrite the certified lower-count runs.
#
#   ./run_raised.sh llmtime     # MPS, ~5.5 h per asset
#   ./run_raised.sh t5          # CPU,  ~1 h per asset, runs alongside the above
set -u

PY="${PY:-$HOME/.venvs/aida-2026/bin/python}"
ASSETS="${ASSETS:-SPX DAX N225 GOLD BTC}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/logs"
cd "$HERE"

case "${1:-llmtime}" in
llmtime)
  # fp16 rather than fp32: 2.6x the throughput, and measured against fp32 on four
  # dates the VaR(1%) differs by at most 0.010 points on levels of 1.7 to 2.6,
  # which is far inside the sampling noise of a 1% quantile taken from 1000 draws.
  for A in $ASSETS; do
    echo "=== llmtime $A $(date '+%F %T') ==="
    $PY src/09_llmtime.py --asset "$A" --samples 1000 --batch 128 --dtype float16 \
      >> "logs/llmtime_${A}_N1000_fp16.log" 2>&1
    echo "    exit $? $(date '+%F %T')"
  done
  ;;
t5)
  for A in $ASSETS; do
    echo "=== t5 $A $(date '+%F %T') ==="
    $PY src/02_chronos_lab.py t5 "$A" --samples 2000 \
      >> "logs/t5_${A}_N2000.log" 2>&1
    echo "    exit $? $(date '+%F %T')"
  done
  ;;
sizes)
  # The "bigger is worse" exhibit currently rests on two points, one of them an
  # eight-date probe. 3B, 7B and 14B over the full span turn it into a curve or kill
  # it. fp16 throughout: 14B in fp32 is 56 GB of weights, and on the 1.5B model fp16
  # and fp32 gave byte-identical greedy output on the probe dates.
  for M in 3B:16 7B:16 14B:8; do
    SZ="${M%%:*}"; B="${M##*:}"
    echo "=== local ${SZ} $(date '+%F %T') ==="
    $PY src/07_local_llm.py --asset SPX --config series \
      --model "Qwen/Qwen2.5-${SZ}-Instruct" --batch "$B" --dtype float16 \
      >> "logs/local_series_SPX_${SZ}_fp16.log" 2>&1
    echo "    exit $? $(date '+%F %T')"
  done
  ;;
*)
  echo "usage: $0 {llmtime|t5|sizes}" >&2
  exit 2
  ;;
esac

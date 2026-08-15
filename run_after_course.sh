#!/usr/bin/env bash
# The published LLM-VaR sampling protocol on the S&P 500, to be run after the lecture.
#
# What this reproduces that src/09_llmtime.py did not, before 2026-08-15:
#   --wrapper   the sequence goes inside the published system message and the "continue
#               without producing any additional text" instruction (models/gpt.py)
#   --scaler    the context is passed through get_scaler(alpha=0.95, beta=0.3) before
#               serialising, and the draws are mapped back afterwards (models/llmtime.py)
# Both are transcribed from github.com/QuantLet/LLM_Risk, retrieved 2026-08-15.
#
# Cost, measured on this machine at 36 s per date per 200 draws: about 25 hours at
# N = 1000 over the 500 scored days. The stage checkpoints every date, so it can be
# stopped and restarted; it resumes from precomputed/_llmtime_*.csv.
#
# Output lands in its own file, so nothing already in precomputed/ is overwritten:
#   precomputed/llmtime_SPX_Qwen2.5-1.5B-Instruct_T0.7_N1000_float16_pubprompt_scaled.csv
#
# Afterwards: rerun src/04_slide_figures.py, src/10_slide_tables.py and
# src/05_verify_slide_claims.py, then compare the new leg with the current one. On a
# four-date smoke test the published path ran wider, mean VaR 2.12 against 1.93, so the
# comparison is worth making rather than assuming.

set -euo pipefail
cd "$(dirname "$0")"

PY="../aida-ensemble/venv/bin/python"

$PY src/09_llmtime.py \
    --asset SPX \
    --wrapper \
    --scaler \
    --samples 1000 \
    --temp 0.7 \
    --dtype float16

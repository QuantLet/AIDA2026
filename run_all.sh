#!/usr/bin/env bash
# Regenerate the whole August package from the certified aida-risk run.
#
#   ./run_all.sh            everything except the LLM calls (no API key needed)
#   ./run_all.sh --llm      also re-run the LLM forecasts (costs money, see README)
#
# Chronos segfaults on import under the main Anthropic/conda environment, so the two
# Chronos stages run under ../aida-ensemble/venv, which has a working install. Every
# other stage runs under whatever python3 is on the path.
#
# Pinned versions per interpreter: requirements.txt (PY), requirements-chronos.txt
# (CHRONOS_PY), requirements-openweights.txt (run_raised.sh). See README.md.

set -euo pipefail
cd "$(dirname "$0")"

PY=python3
CHRONOS_PY=../aida-ensemble/venv/bin/python
ASSETS="SPX DAX N225 GOLD BTC"
RUN_LLM=0
[[ "${1:-}" == "--llm" ]] && RUN_LLM=1

echo "== 01  lab dataset from the certified run"
$PY src/01_lab_data.py

echo "== 02a Chronos-Bolt, all assets (seconds)"
$CHRONOS_PY src/02_chronos_lab.py bolt $ASSETS 2>&1 | grep -viE "deprecated|quantiles to be"

echo "== 02b Chronos-T5 sampling, SPX and BTC (~10 min each, resumable)"
$CHRONOS_PY src/02_chronos_lab.py t5 SPX BTC 2>&1 | grep -viE "deprecated|quantiles to be"

if [[ $RUN_LLM -eq 1 ]]; then
  echo "== 03a EODHD headlines: fetch, align to forecast dates, audit the as-of rule"
  : "${EODHD_API_KEY:?set EODHD_API_KEY to re-fetch the news}"
  for A in $ASSETS; do $PY src/06_news.py --asset $A; done

  echo "== 03b LLM forecasts, four configurations (Batches API)"
  : "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY to re-run the LLM stage}"
  for A in $ASSETS; do
    for C in series series+state dated dated+news; do   # `dated` controls the news test
      $PY src/03_llm_lab.py --config $C --asset $A
    done
  done
else
  echo "== 03  news and LLM forecasts SKIPPED (pass --llm to re-run; shipped files used)"
  echo "       news alignment can be re-audited without any API call:"
  for A in $ASSETS; do $PY src/06_news.py --asset $A --audit; done
fi

echo "== 03c open-weights LLM (no key needed; ~10 min per asset on CPU/GPU)"
if [[ $RUN_LLM -eq 1 ]]; then
  for A in $ASSETS; do
    $CHRONOS_PY src/07_local_llm.py --asset $A --config series --batch 8 \
      2>&1 | grep -E "distinct|mean VaR|->"
  done
else
  echo "       SKIPPED (shipped files used); pass --llm to re-run"
fi

echo "== 03d LLMTime: the published sampling method (open weights, ~4 h for 500 days)"
if [[ $RUN_LLM -eq 1 ]]; then
  $CHRONOS_PY src/09_llmtime.py --asset SPX --samples 120 2>&1 | grep -E "usable|VaR|->"
else
  echo "       SKIPPED (shipped file used); pass --llm to re-run"
fi

echo "== 04  slide figures and the fact table"
$PY src/04_slide_figures.py SPX

echo "== 04b LaTeX table fragments for the deck"
$PY src/10_slide_tables.py

echo "== 05  verify every number printed on a slide"
$PY src/05_verify_slide_claims.py SPX

echo "== 05b quantlet packages"
$PY src/08_package_quantlets.py

echo "== 06  student notebook"
$PY notebook/build_notebook.py

echo "== 07  student data bundle"
$PY - <<'EOF'
import zipfile, pathlib
root = pathlib.Path(".")
out = root / "aida_lab_data.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    # Paths are relative to the extraction root, so `extractall(".")` on Colab puts
    # everything exactly where aidalab.py's loaders look for it. Nesting the archive
    # under its own directory would break data_root() and the whole lab with it.
    for p in sorted((root / "data" / "lab").glob("*")):
        z.write(p, f"data/lab/{p.name}")
    for p in sorted((root / "data" / "news").glob("headlines_*.csv")):
        z.write(p, f"data/news/{p.name}")
    for p in sorted((root / "precomputed").glob("*.csv")):
        if p.name.startswith("_"):          # resumable checkpoints, not shipped
            continue
        z.write(p, f"precomputed/{p.name}")
    z.write(root / "notebook" / "aidalab.py", "aidalab.py")
print(f"  -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
EOF

echo "== 08  lecture deck"
( cd slides && pdflatex -interaction=nonstopmode ai_risk_course_2026-08.tex >/dev/null \
  && pdflatex -interaction=nonstopmode ai_risk_course_2026-08.tex >/dev/null \
  && echo "  -> slides/ai_risk_course_2026-08.pdf" )

echo
echo "done."

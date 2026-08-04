# Moving the package to another machine

The work product is small: **7.2 MB compressed**. Everything heavy — the Python
environment, the Hugging Face model cache, the raw EODHD news dump — is rebuilt or
re-downloaded on first use, and none of it is worth copying.

## What to copy

```bash
# on the old machine
tar --exclude='data/news/raw_*.jsonl' --exclude='precomputed/_*' \
    --exclude='**/__pycache__' --exclude='slides/*.aux' --exclude='slides/*.log' \
    --exclude='slides/*.nav' --exclude='slides/*.out' --exclude='slides/*.snm' \
    --exclude='slides/*.toc' \
    -czf august-course-transfer.tgz august-course
```

Also copy, if the new machine does not already have them:

- `../aida-risk/` — the certified run this package carves its dataset out of. Only
  `data/raw/*.csv` and `data/processed/**/forecasts.csv` are actually needed.
- `../beamer-deck/preamble.tex` and the logo PNGs — the deck `\input`s them.

What is **not** copied and why:

| Item | Size | Why not |
|---|---|---|
| `data/news/raw_*.jsonl` | 154 MB | Regenerable with `src/06_news.py`; the aligned headline files that the lab needs are 200 KB and *are* in the archive |
| `precomputed/_*` | — | Resume checkpoints, meaningful only to the run that wrote them |
| Python environment | 1.3 GB | Rebuilt below |
| Hugging Face cache | 12 GB | Re-downloaded on first use |

## Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers accelerate chronos-forecasting \
            pandas numpy scipy matplotlib anthropic
```

TeX Live is needed for the deck. Chronos segfaults under some conda builds; if
`import chronos` crashes, use a clean venv as above rather than a conda environment.

Keys go in `~/.zshrc`, which is where this project keeps them:

```bash
export ANTHROPIC_API_KEY=...   # only to re-run stage 03; shipped files need no key
export EODHD_API_KEY=...       # only to re-fetch news; aligned files are in the archive
```

Neither key is needed to run the laboratory or to rebuild the deck.

## Verify the move landed intact

```bash
python3 src/05_verify_slide_claims.py SPX      # expect: 61 claims verified, 0 failed
python3 notebook/build_notebook.py             # expect: 53 cells
( cd slides && pdflatex -interaction=nonstopmode ai_risk_course_2026-08.tex )
```

If those three pass, nothing was lost in transit.

## What a larger machine unlocks

The constants below were set by what fitted on a 12-core laptop GPU. On a machine with
substantially more unified memory they are the first things to raise, and each buys
something specific rather than just speed.

| Constant | Now | Raise to | What it buys |
|---|---|---|---|
| `09_llmtime.py: N_SAMPLES` | 120 | 500–1000 | The 1% quantile is currently an order statistic of **one** draw. At 1000 it is ten. This is the same floor argument the deck makes against Chronos-T5's default of 20, and it currently applies to our own run. |
| `09_llmtime.py: BATCH` | 32 | 128–256 | Draws are independent, so this is pure throughput. The 4-hour SPX run should fall to well under an hour. |
| `09_llmtime.py` assets | SPX only | all five | The published method on every asset, so the sampling-against-elicitation comparison replicates the way the news test does. |
| `07_local_llm.py: --model` | 1.5B, 3B | add 7B, 14B | The "bigger is worse" finding rests on two points. A 7B and a 14B turn an anecdote into a curve, and 14B in fp16 needs ~28 GB. |
| `02_chronos_lab.py: NUM_SAMPLES` | 500 | 2000 | Same floor problem for Chronos-T5 at 1%: five draws below the quantile becomes twenty. |

**Do the sample-count increases first.** They are not performance tuning — at
`N_SAMPLES = 120` and $\alpha = 1\%$ our own sampling run sits on the exact floor the
lecture criticises, and raising it removes a caveat rather than saving time.

## In-flight runs

Two runs were detached with `nohup` when this was written:

- `09_llmtime.py --asset SPX --samples 120` — checkpointing was added *after* it
  started, so this particular run still writes nothing until it finishes. Killing it
  loses its work. Every future run resumes from `precomputed/_llmtime_*.csv`.
- `07_local_llm.py --model Qwen/Qwen2.5-3B-Instruct` — cheap to restart, and the deck
  currently states only the eight-date probe until it lands.

Restarting both on the new machine, with the raised sample counts, is the better use of
the time than waiting for either to finish here.

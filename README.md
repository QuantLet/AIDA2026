# august-course — AI in Financial Risk Modelling (August 2026)

Lecture deck and coding laboratory for the four-hour August session of the AIDA Summer
School 2026. Two hours of lecture, two hours in one notebook.

**Reported level: alpha = 1%.** The 5% column is carried alongside because the language
model returns both quantiles in one reply, but every headline number is the 1% tail.

**No API key is needed at any point.** Every commercial forecast is shipped as data, and
the language-model leg students run themselves uses open weights.

The question the session is built around is not which model is cleverest. It is whether
you can tell, from the output alone, that a model answered the question you asked.

Nothing here is simulated and nothing is downloaded at build time. The price series are
the raw Yahoo Finance files already in `../aida-risk/data/raw` (retrieved 2026-07-20 for
the S&P 500, 2026-07-25 for the rest), and the classical forecasts are the ones the
July deck's numbers trace to.

## Quick start

```bash
./run_all.sh                # everything except the LLM calls; no API key needed
./run_all.sh --llm          # also re-runs the LLM forecasts (see Cost below)

python3 src/05_verify_slide_claims.py SPX    # 61 slide numbers, re-derived from data
```

### Environments

The package needs three, and every pin below was read from the interpreter that produced
the committed results rather than assumed:

| file | interpreter | stages | what it regenerates |
|---|---|---|---|
| `requirements.txt` | `python3` | 01, 04, 04b, 05, 06 | every figure, every table, all 105 checks |
| `requirements-chronos.txt` | `../aida-ensemble/venv/bin/python` | 02 | the Chronos panels |
| `requirements-openweights.txt` | `~/.venvs/aida-2026/bin/python` | 03, 07, 09 | the language-model panels |

**The first one is enough to rebuild the deck.** The model panels are committed under
`precomputed/`, so a reviewer reproduces every number without weights, GPU or API key.

#### The notebook is deliberately not pinned to any of them

The three files above pin *this machine*, so the committed panels can be regenerated.
The notebook has a different job: survive Google Colab. Transplanting `torch==2.13`
from the local MPS environment onto a Colab runtime buys nothing pedagogically and
invites a mass reinstall, a CUDA mismatch or a runtime restart in the middle of a
two-hour laboratory.

So the two install cells stay unpinned, and every stage prints the versions it actually
resolved: the setup cell prints Python, numpy, pandas, scipy and matplotlib; the Chronos
cell prints `chronos-forecasting`, `transformers` and `torch`; the open-weights cell
prints `transformers`, `accelerate`, `torch` and the device. A result from a foundation
model belongs to an artefact and an interface, which is the argument of the session --
the notebook now records its own interface.

To harden it for a specific cohort, in this order:

1. Pin the **Colab runtime**, not the packages --- Colab supports this precisely so a
   platform update cannot break a workshop notebook. Run the notebook end to end on the
   target runtime first.
2. If the Chronos section needs pinning, pin only `chronos-forecasting`, and to the
   version the printout showed working.
3. Pin `transformers` and `accelerate` only after a full run on that runtime, and leave
   `torch` to the runtime unless the run proves otherwise.

Nothing here is pinned on the strength of the local environment alone.

The split is not tidiness. Chronos segfaults on import under the main environment
(chronos-forecasting 2.2.2 / transformers 4.57 / torch 2.12, Python 3.13), and the
open-weights stages have since moved to transformers 5.14 / torch 2.13 under Python
3.12, which chronos-forecasting 2.2.2 does not accept. `run_all.sh` and `run_raised.sh`
route each stage to the right interpreter; if you run a stage by hand, use the one in
the table.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 01 | `src/01_lab_data.py` | `data/lab/returns_*.csv`, `bench_*.csv`, `manifest.json` |
| 02 | `src/02_chronos_lab.py bolt\|t5 ASSETS` | `precomputed/chronos_{bolt,t5}_*.csv` |
| 03a| `src/06_news.py --asset SPX` | `data/news/headlines_*.csv`, `coverage_*.csv` |
| 03b| `src/03_llm_lab.py --config ...` | `precomputed/llm_*_*.csv` |
| 04 | `src/04_slide_figures.py` | `figures/slides/*.pdf`, `precomputed/slide_facts_SPX.csv` |
| 05 | `src/05_verify_slide_claims.py` | pass/fail on every number printed on a slide |
| 03c| `src/07_local_llm.py` | `precomputed/local_*.csv` — open weights, no key |
| 03d| `src/09_llmtime.py` | `precomputed/llmtime_*.csv` — the published sampling method |
| 05b| `src/08_package_quantlets.py` | `quantlets/` — seven citable packages |
| 07 | `notebook/build_notebook.py` | `notebook/AIDA_Risk_Lab.ipynb` |

`src/labcommon.py` holds the design constants and is the only place they are written.
`notebook/aidalab.py` is the student-facing helper: loaders, historical simulation, the
pinball loss, Kupiec, Christoffersen, Brier, Diebold-Mariano with a HAC standard error,
and the plots. It depends only on numpy, pandas, scipy and matplotlib, all preinstalled
on Colab, so the notebook's first result appears before anything is installed.

## Data

Five assets, one per group. Each carries its full sample plus a 500-day test span carved
from the end of the certified `aida-risk` run.

| Key | Ticker | Obs | Test span |
|---|---|---|---|
| `SPX` | `^GSPC` | 6791 | 2023-01-04 .. 2024-12-30 |
| `DAX` | `^GDAXI` | 6851 | 2023-01-13 .. 2024-12-30 |
| `N225` | `^N225` | 6617 | 2022-12-20 .. 2024-12-30 |
| `GOLD` | `GC=F` | 6104 | 2023-01-04 .. 2024-12-30 |
| `BTC` | `BTC-USD` | 3757 | 2023-08-19 .. 2024-12-30 |

Protocol, inherited from `aida-risk` and not re-derived here: rolling window 1000 days,
parameters refitted every 20 days, burn-in 250 days, and at every date the forecast uses
information up to `t-1` and nothing else.

Sign convention: VaR and ES are stored as positive losses, and day `t` is a breach when
`ret_t < -VaR_t`. Figures are drawn on the return axis with the tail on the left.

## Why the test span is 500 days

The design this session started from proposed 40 to 60 days. A backtest is a hypothesis
test and needs observations: at alpha = 5% over 50 days, Kupiec rejects only a true
breach rate above **12.1%**. Over 500 days it rejects above **7.1%**. On the short span
every model passes, which teaches the opposite of the intended lesson. Both numbers are
computed in `src/05_verify_slide_claims.py` and plotted in `figures/slides/s_power.pdf`.

## Models

| Key | What it is | Which factor of `VaR = sigma_t * q_nu(alpha)` it learns |
|---|---|---|
| `HS` | Historical simulation, 1000-day window | neither |
| `GARCH-t` | GARCH(1,1), Student-t innovations | scale, parametrically |
| `NN-t` | Neural volatility + parametric t tail | the scale only |
| `Chronos-Bolt` | Chronos-Bolt-tiny, zero-shot, quantile head | the whole predictive law |
| `Chronos-T5` | Chronos-T5-mini, zero-shot, 500 sampled paths | the whole predictive law |
| `LLM-series` | Serialised return history, prompted | the whole predictive law |
| `LLM-series+state` | The same plus volatility, drawdown, HS VaR | the whole predictive law |
| `LLM-dated` | The same plus the asset name and the date | the whole predictive law |
| `LLM-dated+news` | The same plus real headlines available at t-1 | the whole predictive law |
| `Open-1.5B` | Qwen2.5-1.5B-Instruct, open weights, same prompt | the whole predictive law |
| `Open-3B` | Qwen2.5-3B-Instruct — the degenerate exhibit | nothing; it is constant |

All seven model families are precomputed for all five assets.

Chronos is fed **price levels**, not returns. It is trained on series levels, and the
implied return is recovered as `r = 100 log(P_hat / P_{t-1})`, a monotone transform, so a
price quantile maps to the return quantile of the same level. Feeding it returns was
tried in the certified pipeline and is not a fair test: returns are near-zero-mean noise
and the model collapses to a predictive standard deviation of 0.14 against a realised
1.22 (documented in `../aida-risk/src/02c_chronos.py`).

## The LLM leg: two methods, one serialisation

**Sampling — the published method.** Pele, Bolovaneanu, Lin, Ren, Ginavar, Spilak,
Andrei, Toma, Lessmann and Härdle (2025), *In the Beginning was the Word: LLM-VaR and
LLM-ES*, Expert Systems with Applications, following LLMTime (Gruver et al., 2023).
Serialise the returns, sample N completions at temperature T, and read the quantile off
the empirical distribution of the sampled next values. Architecturally the same object as
the Chronos-T5 sampling head, and with the same floor: at 1% with 120 draws, roughly one
draw lies below the quantile. Implemented in `src/09_llmtime.py` with open weights,
because sampling needs N generations per forecast.

**Elicitation — this course's addition.** Serialise the returns, ask for the quantile in
words, parse the reply. One call instead of N, and it can be wrong in ways sampling
cannot be: a positive quantile, an inverted pair. Implemented in `src/03_llm_lab.py`.

**The serialisation is identical, and that is not a coincidence of style.** The published
settings, verified against the code at `github.com/QuantLet/LLM_Risk`, are
`SerializerSettings(base=10, prec=2, signed=True, time_sep=', ', bit_sep='',
minus_sign='-')` — literally `", ".join(f"{x:.2f}")`. So the two legs share the input
representation and differ only in how the number is extracted, which is what makes the
comparison between them mean something.

First measurement of that comparison, 12 dates at 120 draws on the same open-weights
model: sampling produced **12 distinct forecasts out of 12**; elicitation produces a
distinct value on about 10% of days. Sampling varies continuously; elicitation quantises.

There is no VaR head and no fine-tuning in either leg.

Four information sets:

- `series` — the last 60 returns, **anonymised**: no ticker, no dates, no price levels.
- `series+state` — the same plus realised volatility at two horizons, the drawdown from
  the 250-day peak, and the rolling historical-simulation VaR.
- `dated` — the same returns plus the asset name and the forecast date. Both the
  contamination probe and the control for the news test.
- `dated+news` — the same as `dated`, plus the real headlines available at `t-1`.

## News: real headlines, and the rule that makes them admissible

Headlines come from EODHD's financial-news endpoint via `src/06_news.py`. Nothing is
invented: a headline file written to fit the returns would produce a text-to-risk
relation that exists only because it was constructed.

**The as-of rule.** A headline may enter the prompt for forecast date `t` only if it was
published at or before the as-of cutoff on `t-1` — the previous *decision point*.
Consecutive cutoffs tile the calendar: no article falls in two windows and none falls in
none. A fixed 24-hour window, which is the obvious implementation, would silently drop
every evening and every weekend.

**The cutoff is each asset's own market close, measured per exchange.** A single US close
applied to every asset would hand a Tokyo forecast fourteen hours of news published after
Tokyo shut — the exact leak the rule exists to prevent. Where a market observes daylight
saving, the earlier of the two UTC closes is used, so the window is never too generous.

| Asset | Market | Local close | `AS_OF_HOURS` (UTC) |
|---|---|---|---|
| `SPX` | NYSE | 16:00 ET | 20.0 |
| `DAX` | Xetra | 17:30 CET | 15.5 |
| `N225` | TSE | 15:00 JST (no DST) | 6.0 |
| `GOLD` | COMEX | 13:30 ET pit | 17.5 |
| `BTC` | 24/7 | Yahoo's bar runs 00:00–24:00 UTC | 24.0 |

Each headline carries its publication timestamp and its cutoff into the aligned file, so
the rule is audited rather than trusted. `src/06_news.py --audit` re-checks two things
independently — that no headline post-dates its own cutoff, and that the stored cutoff
really is the previous forecast date at 20:00 UTC — and exits non-zero on any violation.
This matters more than the price alignment: a price series can only leak by being
shifted, whereas a news series leaks the moment an article published at 22:00 describing
the day's close is handed to a model asked to forecast that close, and nothing in the
reply would look wrong.

**Symbols were chosen on measured coverage, not on which ticker looked right.** Each
candidate was probed over 2023-01-01 to 2024-12-30 and the article count decided it. The
counts on the first page (cap 1000) were:

| Asset | Chosen | Count | Rejected candidates |
|---|---|---|---|
| `SPX` | `GSPC.INDX` | 1000 | `SPY.US` 222, `QQQ.US` 200, `DIA.US` 63, `VOO.US` 22 |
| `DAX` | `GDAXI.INDX` | 1000 | `DAX.INDX` 0, `EXS1.XETRA` 0, `DAX.XETRA` 0, `EWG.US` 0 |
| `N225` | `N225.INDX` | 444 | `NI225.INDX` 0, `EWJ.US` 11, `1321.TSE` 404 |
| `GOLD` | `XAUUSD.FOREX` | 1000 | `GLD.US` 89, `IAU.US` 3, `GC.COMM` 404 |
| `BTC` | `BTC-USD.CC` | 1000 | `BTC.CC` 147, `GBTC.US` 174, `BTCUSD.CC` 0 |

`GOLD` news is quoted on spot XAU/USD while the price series is the COMEX future: the
same market through two instruments, stated rather than hidden. `--probe` exists so any
of these can be re-tested.

**Coverage after alignment**, with zero as-of violations on every asset:

| Asset | Articles fetched | Admissible | Dates covered | Per covered date | Median lead |
|---|---|---|---|---|---|
| `SPX` | 35,293 | 2,128 | 447 / 500 (89%) | 4.8 | 1.1 h |
| `DAX` | 1,016 | 955 | 386 / 500 (77%) | 2.5 | 7.5 h |
| `N225` | 464 | 449 | 254 / 500 (51%) | 1.8 | 21.8 h |
| `GOLD` | 1,839 | 1,352 | 391 / 500 (78%) | 3.5 | 9.1 h |
| `BTC` | 12,047 | 2,481 | 500 / 500 (100%) | 5.0 | 0.0 h |

`N225` is the thin one: half its dates carry no headline at all, and its median headline
is nearly a day old at the cutoff. That is reported rather than smoothed over, and a
group working on the Nikkei should read its news result as underpowered.

**Why the control is `dated` and not `series`.** Real headlines reveal the period on
their own — they name companies, events and numbers that place the text in time. So
comparing `series` with a news configuration would confound two effects, "the text
carries information" and "the model now knows which period this is". `dated` holds the
second fixed, so the difference between `dated+news` and `dated` identifies the first.
On a date with no headline in its window the two prompts are identical by construction,
so the informative test is on the covered subset; both are reported.

### Cost

Forecasts go through the Message Batches API at half price. The four shipped
configurations are 500 requests each on `claude-haiku-4-5`, about **US$1.60 in total**;
each batch returned in three to four minutes. EODHD usage for the news fetch: 36 requests. `--live` sends one request per date instead, which is
slower and dearer and exists for smoke tests (`--limit 3 --live` costs fractions of a
cent).

### Structured outputs

The pipeline enforces JSON by prompt and validates on parse, rather than using the API's
structured-output constraint. The reason is version-specific: `anthropic` 0.71.0, the
version installed here, does not type `output_config`, and the Batches request builder
would not carry it. On a newer SDK the constraint is

```python
output_config={"format": {"type": "json_schema", "schema": SCHEMA}}
```

with `additionalProperties: false` on every object and no numerical range constraints,
which the API does not support. Prompt-enforced JSON plus a validator is version-proof
and provider-portable, and it keeps the failure counts visible, which is the point of
the exercise.

## Validation and excluded specifications

This section exists so the package's failures are as easy to find as its results.

**Chronos-Bolt cannot produce a 5% quantile.** Verified 2026-07-31 with
chronos-forecasting 2.2.2: it was trained on the grid 0.1, 0.2, ..., 0.9, and any request
below 0.1 is clamped to the value at 0.1. On **100% of the 500 test days** the 1%, 5% and
10% outputs are bit-identical. The library emits a warning and returns successfully. A
pipeline that requested `quantile_levels=[0.05]` would have produced a complete set of
results silently reporting the 10% quantile. Chronos-Bolt is therefore reported at 10%,
its lowest unclamped level, and the genuine 5% quantile comes from the Chronos-T5 sampling
head. The clamp is not repaired anywhere: `src/02_chronos_lab.py` records a `clamped`
flag per row and the notebook puts it on screen.

**Sampling floor.** Chronos-T5 uses 500 sample paths, which puts about 25 draws below the
5% quantile and 5 below the 1%. The library default of 20 samples cannot express a 5%
quantile at all. The sample count is a modelling choice of the same standing as the
window length in historical simulation.

**A partial run cannot overwrite a finished one.** During the build a two-day smoke
test replaced a completed 500-day file, and nothing downstream would have complained:
the loaders would simply have scored two days and reported a result. `--limit` now
writes to a `_smoke<N>` filename, so the canonical file can only ever come from a full
run. The failure is recorded here rather than quietly patched, because it is the same
class of error the laboratory spends two hours on.

**LLM reply failures, counted not dropped.** Over 500 days and four configurations:
`series` 500 of 500 parsed, `series+state` 499, `dated` 500, `dated+news` 499 — two
unparseable replies out of 1998. No configuration produced a parsed reply with a
non-negative 5% quantile, which would be worse than an unparseable one because it asserts
that the worst 5% of days is a gain and would flow downstream without tripping anything.
Every count is printed by the generating script and appears on a slide.

**The headline result at alpha = 1%.** S&P 500, 500 days, five expected breaches:

| Model | n | Breaches | Rate | UC | IND | Pinball |
|---|---|---|---|---|---|---|
| NN vol + t | 500 | 3 | 0.60% | pass | pass | 0.0269 |
| LLM, dated + real headlines | 500 | 3 | 0.60% | pass | pass | 0.0276 |
| GARCH(1,1)-t | 500 | 7 | 1.40% | pass | pass | 0.0277 |
| LLM, series + market state | 500 | 6 | 1.20% | pass | pass | 0.0291 |
| LLM, dated (control) | 500 | 3 | 0.60% | pass | pass | 0.0298 |
| LLM, series only | 500 | 3 | 0.60% | pass | pass | 0.0299 |
| Chronos-T5 (zero-shot) | 500 | 6 | 1.20% | pass | pass | 0.0303 |
| Historical simulation | 500 | 2 | 0.40% | pass | pass | 0.0379 |
| **Open weights, 1.5B** | 494 | **35** | **7.09%** | **reject** | pass | 0.0533 |

**Eight of nine pass, and that is the finding.** Not because eight models are good: with
five expected breaches, Kupiec cannot separate counts of 2, 3, 6 and 7. At 500 days and
1% it rejects only a true rate above **2.00%** — a doubling. The single rejection misses
by a factor of seven. "Passed the backtest" at this sample size means very little, and it
is exactly the sentence that appears in model documentation.

**The news effect reverses at 1%.** At 5% no asset was significant. At 1%:

| Asset | Dates covered | n | Mean loss difference | DM p |
|---|---|---|---|---|
| `SPX` | 447 | 447 | -0.00185 | **0.009** |
| `N225` | 254 | 254 | -0.00183 | **0.001** |
| `DAX` | 386 | 386 | -0.00069 | 0.214 |
| `GOLD` | 391 | 391 | +0.00042 | 0.667 |
| `BTC` | 500 | 500 | -0.00032 | 0.857 |

Negative means the headlines helped. Text matters for the extreme tail and not for
ordinary moves, which is the only place anyone claimed it would. Five tests were run, so
Bonferroni puts the threshold at 0.01: the Nikkei survives it, the S&P sits on the line.

**Limits, all stated on the slide that carries the table:** one asset for the headline,
one 500-day span, one commercial model, four prompts; 2023-2024 is a calm period with no
2008 and no March 2020; contamination is not excluded, since naming the date changes
little but the regime is in the corpus regardless; and nine models with three tests each,
with no multiple-testing control on the backtest table.

**What the LLM appears to be tracking.** Correlation with GARCH-t runs 0.58 to 0.68
across the four configurations; with historical simulation, 0.19 to 0.35. Nothing added
to the prompt — market state, the date, two thousand headlines — improved the forecast.
Whatever it is reading was already in the return series: the language model is doing
volatility estimation in words.

## Delivering the laboratory

**Students need no API key of any kind.** Every language-model forecast and every
headline is shipped as data. The only thing fetched at run time is the Chronos checkpoint
from Hugging Face, which is a public download.

They need `notebook/AIDA_Risk_Lab.ipynb` and either the repository or
`aida_lab_data.zip` (1.6 MB: five return series, the benchmark forecasts, the aligned
headlines, every precomputed forecast, and `aidalab.py`). On Colab the first cell prompts
for the upload and extracts it into the working directory — the archive's internal paths
are `data/lab/`, `data/news/`, `precomputed/` and `aidalab.py`, which is exactly where
the loaders look, so nesting it under its own folder would break `data_root()` and the
lab with it. If the data is published to a public URL, set `AIDA_DATA_URL` instead. With
neither, the loaders fall back to a live yfinance download and say so; that path will not
match the precomputed files.

**Every asset carries a complete set** — Chronos-Bolt, Chronos-T5, aligned headlines and
all four LLM configurations — so no group opens the notebook with an empty column. The
bundle was tested by unpacking it into an empty directory with `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY` and `EODHD_API_KEY` unset, loading all five assets and asserting that
each resolves to all eight models. The only live model call in the lab is a 40-day Chronos-T5 reproduction that
takes about two minutes on a Colab CPU.

## Layout

```
data/lab/          returns and benchmark forecasts, five assets, plus manifest.json
data/news/         raw EODHD articles (136 MB, git-ignored, regenerable), the
                   aligned headline file (~200 KB, the one the lab needs) and coverage
precomputed/       Chronos and LLM forecasts, the backtest table, the slide fact table
figures/slides/    presentation-scale figures, PDF for LaTeX and 300 dpi PNG
notebook/          aidalab.py, build_notebook.py, AIDA_Risk_Lab.ipynb
slides/            ai_risk_course_2026-08.tex and its PDF
src/               the numbered pipeline
run_all.sh         regenerates all of the above
```

Files beginning with an underscore in `precomputed/` are resume checkpoints for the slow
Chronos-T5 stage. They are kept on disk and excluded from the student bundle.

## References

- Pele, Bolovaneanu, Lin, Ren, Ginavar, Spilak, Andrei, Toma, Lessmann and Härdle (2025).
  *In the Beginning was the Word: LLM-VaR and LLM-ES*. Expert Systems with Applications.
  Code: `github.com/QuantLet/LLM_Risk`.
- Ansari et al. (2024). *Chronos: Learning the Language of Time Series*. TMLR.
- Das, Kong, Sen and Zhou (2024). *A Decoder-Only Foundation Model for Time-Series
  Forecasting*. ICML.
- Kupiec (1995). *Techniques for Verifying the Accuracy of Risk Measurement Models*.
  Journal of Derivatives.
- Christoffersen (1998). *Evaluating Interval Forecasts*. International Economic Review.
- Fissler and Ziegel (2016). *Higher Order Elicitability and Osband's Principle*. Annals
  of Statistics.
- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market
  Risk*.

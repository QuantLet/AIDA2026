# August 2026 — physical, 4 hours
## AI in Financial Risk Modelling: can foundation models produce valid risk forecasts?

**Status: built.** Deck is `slides/ai_risk_course_2026-08.tex` (53 frames, 102 pages with
overlays; alpha = 1%, IDA two-level bullets, 14 quantlet marks). Laboratory is `notebook/AIDA_Risk_Lab.ipynb` (53 cells, executes end to end
with zero errors). Empirical content is `august-course/`, derived from `aida-risk/`.
See `DECISIONS.md` 2026-07-31.

**Level: intermediate.** Assume the 27 July online hour, or its equivalent: VaR and ES
defined, the scale/shape split seen once, GARCH recognised. Act II re-establishes those
in fifteen minutes for anyone who missed it.

---

## The spine

The July hour established that

```
VaR_t(alpha) = sigma_t          x        q_nu(alpha)
               ^^^^^^^                   ^^^^^^^^^^
               THE SCALE                 THE SHAPE
               learnable                 not learnable
```

This session takes the next step. What happens when the model was **pretrained somewhere
else** and applied here with no fitting at all — a time-series foundation model, or a
language model? The question stops being *which model wins* and becomes:

> **Can you tell, from the output alone, whether the model answered your question?**

Four failures carry the argument, ordered by how quietly they fail:

1. **Chronos-Bolt returns the 10% quantile when asked for 1%.** Nothing errors, and at
   the reported level that is a factor-of-ten substitution.
2. **An LLM returns a well-formed number** that need not be a quantile of anything.
3. **An open-weights model passes every validity check** — parses, correct sign, correct
   ordering — and is still the only model the backtest rejects.
4. **A 500-day backtest at 1% passes eight models of nine**, because five expected
   breaches cannot separate them.

**The single sentence they should leave with:** a pretrained model will always answer;
validation is the only thing that tells you whether it answered *your* question.

---

## Hour 1 and 2 — lecture (53 frames)

| Act | Frames | Minutes | Content |
|---|---|---|---|
| Opening | 3 | 0–8 | Three models, one number. The roadmap. |
| I — What is a risk forecast? | 5 | 8–25 | Forecast vs risk model; VaR and ES; point vs distribution; the two factors and their unequal evidence. |
| II — The classical answers | 5 | 25–45 | Historical simulation and why it cannot react; GARCH as scale-plus-assumed-shape; the pinball loss and its weak identification in the tail; where AI can actually help. |
| *break* | | 45–55 | |
| III — Time-series foundation models | 8 | 55–80 | Supervised / pretraining / fine-tuning / zero-shot; point vs marginal quantiles vs a predictive law; Chronos as a language over series; prices not returns; **the clamp**; the sampling floor. |
| IV — LLMs as risk forecasters | 13 | 80–103 | Four things people mean by "LLMs for risk"; the LLM-VaR pipeline; the prompt as a modelling decision; **the as-of rule for real headlines**; what the feed gave us; **why the control is `dated`, not `series`**; the five failure modes; what came back; what it is really tracking. |
| V — Validation | 11 | 100–115 | Kupiec and Christoffersen; the power curve; the results table; **Diebold-Mariano**; what the result is not; the leakage taxonomy; disagreement and SYNCRISK; what AI adds to the validation burden. |
| Lab brief + close | 5 | 115–120 | What you will do; three things to leave with; references. |

**Load-bearing frames.** "You ask for the 5% quantile. You get the 10%." and the frame
after it, "Why this is the important slide". Everything in Acts III to V is a variation on
that one point. If time runs short, compress Act II, never those two.

**Two frames that must not be softened.** "Before believing the ordering: is any gap
real?" reports that not one information manipulation — market state, the date, two
thousand real headlines — is distinguishable from zero, and that the LLM did not win.
"What this result is not" lists the four limits. A deck that keeps the leaderboard and drops these
two would be making a claim the data does not support.

---

## Hour 3 and 4 — laboratory

One notebook, six sections, one asset per group (S&P 500, DAX 40, Nikkei 225, gold
futures, Bitcoin).

| Minutes | Section | What happens |
|---|---|---|
| 0–10 | Setup and the target | Data loads; the power cell shows why the span is 500 days and not 50. |
| 10–25 | Historical simulation | Students write `my_hs_var` themselves and check it against the reference. The window ends at `t-1`; the leak is discussed, not merely avoided. |
| 25–45 | Chronos-Bolt | They request `[0.01, 0.05, 0.10, 0.50, 0.90]`, read the warning, and **verify with an equality test** that the three lowest are bit-identical. Bolt is then run at 10%, its lowest unclamped level. |
| 45–60 | Chronos-T5 | Precomputed 500 days loaded; 40 days reproduced live (~2 min) and correlated against the file. |
| 60–72 | LLM forecasts | Four configurations loaded; parse, sign and ordering failures counted before any result is read; correlation against HS and GARCH. |
| 72–82 | **Run an LLM yourself** | The open-weights model, live, on a slice of dates. Count distinct outputs: the check no validity test can perform. Compare with the shipped file and with the commercial column. |
| 82–92 | Real news | Coverage of the aligned headline file; `dated+news` against the `dated` control, on the full sample and on the covered subset; does the effect scale with headline count; then the same test on all five assets. |
| 92–106 | Backtesting | Kupiec, Christoffersen, pinball, leaderboard, then the DM test on every gap in it. |
| 106–114 | Disagreement | Daily spread band, pairwise correlations, the widest day. |
| 114–120 | Close | Three conclusions; three extensions (1% level, the `dated` probe, a different asset). |

**Five questions are embedded in the notebook**, one per section, and they are the
assessment. None has a lookup answer; each asks the student to interpret a number they
just produced. Question 4a — how many pairwise gaps survive the DM test, and what
sentence a report should therefore contain — is the one to discuss as a group.

**Infrastructure.** Students need the notebook plus either the repository or
`aida_lab_data.zip` (1.4 MB). **No API key of any kind is required** — every
language-model forecast and every headline is shipped as data, and the bundle was
tested by unpacking it into an empty directory with all three keys unset. The only live
download is the Chronos checkpoint from Hugging Face, about 40 MB for Bolt-tiny;
`chronos-forecasting` installs from PyPI in the first code cell of Section 2.

---

## Empirical content

`august-course/`, five assets, 500 out-of-sample days each, alpha = 1% reported with the
5% column carried alongside, one frozen
protocol inherited from `aida-risk`. Every asset carries Chronos-Bolt, Chronos-T5,
aligned headlines and four LLM configurations; the deck's numbers are the S&P 500 run,
stated on the frame.

`src/05_verify_slide_claims.py` re-derives every number printed inline on a slide from
the data and fails on a mismatch, and additionally audits the news alignment on all five
assets. **61 claims, 0 failures.** One claim is deliberately not
machine-checked and is attributed on the slide: the predictive standard deviation of 0.14
against a realised 1.22 when Chronos is fed returns instead of prices, which belongs to
the certified `aida-risk` run.

---

## Changes from the design this session started from

The original plan proposed a bake-off: historical simulation, Chronos-Bolt at 5%, two LLM
configurations, TimesFM as a bonus, over a 40–60 day window with fabricated news
headlines. Three changes were made on evidence, and each is logged in `DECISIONS.md`:

- **The 5% Chronos-Bolt VaR does not exist.** It clamps to the 10% quantile on 100% of
  days. The lab now discovers this rather than being built on it, and this became the
  spine of the whole session.
- **40–60 days cannot support a backtest.** 500 days, with the power curve shown.
- **No fabricated headlines — real ones instead.** The original plan called for 20–30
  invented information packets per date. Headlines now come from EODHD (`src/06_news.py`)
  under an audited as-of rule, and the news effect is identified against a `dated`
  control rather than against the anonymised baseline, because headlines reveal the
  period on their own. 35,293 articles fetched, 2,128 admissible, 89% of dates covered,
  zero rule violations.

Also dropped: TimesFM, which in the certified run has no quantile below 10% either and
would repeat the Chronos-Bolt lesson at the cost of a second dependency-resolution
exercise. It stays available in `aida-risk/src/02d_timesfm.py` for anyone who wants it as
an extension.

Added, not in the original plan: the Diebold-Mariano test. A leaderboard whose top two
rows differ by 0.0008 in pinball loss is an ordering, not a finding, and without the test
the session would have taught students to over-read exactly the kind of table it is
warning them about.

## Reuse from existing material

- `beamer-deck/preamble.tex` — verbatim, via `\input{../../beamer-deck/preamble}`
- Institutional logos and `ql_logo.png` — via `graphicspath`
- `fig_02_point_vs_distribution` — Act I
- Sign, plotting and table conventions — from `aida-risk/src/common.py`
- Kupiec and Christoffersen implementations — reimplemented in `notebook/aidalab.py`
  with the same maths as `aida-risk/src/03_backtest.py`, so students read one file
  rather than a package

## Open

- **Length.** 45 frames against the repository's ~75-frame guidance for a lecture. At
  a normal pace this delivers about 90 minutes, leaving 30 for discussion, which suits a
  physical session but is a judgement for the coordinator. Act II is the natural place to
  expand if a fuller two hours of lecture is wanted.
- **The `\qlbase` URL** (`github.com/QuantLet/AIDA2026`) still does not exist, carried
  over from the July deck. No Quantlet links are used in this deck, so nothing 404s here.
- **Every group has a complete set.** All five assets carry Chronos-Bolt, Chronos-T5,
  aligned headlines and four LLM configurations, so no group opens a notebook with an
  empty column. Coverage of the news leg differs by asset and is stated in the README:
  the Nikkei is the thin one at 51% of dates, and a group working on it should read its
  news result as underpowered rather than as a null.

# August 2026 — physical, 4 hours
## AI in Financial Risk Modelling: can AI produce valid risk forecasts?

**Status: built.** Deck is `slides/ai_risk_course_2026-08.tex` (66 content frames plus 8 act separators, 75 pages,
alpha = 1%, IDA two-level bullets, 21 quantlet marks, 15 references). Laboratory is
`notebook/AIDA_Risk_Lab.ipynb` (57 cells, executes end to end with zero errors).
Empirical content is `august-course/`, derived from `aida-risk/`. See `DECISIONS.md`.

**Level: intermediate.** Assume the 27 July online hour, or its equivalent: VaR and ES
defined, the scale/shape split seen once, GARCH recognised. Act II re-establishes those
in twenty minutes for anyone who missed it.

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

1. **Chronos-T5 changes its answer between two identical runs.** Same weights, same
   prices, same dates: only the number of sampled paths differs, and the two estimates
   part on 29% of days, by one token each time.
2. **A language model returns a number that parses, has the right sign and the right
   ordering** and still repeats itself on 89% of days.
3. **An open-weights model passes every structural check** and is the model the backtest
   rejects on all five markets, at realised rates of 4.8% to 13.2%.
4. **A 500-day backtest at 1% passes eight models of ten**, because five expected
   breaches cannot separate a count of 2 from a count of 7.

**The single sentence they should leave with:** a pretrained model will always answer;
validation is the only thing that tells you whether it answered *your* question.

**And the answer to the title question is split.** NN-t and Claude Haiku 4.5 pass
coverage on all five markets and score level with GARCH-t; Chronos-T5 and Qwen2.5-1.5B
are rejected. So the answer depends on which AI, which is the finding rather than a
hedge.

---

## Hour 1 and 2 — lecture (66 content frames, 75 pages)

| Act | Pages | Minutes | Content |
|---|---|---|---|
| Motivation | 2–6 | 0–8 | Pele et al. (2026) as the starting point; the research question; four ways to produce tomorrow's 1% VaR; the outline and the learning outcomes. |
| I — What is a risk forecast? | 8–13 | 8–22 | Forecast versus risk model; VaR and ES; coherence and the axiom VaR loses; the two factors and their unequal evidence; the four questions people ask of one table. |
| II — The classical answers | 15–25 | 22–48 | The data and the train/test split; historical simulation and its steps; GARCH as scale-plus-assumed-shape; quantile regression and the pinball loss; why 1% is the hard case; where AI can actually help; NN-t. |
| *break* | | 48–58 | |
| III — Time-series foundation models | 27–32 | 58–74 | Supervised / pretraining / fine-tuning / zero-shot; point versus marginal quantiles versus a predictive law; Chronos as a language over series; prices rather than returns; **the sampling floor and the two-run experiment**. |
| IV — LLMs as risk forecasters | 34–44 | 74–98 | Five things people mean by "LLMs for risk"; the Qwen architecture end to end; sampling versus elicitation; **both prompts in full**; the prompt as a modelling decision; **why the control is `dated`**; distinct-value counting; the seven routes to one quantile. |
| V — Validation | 46–59 | 98–116 | Kupiec and Christoffersen; the power curve; every forecast on one axis; the results table; **Diebold-Mariano**; replication on five markets; the news effect; the leakage taxonomy; disagreement. |
| Conclusions | 60 | 116–118 | YES and NO by model family, and what the two passes actually bought. |
| Lab brief + references | 61–64 | 118–120 | What you will do this afternoon; what produced every number. |
| Appendix | 65–75 | — | Derivations, the conventions, the full test expressions, SYNCRISK. |

**Load-bearing frames.** "The same model, the same day, two different VaRs" (p32) and
"Counting the distinct answers, and what it does not prove" (p43). Everything in Acts III
to V is a variation on the point they make: an answer that looks like a quantile is not
evidence that a quantile was computed. If time runs short, compress Act II, never those
two.

**Two frames that must not be softened.** "Eight of ten pass, and that says more about
the test" (p51) reports that the coverage test cannot separate a breach count of 2 from
one of 7 on this span. "Does the language model match the benchmark?" (p53) reports that
matching GARCH-t is the best result on offer, on four of five markets by the test's
silence and on the fifth by losing. A deck that keeps the leaderboard and drops these two
would make a claim the data does not support.

---

## Hour 3 and 4 — laboratory

One notebook, seven sections, one asset per group (S&P 500, DAX 40, Nikkei 225, gold
futures, Bitcoin). Compute is about 14 minutes of the two hours, and 11 of those are the
optional live open-weights cell; the rest of the time is writing, reading and arguing.

| Minutes | Section | What happens |
|---|---|---|
| 0–10 | 1. Setup and reproducibility checks | The repository clones into the runtime; versions and seed are printed; the panel loads. |
| 10–18 | 1.1 Power | The power cell shows why the span is 500 days rather than 50, and what a 500-day span can and cannot reject. |
| 18–33 | 2. Historical simulation | Students write `my_hs_var` themselves and check it against the reference. The window ends at `t-1`; the leak is discussed rather than merely avoided. |
| 33–45 | 3. Chronos-T5 sampling resolution | 12 dates at 500 paths, run live (~2 min), then the same dates at a different draw count: how much of the forecast is the model and how much is the sampler. |
| 45–57 | 4. LLM parsing and logical checks | Four configurations loaded; parse, sign and ordering failures counted before any result is read; distinct-value share computed. |
| 57–70 | 5. Dated versus dated+news | `dated+news` against the `dated` control on the covered subset, then the same test on all five assets. |
| 70–85 | 5.1 Open weights live | Qwen2.5-1.5B, live, on 16 dates. Compare with the shipped file and with the commercial column. Optional: `RUN_OPEN_LIVE = False` uses the shipped run. |
| 85–105 | 6. Coverage, loss and disagreement | Kupiec, Christoffersen, pinball, the leaderboard, the DM test on every gap in it, and the disagreement band. Three exercises are written here: the distinct-value audit, the Kupiec statistic from scratch, and the group's own asset rerun end to end. |
| 105–120 | 7. Close and extensions | Five questions; three extensions (a different asset, the 1% level everywhere, and the one extension a rented endpoint blocks). |

**Five questions are embedded in the notebook**, one per section, and they are the
assessment. None has a lookup answer; each asks the student to interpret a number they
just produced.

**Infrastructure.** Students need the notebook plus either the repository or
`aida_lab_data.zip`. Everything runs on open weights and shipped data: every commercial
language-model forecast and every headline arrives as a file, and the bundle was tested
by unpacking it into an empty directory with all three API keys unset. Two live
downloads: the Chronos-T5 checkpoint (~80 MB) in section 3, and Qwen2.5-1.5B (3.1 GB) in
the optional section 5.1.

---

## Empirical content

`august-course/`, five assets, 500 out-of-sample days each, alpha = 1% reported with the
5% column carried alongside, one frozen protocol inherited from `aida-risk`. Every asset
carries Chronos-T5, aligned headlines and four LLM configurations; the deck's numbers are
the S&P 500 run, stated on the frame.

Ten models are scored: historical simulation, GARCH(1,1)-t, NN-t, Chronos-T5, four Claude
Haiku 4.5 configurations, and Qwen2.5-1.5B both asked for the quantile and sampled 1000
times. Qwen-asked scores 494 days rather than 500, because structurally invalid replies
are dropped and counted rather than repaired.

`src/05_verify_slide_claims.py` re-derives every number printed inline on a slide from
the data and fails on a mismatch, and additionally audits the news alignment on all five
assets. **200 claims, 0 failures.** One claim is deliberately not machine-checked and is
attributed on the slide: the NN-t architecture selection, which belongs to the certified
`aida-risk` run.

`src/06_lint_slides.py` enforces the deck's own rules — frame titles, two-level bullets,
five items per list, banned phrasings, citation resolution, figure captions — and every
rule in it was a correction made by hand at least once.

News coverage differs by asset and is stated in the README:

| asset | days covered of 500 | admissible headlines |
|---|---|---|
| S&P 500 | 447 (89%) | 2,128 |
| DAX 40 | 386 (77%) | 955 |
| Nikkei 225 | 254 (51%) | 449 |
| Gold futures | 391 (78%) | 1,352 |
| Bitcoin | 500 (100%) | 2,481 |

A group working on the Nikkei should read its news result as underpowered rather than as
a null.

---

## Changes from the design this session started from

The original plan proposed a bake-off: historical simulation, Chronos-Bolt at 5%, two LLM
configurations, TimesFM as a bonus, over a 40–60 day window with fabricated news
headlines. Each change below was made on evidence and is logged in `DECISIONS.md`:

- **The 5% Chronos-Bolt VaR does not exist.** It returns the 10% quantile on 100% of
  days, because it was trained on the grid 0.1, 0.2, ..., 0.9. Since the course is scored
  at 1%, a model whose lowest usable level is 10% has no comparable number to contribute,
  so Chronos-Bolt is scored nowhere. The measurement is kept: `src/02_chronos_lab.py`
  still has the `bolt` path and its output stays in `precomputed/chronos_bolt_*.csv`. The
  argument it used to carry is now made by Chronos-T5's two-run experiment, which uses a
  model the course actually scores.
- **40–60 days cannot support a backtest.** 500 days, with the power curve shown.
- **Real headlines rather than fabricated ones.** The original plan called for 20–30
  invented information packets per date. Headlines come from EODHD (`src/06_news.py`)
  under an audited as-of rule, and the news effect is identified against a `dated`
  control rather than against the anonymised baseline, because headlines reveal the
  period on their own.

Also dropped: TimesFM, which in the certified run has no quantile below 10% either and
would repeat the Chronos-Bolt lesson at the cost of a second dependency-resolution
exercise. It stays available in `aida-risk/src/02d_timesfm.py` for anyone who wants it as
an extension.

Added, not in the original plan: the Diebold-Mariano test. A leaderboard whose top three
rows differ by 0.0008 in pinball loss is an ordering rather than a finding, and without
the test the session would teach students to over-read exactly the kind of table it is
warning them about.

## Reuse from existing material

- `beamer-deck/preamble.tex` — verbatim, via `\input{../../beamer-deck/preamble}`
- Institutional logos and `ql_logo.png` — via `graphicspath`
- Sign, plotting and table conventions — from `aida-risk/src/common.py`
- Kupiec and Christoffersen implementations — reimplemented in `notebook/aidalab.py`
  with the same maths as `aida-risk/src/03_backtest.py`, so students read one file
  rather than a package

## Open

- **Length.** 75 pages, at the repository's ~75-frame guidance for a lecture. At a normal pace this delivers about two hours including the break, which is
  what the session is scheduled for.
- **The published sampling protocol on the S&P.** `src/09_llmtime.py` gained `--wrapper`
  and `--scaler`, which reproduce the instruction wrapper and the affine scaler of
  Pele et al. (2026). A four-date smoke test ran wider than the current leg, mean 1% VaR
  2.12 against 1.93, so `run_after_course.sh` runs the full 500 days (~25 hours at
  N = 1000) and the comparison is made afterwards.
- **The alpha = 5% news effect.** An earlier version of the deck quoted p = 0.61;
  recomputation gives p = 0.065. The claim appears nowhere in the current deck, so
  nothing rests on it, and it is unreconciled.

# Changelog

## 2026-08-04 — teachability revision

The course was scientifically complete but too dense for the time available. This
revision changes pacing, structure and wording. **No empirical result changed.**

### Structural changes

- **Core lecture cut from 58 to 46 content frames**, plus an 11-frame technical
  appendix. Frames per act: opening 2, Act I 3, Act II 5, Act III 7, Act IV 14,
  Act V 12, laboratory and close 3.
- **Overlays removed earlier in the week stay removed**: one page per frame, 67 pages
  for 66 frames.
- **New: learning outcomes.** "By the end of the session, you should be able to…"
  replaces the old roadmap and carries the four operational outcomes; the empirical-design
  summary survives as a short block on the same frame.
- **New: the validation triad**, at the end of Act I and before any empirical
  comparison — RANKING, VALIDITY, POWER, plus PIPELINE AUDIT as the check that precedes
  all three. Seventeen empirical frames now carry the label of the question they answer.
- **New: a capability-check frame** opening the Chronos-Bolt sequence, which is now
  three frames: what levels the model supports, what happens when you ask for 1%, and
  what the episode is evidence about.
- **New: one comparison table**, after all seven ways of producing a quantile have been
  introduced — input, native output, VaR extraction, hidden failure, first check.
- **Merged**, each pair carrying one idea: forecast-versus-risk-model with the VaR/ES
  definitions; the two-factor split with "why 1% is different"; the four pretraining
  words with the three output types; the sampling floor with its measurement; the five
  failure modes with the schema audit that settles three of them.
- **Conclusion rewritten** to five messages, followed by a closing frame that puts the
  title question and answers it. The old "what AI adds to the work of checking" frame
  folded into message 5.

### Slides moved to the appendix

| frame | reason |
|---|---|
| What makes a loss the right loss? | formal consistency and elicitability definitions |
| Point forecast, or the whole predictive distribution? | illustration, superseded by the merged Act III frame |
| The published study, on the S&P 500 | one of two published-study illustrations |
| What the feed gave us | feed diagnostics, not needed for the argument |
| Coverage and ranking, side by side | the same content as the results table |
| Kupiec and Christoffersen in full | full likelihood-ratio expressions |
| Comparing two models on the same days | the Diebold–Mariano statistic and its HAC variance |
| Scoring ES means scoring the pair | the FZ₀ loss |
| The minimum sits at the truth in both coordinates | the joint-minimum surface |
| SYNCRISK: when every desk runs the same model | the formal definition |

The core keeps the intuitive statements, the breach indicator, what each coverage test
tests, why five expected breaches imply weak power, and why ranking and acceptance are
different operations. Two navigation pointers were added ("Full likelihood-ratio
expressions are in the appendix", and the same for the FZ₀ loss).

A one-page prerequisite frame opens the appendix, and the same material ships as
`HANDOUT_prerequisites.md`.

### Wording corrections

| was | now |
|---|---|
| "It therefore cannot react." | "It reacts only discretely, when observations enter or leave the window" — coarse, window-dependent, equally weighted unless a weighted variant is used |
| "it is not sub-additive" | "not sub-additive in general" |
| "Two factors, and only one of them has data" | "Two factors, but only one is richly identified" |
| "With 20 samples the model cannot express a 1% quantile" | "cannot resolve a 1% quantile below the 1/N Monte Carlo granularity" — a library may still return an interpolated value or the sample minimum, which is not tail resolution |
| "The bigger the model, the less it says" | "In this setup, larger Qwen models give fewer distinct forecasts" — conditional on this family, prompt, decoding configuration and elicitation protocol, and not a scaling law |
| "Contamination is invisible in principle" | "With undisclosed training data, contamination cannot be ruled out from outputs alone" — risk, evidence, retrospective recall and genuine forecasting named as four different claims |
| "A flexible model does not create tail observations. It redistributes the ones you have." | "A flexible architecture cannot create additional realised tail events; any gain has to come from transferable structure, extra information or a modelling assumption" |
| "nobody can mis-specify historical simulation" | "there is no parametric distribution to mis-specify; window length and weighting remain modelling decisions" |
| "The tail shape is imported, not learned" | "The tail family is imposed; its parameters are estimated" |
| "Architecturally identical to Chronos-T5" | "Statistically analogous at the sampling-and-quantile stage" |
| "Chronos is trained on series levels" | "In the tested pipeline, price levels are the compatible input representation" |
| "The language model is tracking volatility" | "consistent with the language model primarily tracking volatility" |
| "Same information set" | "Same forecast dates, target and scoring rule" — the representation differs by construction |

### Readability

- Figures are now drawn near the size they are displayed at rather than at 13 inches and
  scaled down, so type in a panel is three times larger on the projector. Nothing in any
  figure is set below 12 pt in source; after scaling, no text on any page falls below
  5 pt except mathematical subscripts and the Quantlet marks.
- Legend and axis labels raised; long bar labels shortened ("Claude Haiku", "Qwen 14B");
  the correlation matrix uses short model keys.
- No overfull boxes anywhere in the deck. Figure sizes on frames were fitted per frame
  from the LaTeX log rather than shrunk uniformly.
- Red is used for failures and rejections only; blue for definitions and valid controls.

### Laboratory changes

- Renumbered into the eight sections of the revised design: setup and reproducibility,
  historical simulation, the Chronos-Bolt capability audit, the Chronos-T5
  sampling-resolution experiment, LLM parsing and logical checks, dated versus
  dated+news, coverage and loss and disagreement, and the optional extension.
- Every section header carries one of four badges: `REQUIRED`, `INTERPRET`, `OPTIONAL`,
  `SLOW / PRECOMPUTED FALLBACK AVAILABLE`.
- The disagreement and SYNCRISK material is marked optional.
- New: five interpretation questions before the closing section, one each on pipeline
  validity, ranking against calibration, power, contamination and experimental control.
- The laboratory brief on the slide states the execution plan: historical simulation and
  the whole Bolt experiment live, Chronos-T5 and the open model on 5–20 dates, the
  500-day panels loaded from `precomputed/`, and every test run on the full panel.

### Unchanged empirical results

Every number, sample period, model output, table and statistical conclusion is
unchanged. `src/05_verify_slide_claims.py` re-derives all of them from the data:
**96 claims, 0 failures**, before and after this revision.

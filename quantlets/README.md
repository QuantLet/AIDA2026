# Quantlets — AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

One directory per pipeline stage. Each carries its script, its dependencies, a
Metainfo.txt and a README, and is the citable form of the stage the deck's
`\quantlet{}` marks point at.

- **AIDA_aug_data** — Carves the laboratory dataset out of the certified aida-risk run: five assets, a common 500-day test span, and the classical VaR forecasts that span traces to.
- **AIDA_aug_chronos** — Chronos zero-shot VaR through both heads.
- **AIDA_aug_llm** — LLM-VaR through a commercial endpoint: the return series serialised as text, the tail quantile asked for in words, four information sets, and a parser that counts the replies that fail rather than dropping them.
- **AIDA_aug_local_llm** — The same experiment with open weights, so it can be reproduced without a commercial key.
- **AIDA_aug_news** — Real headlines from EODHD aligned to forecast dates under an audited as-of rule: each asset's own market close, consecutive cutoffs tiling the calendar, and every publication timestamp kept so the rule can be checked afterwards.
- **AIDA_aug_backtest** — Backtesting and scoring at alpha = 1%: Kupiec unconditional coverage, Christoffersen independence, the pinball loss, Diebold-Mariano with a HAC standard error, model disagreement, and a checker that re-derives every number printed on a slide.
- **AIDA_aug_lab** — The student notebook and its helper module: loaders, historical simulation, the pinball loss, the coverage tests, Diebold-Mariano and the plots, depending only on numpy, pandas, scipy and matplotlib so it runs on a bare Colab session.

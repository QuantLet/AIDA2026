# AIDA_aug_llm

LLM-VaR through a commercial endpoint: the return series serialised as text, the tail quantile asked for in words, four information sets, and a parser that counts the replies that fail rather than dropping them.

- **Datafile:** data/lab/returns_*.csv, data/news/headlines_*.csv
- **Output:** precomputed/llm_*.csv
- **Part of:** AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

`AIDA_aug_llm.ipynb` shows the result without running anything: 1 figure(s) at the folder root and 23 data file(s) under `output/`.

Reproduce with `../../run_all.sh` from the package root; this folder is the extracted, citable form of one stage.

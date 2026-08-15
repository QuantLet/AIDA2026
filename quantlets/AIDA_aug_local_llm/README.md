# AIDA_aug_local_llm

The same experiment with open weights, so it can be reproduced without a commercial key. Imports the prompt and parser from the commercial stage, so the two legs differ in exactly one thing: which model reads the prompt. Also runs the LLMTime sampling route, where the quantile is an order statistic of 1000 generated continuations.

- **Datafile:** data/lab/returns_*.csv
- **Output:** precomputed/local_*.csv, llmtime_*.csv
- **Part of:** AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

`AIDA_aug_local_llm.ipynb` shows the result without running anything: 3 figure(s) at the folder root and 16 data file(s) under `output/`.

Reproduce with `../../run_all.sh` from the package root; this folder is the extracted, citable form of one stage.

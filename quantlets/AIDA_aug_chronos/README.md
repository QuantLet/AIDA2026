# AIDA_aug_chronos

Chronos-T5 zero-shot VaR: the last 512 closing prices are handed to the pretrained model, 500 paths are sampled for the next day, and the tail quantile is read off the draws as an order statistic.

- **Datafile:** data/lab/returns_*.csv
- **Output:** precomputed/chronos_t5_*.csv
- **Part of:** AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

`AIDA_aug_chronos.ipynb` shows the result without running anything: 2 figure(s) at the folder root and 12 data file(s) under `output/`.

Reproduce with `../../run_all.sh` from the package root; this folder is the extracted, citable form of one stage.

# AIDA_aug_chronos

Chronos zero-shot VaR through both heads. The quantile head is asked for levels below its trained grid and returns the clamped value, which the script records rather than repairs; the sampling head supplies the genuine tail quantile.

- **Datafile:** data/lab/returns_*.csv
- **Output:** precomputed/chronos_{bolt,t5}_*.csv
- **Part of:** AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

Reproduce with `../../run_all.sh` from the package root; this folder is the extracted, citable form of one stage.

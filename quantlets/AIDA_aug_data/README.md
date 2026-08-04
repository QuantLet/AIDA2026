# AIDA_aug_data

Carves the laboratory dataset out of the certified aida-risk run: five assets, a common 500-day test span, and the classical VaR forecasts that span traces to. Nothing is downloaded and nothing is simulated.

- **Datafile:** aida-risk/data/raw/*.csv
- **Output:** data/lab/returns_*.csv, bench_*.csv
- **Part of:** AIDA Summer School 2026 - AI in Financial Risk Modelling (August course)

Reproduce with `../../run_all.sh` from the package root; this folder is the extracted, citable form of one stage.

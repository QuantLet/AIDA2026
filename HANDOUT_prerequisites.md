# One page before the session

*AIDA Summer School 2026 — AI in Financial Risk Modelling.* Everything below is assumed
from the first slide. Nothing else is.

## Signs, and the objects we score

| symbol | meaning |
|---|---|
| $r_t$ | log return on day $t$, in per cent |
| $L_t = -r_t$ | the loss |
| $q_t(\alpha) = F_t^{-1}(\alpha) < 0$ | the **return quantile**: a negative number |
| $\mathrm{VaR}_t(\alpha) = -q_t(\alpha) > 0$ | **value at risk**, always a positive magnitude |
| $\mathrm{ES}_t(\alpha) = \mathbb{E}[L_t \mid L_t > \mathrm{VaR}_t(\alpha)]$ | **expected shortfall**, the mean of the tail beyond VaR |
| $I_t = \mathbf{1}\{r_t < -\mathrm{VaR}_t(\alpha)\}$ | the **breach indicator**: 1 on a day the threshold was crossed |

Two conventions hold everywhere in the course, and mixing them is the most common source
of confusion:

- **VaR and ES are reported as positive magnitudes.** A VaR of 2.3 means a loss of 2.3%.
- **Every figure is drawn in return space**, where the same object is the negative
  threshold $q_t(\alpha)$ and the tail of interest is on the left.

## The two factors behind any VaR

$$q_t(\alpha) = \underbrace{\sigma_t}_{\text{scale}} \times \underbrace{q_\nu(\alpha)}_{\text{shape}}$$

The **scale** is observed and updated every day: volatility clusters, so it is
forecastable and a learning algorithm has data to learn from. The **shape** is identified
by the few observations that actually fall in the tail. Most of the course is about which
of the two a given model got wrong.

## The loss that ranks quantile forecasts

$$L_\alpha(r, q) = \left(\alpha - \mathbf{1}\{r < q\}\right)(r - q)$$

This is the **pinball** (or check) loss. It is *strictly consistent* for the quantile:
its expectation is minimised at the true $q_t(\alpha)$ and nowhere else, which is what
makes it the right thing to rank models on. The breach count is not.

At $\alpha = 1\%$ the two arms have slopes $0.01$ and $-0.99$: a threshold that sits too
close to zero is punished far harder than one that sits too far away.

## How many breaches to expect at 1%

| out-of-sample days | expected breaches |
|---|---|
| 250 | 2.5 |
| 500 | **5** |
| 1000 | 10 |
| 5541 | 55 |

Five events is the entire evidence base a 500-day backtest has at the 1% level. That
single fact is why the course separates three questions that are usually run together:

- **ranking** — which forecast has the lower consistent loss?
- **validity** — is this forecast compatible with the target coverage and dynamics?
- **power** — would the test have detected a materially invalid model on this sample?

## What you need for the laboratory

A laptop, a browser and a Google account. The notebook runs in **Google Colab** — open
it from the badge at the top of `notebook/AIDA_Risk_Lab.ipynb` — so there is **nothing
to install**, and **no API key is needed at any point**. The language-model forecasts
and the aligned headlines ship as data.

One thing to plan around. Colab keeps nothing between sessions, so the 3.1 GB of Qwen
weights behind the optional live open-weights leg are fetched inside the Colab machine
every time; you cannot pre-download them at home. Start that cell early, and set the
runtime to a T4 GPU: about one minute of compute against eleven on the free CPU runtime.
Skipping it costs you nothing else — the shipped file covers all 500 days, and every
`SLOW` cell has the same fallback.

Running locally instead works too: Python 3.11+ with `numpy`, `pandas`, `scipy` and
`matplotlib`.

# Figures from the ESWA paper

`eswa_ES_LLM_120.png` and `eswa_VaR_exceedances_llmtime_120.png` are the published
figures of

> Pele, D. T., Bolovăneanu, V., Lin, M.-B., Ren, R., Ginavar, A. T., Spilak, B.,
> Andrei, A.-V., Toma, F.-M., Lessmann, S., & Härdle, W. K. (2026). In the beginning was
> the Word: LLM-VaR and LLM-ES. *Expert Systems with Applications, 295*, 128676.
> https://doi.org/10.1016/j.eswa.2025.128676

retrieved on 2026-08-02 from the paper's own code repository,
https://github.com/QuantLet/LLM_Risk (GPT-4 results, 120-day context).

`eswa_var_sp500.png` and `eswa_dist_sp500.png` are the S&P 500 panel cropped out of
each, because the 3x3 asset grid is unreadable at slide size. The crop is reproducible:
the S&P 500 panel is row 2, column 2, and the deck states that it is one of nine.

Colour semantics come from that repository's plotting code, not from inspection:
on the threshold plot black is the VaR path, green marks positive returns, orange
negative returns that did not breach, and red the breaches; on the density plot blue is
the realised return distribution and red the model's draws.

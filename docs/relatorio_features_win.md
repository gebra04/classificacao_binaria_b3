# Relatório de Engenharia e Seleção de Features WIN

## Top 20 Features (IC Global)
| Feature | IC Spearman |
|---|---|
| close_position_d1 | 0.0963 |
| body_pct_d1 | 0.0932 |
| spx_slope_20d | 0.0760 |
| ewz_close | 0.0744 |
| spx_ret_neg | 0.0642 |
| winfut_change_pct | 0.0641 |
| ret_close_d1 | 0.0639 |
| ret_lag1 | 0.0639 |
| winfut_close | 0.0625 |
| spx_drawdown_60 | 0.0574 |
| spx_zscore | 0.0558 |
| nasdaq100_change_pct | 0.0554 |
| ndaq_ret_d1 | 0.0554 |
| ipca_var_mensal_pct | 0.0549 |
| usd_brl_change_pct | 0.0524 |
| usd_brl_ret_d1 | 0.0524 |
| minerio_ferro_maxima | 0.0513 |
| minerio_ferro_abertura | 0.0513 |
| minerio_ferro_ultimo | 0.0512 |
| minerio_ferro_minima | 0.0511 |

## Top 5 Features por Regime
### Regime: bull_forte
| Feature | IC Spearman |
|---|---|
| spx_slope_20d | 0.1506 |
| us2y_slope_20d | 0.1119 |
| petroleo_brent_close | 0.1009 |
| minerio_ma_regime | 0.0984 |
| us10y_slope_20d | 0.0921 |
### Regime: bull_fraco
| Feature | IC Spearman |
|---|---|
| win_vs_commodities | 0.1208 |
| regime_di_subindo | 0.1150 |
| win_vs_spx | 0.1106 |
| ouro_ma_regime | 0.1084 |
| body_pct_d1 | 0.1028 |
### Regime: bear_fraco
| Feature | IC Spearman |
|---|---|
| close_position_d1 | 0.1505 |
| body_pct_d1 | 0.1249 |
| lower_shadow_d1 | 0.1205 |
| winfut_change_pct | 0.1059 |
| ret_close_d1 | 0.1056 |
### Regime: bear_forte
| Feature | IC Spearman |
|---|---|
| body_pct_d1 | 0.2049 |
| ret_lag10 | 0.1711 |
| vix_change_pct | 0.1342 |
| vix_ret_d1 | 0.1342 |
| ewz_ret_d1 | 0.1338 |

## Multicolinearidade
**Removidas (15):** sp500_close, nasdaq_composite_close, minerio_ferro_abertura, minerio_ferro_maxima, minerio_ferro_minima, evento_g6_global_pib_eua, ret_close_d1, ret_lag1, spx_ret_d1, ndaq_ret_d1, vix_ret_d1, usd_brl_ret_d1, di1_ret_d1, minerio_ret_d1, regime_spx_bull

## Resultado Final
De 159 features originais, restaram **69** features finais robustas.
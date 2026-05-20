# PLANO MESTRE — Modelo Preditivo WIN (Mini-Índice Bovespa)
**Versão:** 2.0 — Dataset Real Integrado  
**Dataset:** `base_concatenada_v1.csv` tratado → `base_win_tratada_final.csv`  
**Granularidade disponível:** Diária (D1) — 2015-01-02 a 2026-05-20  
**Atualizado em:** 20/05/2026

---

## Sumário Executivo

Este plano descreve a construção de um modelo de classificação binária para prever a direção do candle diário do WIN (Mini-Índice Bovespa). O dataset disponível é diário — não intraday — o que elimina features de microestrutura (VWAP, candles de 15min, book) e desloca o foco para drivers macro, intermercado e padrões de regime.

O projeto segue uma arquitetura sequencial em sete fases: engenharia de features → baselines triviais → validação walk-forward → Modelo 1 (Logistic Regression) → Modelo 2 (HMM + XGBoost regime-conditioned) → Modelo 3 (TFT) → ensemble. Cada fase tem critérios explícitos de avanço. Nenhum modelo avança sem superar os baselines com métricas financeiras reais.

| Parâmetro | Definição |
|---|---|
| Ativo | WIN — Mini Índice Bovespa (B3) |
| Problema | Classificação binária: alta (0) vs baixa (1) do candle diário |
| Timeframe | Diário (D1) |
| Dataset | 2.826 pregões · 2015–2026 · 75 colunas |
| Modelos | Logistic Regression → HMM+XGBoost → TFT (sequencial) |
| Validação | Walk-forward purged com embargo — mínimo 5 folds |
| Critério final | Profit factor OOS > 1.3 · Sharpe > 0.8 · Drawdown < 20% |

---

## 1. Especificidades do WIN — Por Que Não É Apenas o WDO com Outro Nome

O WIN e o WDO são ativos brasileiros negociados na B3, mas têm dinâmicas fundamentalmente diferentes. Ignorar essas diferenças é a principal causa de modelos que funcionam em backtest e falham fora da amostra.

### 1.1 Driver Primário: Risco Global, Não Câmbio

| Dimensão | WDO | WIN |
|---|---|---|
| Driver primário | DXY (dólar global) — o WDO sobe quando o dólar sobe globalmente | SPX/Nasdaq — o WIN cai quando o risco global cai, independente do câmbio |
| VIX | Fator secundário — afeta pelo canal de emergentes | Fator primário — picos de VIX transmitem diretamente para o WIN via fluxo |
| DI Futuro | Fator endógeno direto — juros afetam câmbio via carry trade | Fator indireto — juros altos competem com renda variável (custo de oportunidade) |
| Commodities | Relevante via balança comercial (petróleo, ouro) | Minério de ferro é crítico — Vale representa >10% do Ibovespa |
| WDO/câmbio | É o próprio ativo | Feature relevante — real fraco = pressão vendedora em bolsa via fluxo estrangeiro |
| Fluxo estrangeiro | Impacto moderado | Impacto alto — investidor estrangeiro representa ~50% do volume do Ibovespa |

### 1.2 Viés de Alta Secular — Problema de Desbalanceamento

O Ibovespa tem viés de alta estrutural de longo prazo. No dataset real (2015–2026), a proporção de dias de alta varia de 46% (2015, bear) a 57% (2019, bull). Isso cria três problemas específicos:

> **Risco central:** Um modelo que aprende apenas o viés de alta do período de treino vai ter acurácia aparentemente alta, mas vai falhar sistematicamente em bear markets. Não é sinal — é viés de amostra.

- **Problema 1 — Acurácia enganosa:** modelo que prevê "alta" todo dia terá acurácia de 55%+ em bull market, mas não tem valor preditivo.
- **Problema 2 — Threshold viciado:** sem calibração explícita, o modelo vai preferir a classe majoritária mesmo quando o sinal aponta para baixa.
- **Problema 3 — Fold contaminado:** um fold de treino em bull market e teste em bear market vai parecer que o modelo degradou, mas pode ser apenas regime shift.

**Distribuição real de classes no dataset (% dias de alta por ano):**

| Ano | % Alta | Regime | Vol Anual |
|---|---|---|---|
| 2015 | 46.2% | BEAR FRACO | 23.7% |
| 2016 | 53.0% | BULL FORTE (+43%) | 27.9% |
| 2017 | 56.3% | BULL FORTE (+26%) | 19.9% |
| 2018 | 51.4% | BULL FRACO (+15%) | 22.6% |
| 2019 | 56.9% | BULL FORTE (+32%) | 18.6% |
| 2020 | 52.2% | BULL FRACO (+2%) COVID | 46.1% |
| 2021 | 51.6% | BEAR FRACO (-12%) | 21.5% |
| 2022 | 52.4% | BULL FRACO (+4%) | 21.5% |
| 2023 | 51.2% | BULL FORTE (+23%) | 18.3% |
| 2024 | 49.8% | BEAR FRACO (-10%) | 13.7% |
| 2025 | 54.0% | BULL FORTE (+35%) | 16.1% |

**Mitigações obrigatórias:**

```python
# 1. Medir desbalanceamento por fold e por período
for fold in folds:
    print(fold.target.value_counts(normalize=True))
    print(f'Regime: {identificar_regime(fold)}')  # bull / bear / lateral

# 2. class_weight='balanced' no sklearn; scale_pos_weight no XGBoost
scale_pos_weight = n_baixa / n_alta  # se n_alta > n_baixa

# 3. Usar balanced_accuracy como métrica de ML principal (não acurácia simples)
from sklearn.metrics import balanced_accuracy_score
# balanced_accuracy = média de recall por classe — imune ao viés de alta

# 4. Avaliar métricas financeiras separadas por regime (bull / bear)
```

### 1.3 Regime Shift de Longo Prazo — O Maior Risco do WIN

O Ibovespa passou por regimes radicalmente distintos nos últimos anos:

| Período | Regime | Características |
|---|---|---|
| 2015 | Bear/recuperação | Crise Brasil, Dilma, Selic subindo |
| 2016–2019 | Bull market / recuperação | Correlação alta com SPX, fluxo estrangeiro positivo, Selic caindo |
| 2020 (Mar) | Crash COVID — queda de 45% em semanas | VIX > 80, fluxo estrangeiro saindo, correlações colapsam |
| 2020–2021 | Recuperação V / bull market global | SPX e WIN em alta simultânea, liquidez global abundante |
| 2022–2023 | Ciclo de juros altos Brasil e EUA | DI > 13%, custo de oportunidade alto, Ibovespa lateral/pressionado |
| 2024–2025 | Novo ciclo | Transição de regime — comportamento distinto |

**Por que isso importa:** Um modelo treinado em 2016–2020 vai aprender a comprar em dips do SPX. Isso funcionava. Em 2022–2023, com Selic a 13%, a dinâmica era diferente: renda variável competia com renda fixa de forma inédita. O modelo precisa identificar em qual regime está operando.

**Mitigações obrigatórias:**
- Feature de regime explícita: criar variável que identifica o regime atual (bull/bear/lateral/juros_altos)
- Walk-forward com análise por regime: reportar desempenho separado por tipo de período
- KS test entre folds: detectar quando a distribuição de features muda significativamente entre treino e teste
- Expanding window preferida sobre rolling: inclui mais contexto histórico de regime

### 1.4 Assimetria de Transmissão SPX → WIN

A correlação entre SPX e WIN não é simétrica. Quedas do SPX transmitem para o WIN de forma mais rápida, intensa e persistente do que altas. Isso tem implicações diretas para o feature engineering:

```python
# Não usar só o retorno do SPX — capturar a assimetria explicitamente
df['spx_ret_pos'] = df['sp500_change_pct'].clip(lower=0)   # altas do SPX
df['spx_ret_neg'] = df['sp500_change_pct'].clip(upper=0)   # quedas do SPX
df['spx_queda_5d'] = df['sp500_change_pct'].rolling(5).sum().clip(upper=0)

# Intensidade do movimento: retorno / volatilidade recente
df['spx_zscore'] = df['sp500_change_pct'] / df['sp500_change_pct'].rolling(20).std()
df['spx_drawdown'] = df['sp500_close'] / df['sp500_close'].rolling(60).max() - 1

# VIX: nível absoluto importa tanto quanto a variação
df['vix_spike'] = (df['vix_change_pct'] > 0.15).astype(int)  # alta > 15% no VIX em 1 dia
df['vix_regime'] = pd.cut(df['vix_close'],
    bins=[0, 15, 20, 25, 35, 100],
    labels=['baixo','normal','elevado','alto','crise'])
```

---

## 2. Definição Formal do Problema

### 2.1 O Problema Central

**Definição:** O modelo prevê a direção do candle diário do WIN. Cada linha do dataset é um pregão. A decisão é tomada no fechamento do dia anterior (D-1). Não há operação intraday.

```python
retorno_dia = (close_D - open_D) / open_D

# Limite adaptativo por volatilidade — nunca menor que o custo estimado
custo_estimado = 0.0004  # ~4 pontos WIN (spread + corretagem)
limite = max(
    df['retorno_oc'].abs().rolling(252).quantile(0.30),
    custo_estimado * 2
)

target_D = 0    se retorno_dia >  limite   # dia de alta
target_D = 1    se retorno_dia < -limite   # dia de baixa
target_D = NaN  se abs(retorno_dia) <= limite  # zona neutra — excluído
```

> **Por que o limite adaptativo é crítico para o WIN:** Em períodos de alta volatilidade (VIX > 25, como 2020), movimentos de 0.5% são comuns e ruidosos. O limite adaptativo se ajusta automaticamente ao regime, evitando que o modelo aprenda ruído em períodos turbulentos.

### 2.2 Targets Alternativos para Diagnóstico

| ID | Definição | Quando usar |
|---|---|---|
| Target A | Retorno open→close do dia D (principal) | Alinhado com quem entra no open e sai no close. Exclui o impacto do gap overnight. |
| Target B | Retorno close→close (D-1 para D) | Inclui o gap overnight. Mais previsível em dias com eventos macro noturnos (Fed, dados EUA). |
| Target C | Retorno vol-ajustado: ret_D / std_rolling(20d) | Normaliza por regime de volatilidade — menos sensível a crashes e períodos de VIX alto. Preferido para comparação entre regimes. |
| Target D | Direção do fechamento vs abertura: sign(close_D - open_D) | Versão binária pura sem limite — serve como sanity check mas inclui muito ruído. |

### 2.3 Momento da Decisão e Fluxo de Informação

| Informação | Status no fechamento de D-1 |
|---|---|
| Candle diário D-1 OHLCV | Completo e disponível |
| SPX fechamento D-1 | Disponível (NYSE fecha ~18h Brasília) |
| VIX fechamento D-1 | Disponível |
| DXY, US10Y, US2Y D-1 | Disponíveis |
| DI, USD/BRL, WIN D-1 | Disponíveis |
| SELIC atual | Disponível (dado público) |
| IPCA divulgado mais recente | Disponível |
| Eventos de calendário para D | Disponível (FOMC, COPOM, Payroll têm datas públicas) |
| Open do WIN em D | **NÃO disponível** — só existe no dia seguinte |
| Notícias de D | **NÃO usar** — apenas calendário econômico previamente publicado |

---

## 3. Dataset Base

### 3.1 Estrutura Real (pós-tratamento)

| Campo | Conteúdo | Regra temporal |
|---|---|---|
| `data` | Data do pregão D | Índice principal |
| `target` | 0 (alta) ou 1 (baixa) do candle diário de D | `shift(-1)` em relação às features |
| WIN OHLCV | open, high, low, close, volume de D-1 | Disponíveis no fechamento de D-1 |
| FX / DI1 / US Rates | USD/BRL, DI1, US2Y, US10Y — OHLC de D-1 | Disponíveis após fechamento |
| Risk / Equity US | VIX, DXY, SP500, NASDAQ100 de D-1 | Disponíveis após fechamento |
| Commodities | Brent, Ouro, Minério de Ferro de D-1 | Disponíveis após fechamento |
| Macro BR | SELIC, IPCA, PIB | SELIC/IPCA: propagação de regime; PIB: anual |
| Eventos | COPOM, Focus, CPI EUA, Fed, Payroll, Vencimento WIN | Calendário público — disponível antes de D |

### 3.2 Volume de Dados e Implicações

| Parâmetro | Valor |
|---|---|
| Amostras por ano | ~252 pregões |
| Total disponível | 2.826 pregões (Jan/2015 – Mai/2026) |
| Região DEV | 2015–2023 → ~2.000 amostras — pesquisa, features, hiperparâmetros |
| Holdout DEV | 2024 (~251 amostras) — seleção de threshold e configuração final |
| Holdout final | 2025–2026 (~344 amostras) — avaliação única, **nunca tocar antes** |
| Tamanho mínimo de treino | ~500 amostras (~2 anos) por fold |
| Implicação para modelos | Regularização agressiva obrigatória — poucos dados para modelos complexos |

**Por que 2015?** Inclui ciclos distintos: crise BR, bull 2016–2019, COVID, juros altos 2022–2023. Sem esses regimes o modelo não generaliza.

### 3.3 Checklist de Integridade do Dataset (já verificado)

- ✅ Zero datas duplicadas no índice de pregões
- ✅ Zero datas de não-pregão (feriados B3, fins de semana removidos)
- ✅ OHLC consistente: High >= max(open,close), Low <= min(open,close) — 13 casos corrigidos via clip
- ✅ Volume > 0 em todos os dias de pregão (12 dias de baixo volume identificados)
- ✅ Zero NaN após tratamento completo — 2.826 × 75 colunas
- ✅ Retornos extremos validados: 6 ocorrências > 10% (todas em Mar/2020 — legítimas)
- ⚠️ PIB 2026 = PIB 2025 (dado anual ainda não divulgado) — atenção ao usar como feature
- ⚠️ Dados de Brent/EWZ/Ouro/NASDAQ Composite têm gap em Jan/2026 preenchido via ffill

---

## 4. Feature Engineering em Camadas

**Princípio:** Todas as features são calculadas sobre dados de D-1 ou anteriores. A ablação por camada é obrigatória: adicionar uma camada por vez e medir o delta de IC e AUC OOS antes de incluir a próxima.

### Camada 1 — Candle Diário D-1 (Preço Puro)

```python
# Estrutura do candle D-1
df['ret_close_d1']       = df['winfut_close'].pct_change()
df['body_pct_d1']        = (df['winfut_close'] - df['winfut_open']) / df['winfut_open']
df['range_pct_d1']       = (df['winfut_high'] - df['winfut_low']) / df['winfut_open']
df['close_position_d1']  = (df['winfut_close'] - df['winfut_low']) / (df['winfut_high'] - df['winfut_low'] + 1e-9)
df['upper_shadow_d1']    = (df['winfut_high'] - df[['winfut_open','winfut_close']].max(axis=1)) / (df['range_pct_d1'] * df['winfut_open'] + 1e-9)
df['lower_shadow_d1']    = (df[['winfut_open','winfut_close']].min(axis=1) - df['winfut_low']) / (df['range_pct_d1'] * df['winfut_open'] + 1e-9)

# Indicadores técnicos (usar ta-lib ou pandas_ta)
df['atr_14']      = ta.ATR(df['winfut_high'], df['winfut_low'], df['winfut_close'], timeperiod=14)
df['atr_ratio']   = df['range_pct_d1'] / (df['atr_14'] / df['winfut_close'])
df['vol_20d']     = df['ret_close_d1'].rolling(20).std()
df['rsi_14']      = ta.RSI(df['winfut_close'], timeperiod=14)
df['macd_hist']   = ta.MACD(df['winfut_close'])[2]
df['ema9_21']     = (ta.EMA(df['winfut_close'], 9) - ta.EMA(df['winfut_close'], 21)) / df['winfut_close']
df['bb_pct']      = (df['winfut_close'] - ta.BBANDS(df['winfut_close'])[2]) / (ta.BBANDS(df['winfut_close'])[0] - ta.BBANDS(df['winfut_close'])[2] + 1e-9)

# Memória de retornos recentes
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'ret_lag{lag}'] = df['ret_close_d1'].shift(lag - 1)

for w in [5, 10, 20]:
    df[f'ret_mean_{w}d'] = df['ret_close_d1'].rolling(w).mean()
    df[f'ret_std_{w}d']  = df['ret_close_d1'].rolling(w).std()
    df[f'ret_skew_{w}d'] = df['ret_close_d1'].rolling(w).skew()
    df[f'vol_ratio_{w}'] = df['vol_20d'] / df['ret_close_d1'].rolling(w).std()
```

### Camada 2 — Gap e Contexto de Abertura

Sem dados intraday, o gap overnight é ainda mais relevante como proxy de posicionamento noturno:

```python
df['gap_d1']          = (df['winfut_open'] - df['winfut_close'].shift(1)) / df['winfut_close'].shift(1)
df['gap_vs_atr']      = df['gap_d1'] / (df['atr_14'] / df['winfut_close'])
df['gap_extremo']     = (df['gap_vs_atr'].abs() > 1.5).astype(int)
df['gap_filled']      = (np.sign(df['gap_d1']) != np.sign(df['body_pct_d1'])).astype(int)
df['gap_positivo']    = (df['gap_d1'] > 0).astype(int)
df['gap_contra_spx']  = (np.sign(df['gap_d1']) != np.sign(df['sp500_change_pct'])).astype(int)
```

### Camada 3 — SPX e Risco Global (Driver Primário do WIN)

Esta é a camada mais importante para o WIN. Diferente do WDO, onde o DXY domina, o WIN é primariamente dirigido pelo apetite a risco global medido pelo SPX e VIX:

```python
# SPX — driver primário, com tratamento assimétrico (altas e quedas têm efeitos diferentes)
df['spx_ret_d1']      = df['sp500_change_pct']
df['spx_ret_pos']     = df['spx_ret_d1'].clip(lower=0)   # contribuição de alta
df['spx_ret_neg']     = df['spx_ret_d1'].clip(upper=0)   # contribuição de queda
df['spx_slope_5d']    = df['sp500_close'].pct_change(5)
df['spx_slope_20d']   = df['sp500_close'].pct_change(20)
df['spx_zscore']      = df['spx_ret_d1'] / df['spx_ret_d1'].rolling(20).std()
df['spx_drawdown_60'] = df['sp500_close'] / df['sp500_close'].rolling(60).max() - 1
df['spx_acima_ma50']  = (df['sp500_close'] > df['sp500_close'].rolling(50).mean()).astype(int)
df['spx_acima_ma200'] = (df['sp500_close'] > df['sp500_close'].rolling(200).mean()).astype(int)

# Nasdaq — útil quando tech lidera o movimento global
df['ndaq_ret_d1']   = df['nasdaq100_change_pct']
df['ndaq_vs_spx']   = df['ndaq_ret_d1'] - df['spx_ret_d1']  # divergência Nasdaq vs SPX

# VIX — nível e dinâmica
df['vix_nivel']     = df['vix_close']
df['vix_ret_d1']    = df['vix_change_pct']
df['vix_spike']     = (df['vix_ret_d1'] > 0.15).astype(int)
df['vix_regime']    = pd.cut(df['vix_close'], bins=[0,15,20,25,35,100],
                       labels=[0,1,2,3,4]).astype(int)  # 0=baixo, 4=crise
df['vix_acima_ma20']= (df['vix_close'] > df['vix_close'].rolling(20).mean()).astype(int)
```

### Camada 4 — Macro Brasil e Intermercado

| Ativo | Relevância para o WIN | Features derivadas |
|---|---|---|
| DI Futuro | Custo de oportunidade — Selic alta compete com renda variável | ret_d1 · slope_10d · nível absoluto (acima/abaixo de 10%) |
| DXY | Dólar global — fluxo estrangeiro: dólar forte = saída de emergentes | ret_d1 · slope_5d · regime (acima/abaixo da MA252) |
| USD/BRL | Real fraco = pressão vendedora no WIN via custo de hedge | ret_d1 · slope_5d · regime |
| US10Y / US2Y | Diferencial de juros — afeta custo de capital e valuation | nível · slope · flag curva invertida |
| Petróleo (Brent) | Petrobras ~10% do Ibovespa — petróleo alto = suporte ao índice | ret_d1 · slope_5d |
| Minério de Ferro | Vale ~10% do Ibovespa — minério é o principal driver da Vale | ret_d1 · slope_5d · regime |
| Ouro | Safe haven — correlação inversa com risco/renda variável | ret_d1 |
| EWZ | ETF de ações brasileiras — proxy de fluxo estrangeiro para Brasil | ret_d1 · vol_10d |

```python
externals = {
    'usd_brl': 'usd_brl_close', 'di1': 'di1_close',
    'us10y': 'us10y_close', 'us2y': 'us2y_close',
    'dxy': 'dxy_close', 'brent': 'petroleo_brent_close',
    'ouro': 'ouro_close', 'minerio': 'minerio_ferro_ultimo', 'ewz': 'ewz_close'
}
for name, col in externals.items():
    df[f'{name}_ret_d1']    = df[col].pct_change()
    df[f'{name}_slope_5d']  = df[col].pct_change(5)
    df[f'{name}_slope_20d'] = df[col].pct_change(20)
    df[f'{name}_vol_10d']   = df[col].pct_change().rolling(10).std()
    df[f'{name}_ma_regime'] = (df[col] > df[col].rolling(252).mean()).astype(int)

# Curva de juros EUA (spread 10Y-2Y): invertida = sinal de recessão
df['us_yield_spread']   = df['us10y_close'] - df['us2y_close']
df['curva_invertida']   = (df['us_yield_spread'] < 0).astype(int)

# Divergências WIN vs drivers (sinal de descorrelação ou pressão acumulada)
df['win_vs_spx']        = df['ret_close_d1'] - df['spx_ret_d1']
df['win_vs_ewz']        = df['ret_close_d1'] - df['ewz_ret_d1']
df['win_vs_commodities']= df['ret_close_d1'] - (0.5*df['brent_ret_d1'] + 0.5*df['minerio_ret_d1'])
```

### Camada 5 — Regime de Mercado (Crítica para o WIN)

Esta camada é mais importante para o WIN do que para o WDO, dada a magnitude dos regime shifts históricos:

```python
# Regime de volatilidade do WIN
df['regime_vol_win'] = pd.cut(df['vol_20d'],
    bins=df['vol_20d'].quantile([0, 0.33, 0.67, 1.0]).values,
    labels=['baixa','media','alta'])

# Regime de tendência do WIN (bull / bear / lateral)
ma50  = df['winfut_close'].rolling(50).mean()
ma200 = df['winfut_close'].rolling(200).mean()
df['regime_tendencia'] = np.where(df['winfut_close'] > ma50,
    np.where(ma50 > ma200, 2, 1),     # 2=bull forte, 1=bull fraco
    np.where(ma50 < ma200, -2, -1))   # -2=bear forte, -1=bear fraco

# Regime de juros Brasil (custo de oportunidade para renda variável)
df['regime_juros_altos'] = (df['di1_close'] > 10.0).astype(int)  # DI acima de 10%
df['regime_di_subindo']  = (df['di1_close'].pct_change(10) > 0).astype(int)

# Regime de risco global
df['regime_vix_alto']    = (df['vix_close'] > df['vix_close'].rolling(252).quantile(0.75)).astype(int)
df['regime_spx_bull']    = (df['sp500_close'] > df['sp500_close'].rolling(200).mean()).astype(int)
df['regime_dxy_forte']   = (df['dxy_close'] > df['dxy_close'].rolling(252).mean()).astype(int)

# Regime composto — ambiente favorável para WIN
df['regime_favoravel'] = (
    (df['regime_spx_bull'] == 1) &
    (df['regime_juros_altos'] == 0) &
    (df['regime_dxy_forte'] == 0)
).astype(int)
```

### Camada 6 — Calendário e Vencimento

```python
df['dia_da_semana']      = df['data'].dt.dayofweek   # 0=Seg, 4=Sex
df['mes']                = df['data'].dt.month
df['trimestre']          = df['data'].dt.quarter
df['semana_do_mes']      = (df['data'].dt.day - 1) // 7 + 1

# Vencimento do WIN (quarta-feira mais próxima do dia 15, meses pares)
df['dias_ate_venc']  = calcular_dias_ate_vencimento_win(df['data'])
df['venc_prox']      = (df['dias_ate_venc'] <= 3).astype(int)
df['semana_ciclo']   = (df['dias_ate_venc'] // 5).clip(0, 4)

# Eventos de calendário para o dia D (já disponíveis no dataset)
# evento_brasil_copom, evento_g6_eua_juros_fed, evento_g6_eua_payroll_nfp, etc.
# Atenção: esses são eventos DO DIA D — são known covariates (calendário público)
```

### 4.1 Seleção de Features — IC com Stratificação por Regime

Para o WIN, calcular o IC não apenas no dataset completo, mas também separado por regime:

```python
from scipy.stats import spearmanr

# IC global
ic_global = {}
for col in feature_candidates:
    mask = df[col].notna() & df['target'].notna()
    corr, _ = spearmanr(df.loc[mask, col], df.loc[mask, 'target'])
    ic_global[col] = abs(corr)

# IC por regime — identificar features regime-dependentes
ic_por_regime = {}
for regime in ['bull_forte', 'bull_fraco', 'bear_fraco', 'bear_forte']:
    df_r = df[df['regime_label'] == regime]
    ic_regime = {}
    for col in feature_candidates:
        mask = df_r[col].notna() & df_r['target'].notna()
        if mask.sum() < 30: continue
        corr, _ = spearmanr(df_r.loc[mask, col], df_r.loc[mask, 'target'])
        ic_regime[col] = abs(corr)
    ic_por_regime[regime] = ic_regime

# Features 'regime-estável': IC alto em ≥ 3 dos 4 regimes → mais robustas
# Features 'regime-específica': IC alto apenas em 1 regime → usar com cautela

# Limiar de seleção: IC médio >= 0.02
features_validas = [f for f, ic in ic_global.items() if ic >= 0.02]

# Remover multicolinearidade (r > 0.95)
corr_m  = df[features_validas].corr().abs()
upper   = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
features_final = [f for f in features_validas if f not in to_drop]
```

---

## 5. Baselines Triviais — Pré-Requisito Absoluto

> **Regra absoluta:** Nenhum modelo de ML é validado sem superar todos os baselines triviais em métricas financeiras. Se o XGBoost não bater os baselines em profit factor e Sharpe OOS, o problema está na definição do target ou na qualidade das features — não nos hiperparâmetros.

| ID | Definição | O que mede |
|---|---|---|
| Baseline A | Sempre prever a classe majoritária (alta na maioria dos períodos) | Piso trivial — qualquer modelo deve superar esse |
| Baseline B | Momentum: prever a mesma direção do dia anterior | Testa persistência de tendência diária |
| Baseline C | Mean reversion: prever direção oposta ao dia anterior | Testa se há reversão dominante dia a dia |
| Baseline D | Seguir o SPX: se SPX subiu ontem → prever alta do WIN | Valida o poder preditivo isolado do driver primário |
| Baseline E | Regra de VIX: se VIX subiu > 10% ontem → prever baixa; senão → prever alta | Valida o poder preditivo isolado do risco global |
| Baseline F | Buy and hold: sempre long (sem previsão) | Benchmark de retorno absoluto — único para o WIN (ativo com viés de alta secular) |

> **Baseline F é específico para o WIN:** para um ativo com viés de alta secular, o buy and hold é um benchmark real, não apenas trivial. O modelo precisa ter Sharpe OOS superior ao buy and hold ajustado por drawdown.

Todos os baselines são avaliados com as mesmas métricas financeiras e nas mesmas janelas de walk-forward que os modelos principais.

---

## 6. Protocolo de Validação — Walk-Forward Purged e Regime-Aware

### 6.1 Arquitetura de Splits

| Parâmetro | Definição |
|---|---|
| Período total | 2015–2026 (~2.826 amostras) |
| Região DEV | 2015–2023 (~2.000 amostras) — pesquisa, features, hiperparâmetros |
| Holdout DEV | 2024 (~251 amostras) — seleção de threshold e configuração final |
| Holdout final | 2025–Mai/2026 (~344 amostras) — avaliação única, **nunca tocar antes** |
| Folds na região DEV | 5 folds mínimo — expanding window |
| Tamanho mínimo de treino | ~500 amostras (~2 anos) |
| Tamanho do teste por fold | ~252 amostras (~1 ano) |
| Purge | 5 dias entre treino e teste |
| Embargo | 3 dias após cada fold |
| Requisito de regime | Cada fold de teste deve ter ao menos 1 regime distinto representado |

### 6.2 Análise de Regime nos Folds — Obrigatória para o WIN

Diferente do WDO, os folds do WIN podem ter composição de regime muito diferente entre si:

```python
def analisar_regime_fold(df_fold):
    """Classificar e reportar composição de regime de um fold"""
    df_fold = df_fold.copy()
    
    # Classificar regime por ano
    annual_ret = df_fold.groupby(df_fold['data'].dt.year)['winfut_close'].apply(
        lambda x: x.iloc[-1] / x.iloc[0] - 1)
    
    regime_counts = {'bull_forte':0, 'bull_fraco':0, 'bear_fraco':0, 'bear_forte':0}
    for ret in annual_ret:
        if ret > 0.15:     regime_counts['bull_forte'] += 1
        elif ret > 0:      regime_counts['bull_fraco'] += 1
        elif ret > -0.15:  regime_counts['bear_fraco'] += 1
        else:              regime_counts['bear_forte'] += 1
    
    pct_alta = df_fold['target_dir'].mean()
    vol_media = df_fold['ret'].std() * np.sqrt(252)
    
    return {'regimes': regime_counts, 'pct_alta': pct_alta, 'vol_anual': vol_media}

# Para cada fold: reportar composição + balanceamento de classes
for i, (train_idx, test_idx) in enumerate(folds):
    info_treino = analisar_regime_fold(df.iloc[train_idx])
    info_teste  = analisar_regime_fold(df.iloc[test_idx])
    print(f'Fold {i+1}: treino={info_treino["regimes"]} | teste={info_teste["regimes"]}')
```

### 6.3 Walk-Forward com Purge e Embargo

```python
class WalkForwardPurgedWIN:
    """
    purge_bars:   candles removidos antes do teste
                  evita features com lookback longo (ex: vol_20d) vazar do treino pro teste
    embargo_bars: candles removidos após cada fold
                  evita seleção de threshold contaminada pelo fold seguinte
    """
    def __init__(self, n_splits=5, test_size=252, purge=5, embargo=3,
                 min_train_size=500, window_type='expanding'):
        self.n_splits      = n_splits
        self.test_size     = test_size
        self.purge         = purge
        self.embargo       = embargo
        self.min_train_size= min_train_size
        self.window_type   = window_type
    
    def split(self, X):
        n = len(X)
        folds = []
        for i in range(self.n_splits):
            test_end   = n - (self.n_splits - i - 1) * self.test_size
            test_start = test_end - self.test_size
            if self.window_type == 'expanding':
                train_end = test_start - self.purge
                train_start = 0
            else:  # rolling
                train_end   = test_start - self.purge
                train_start = max(0, train_end - self.min_train_size * 3)
            if train_end - train_start >= self.min_train_size:
                folds.append((
                    list(range(train_start, train_end)),
                    list(range(test_start, min(test_end, n - self.embargo)))
                ))
        return folds
```

---

## 7. Os Três Modelos

### Modelo 1 — Regressão Logística com Features Clássicas
**Nível: INICIANTE · Prazo estimado: 1–2 semanas**

#### Objetivo
Classificação direcional (UP/DOWN) com threshold calibrado. Serve como baseline supervisionado e estabelece o piso de performance que os modelos avançados devem superar.

#### Arquitetura
```
Features (Camadas 1-3) → StandardScaler → LogisticRegression(C tunado, class_weight='balanced')
→ CalibratedClassifierCV (isotonic) → P(alta) / P(baixa) → threshold → sinal
```

#### Features prioritárias
- Retorno WIN D-1, D-2, D-3, D-5 (memória de curto prazo)
- Retorno SPX D-1 + versões positiva/negativa (assimetria)
- VIX nível D-1 + VIX spike flag
- Retorno USD/BRL D-1 (canal cambial)
- DI1 change D-1 + flag regime juros altos
- RSI(14), MACD histograma, BB%
- Eventos binários: `evento_brasil_copom`, `evento_g6_eua_juros_fed`
- `dia_da_semana`, `venc_prox`

#### Target
```python
# ATENÇÃO: shift(-1) = look-ahead! Treinar apenas com dados D já fechados
df['target'] = (df['winfut_close'].shift(-1) > df['winfut_close']).astype(int)
# Usando target com limite adaptativo (seção 2.1) é preferível
```

#### Validação e threshold
```python
# Threshold selecionado no Holdout DEV (2024), nunca no teste final
thresholds = np.arange(0.50, 0.75, 0.01)
melhor_threshold, melhor_sharpe = None, -np.inf
for t in thresholds:
    trades = aplicar_threshold(proba_val, t)
    if len(trades) >= 30:  # mínimo estatístico
        sh = calcular_sharpe(trades)
        if sh > melhor_sharpe:
            melhor_sharpe, melhor_threshold = sh, t
```

#### Pontos fortes
- Interpretável: coeficientes mostram importância das features e sinal de cada uma
- Rápido de treinar e auditar — ideal para detectar look-ahead bias rapidamente
- Baixo risco de overfitting com regularização L2
- Calibração de probabilidade nativa

#### Critério de avanço para Modelo 2
Modelo 1 deve atingir em OOS (holdout 2024): balanced_accuracy > 53% E profit factor > 1.1

---

### Modelo 2 — HMM + XGBoost Regime-Conditioned
**Nível: INTERMEDIÁRIO · Prazo estimado: 2–3 semanas**

> **Pré-requisito:** Modelo 1 atingiu balanced_accuracy > 53% e profit factor > 1.1 no Holdout DEV.

#### Objetivo
Detectar regimes de mercado via HMM e treinar XGBoost condicionado ao regime. Captura não-linearidades que a regressão logística não consegue, especialmente a mudança de dinâmica entre bull markets, bear markets e períodos de juros altos.

#### Fase 1 — HMM para Regimes

```python
from hmmlearn.hmm import GaussianHMM

# Features para HMM — séries que descrevem o "estado" do mercado
hmm_features = np.column_stack([
    df['ret_close_d1'],          # retorno WIN
    df['vol_20d'],               # volatilidade recente
    df['sp500_change_pct'],      # humor global
    df['vix_close'],             # risco global
    df['di1_change_pct'],        # movim. juros BR
])

# Testar 2 e 3 estados — validar com AIC/BIC e interpretabilidade
for n_states in [2, 3]:
    hmm = GaussianHMM(n_components=n_states, covariance_type='full',
                      n_iter=200, random_state=42)
    hmm.fit(hmm_features)
    print(f'{n_states} estados — AIC: {hmm.aic(hmm_features):.0f}')
```

**Estados típicos esperados para o WIN (3 estados):**
- **Estado 0 — Bull/Risk-On:** retorno positivo, baixa vol, VIX baixo, SPX subindo
- **Estado 1 — Bear/Risk-Off:** retorno negativo, alta vol, VIX alto
- **Estado 2 — Lateral juros altos:** retorno neutro, vol moderada, DI elevado

> **Regra crítica de leakage:** O HMM é treinado na região DEV. No walk-forward, usar *predict* (não *fit*) nos dados de teste. O estado do dia D-1 é predito apenas com dados até D-1 (forward-pass do Viterbi).

#### Fase 2 — XGBoost por Regime

```python
from xgboost import XGBClassifier
import optuna

# Features completas (Camadas 1-6) + label de regime do HMM
features_xgb = features_camada_1_a_6 + ['regime_hmm']

# Treinar um modelo por regime (ou usar regime como feature + interação)
modelos_por_regime = {}
for regime_id in range(n_estados_hmm):
    df_regime = df_treino[df_treino['regime_hmm'] == regime_id]
    if len(df_regime) < 200:
        print(f'Regime {regime_id}: poucas amostras ({len(df_regime)}) — usar modelo global')
        modelos_por_regime[regime_id] = modelo_global
        continue
    
    # Hyperparameter search com Optuna
    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'max_depth':        trial.suggest_int('max_depth', 3, 7),
            'learning_rate':    trial.suggest_float('lr', 0.01, 0.2, log=True),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
            'reg_lambda':       trial.suggest_float('lambda', 1.0, 10.0),
            'scale_pos_weight': n_baixa_regime / n_alta_regime,
        }
        model = XGBClassifier(**params, eval_metric='auc', use_label_encoder=False)
        # Cross-validation interna no treino do regime
        scores = cross_val_score(model, df_regime[features_xgb], df_regime['target'],
                                  cv=3, scoring='balanced_accuracy')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best = XGBClassifier(**study.best_params)
    best.fit(df_regime[features_xgb], df_regime['target'])
    modelos_por_regime[regime_id] = best
```

#### Diagnósticos pós-treino obrigatórios
- Feature importance por regime via SHAP — top-10 features devem se repetir em ≥ 4 dos 5 folds
- Calibration plot: probabilidades devem ser bem calibradas na faixa [0.4, 0.6]
- Matriz de confusão por regime
- Análise de erros: contexto dos falsos positivos e falsos negativos mais custosos

#### Critério de avanço para Modelo 3
Modelo 2 deve atingir em OOS (holdout 2024): Sharpe OOS > 1.0 E profit factor > 1.3 em ≥ 3 dos 5 folds, com feature importance estável.

---

### Modelo 3 — Temporal Fusion Transformer (TFT)
**Nível: AVANÇADO · Prazo estimado: 3–4 semanas**

> **Pré-requisito:** Modelo 2 atingiu Sharpe OOS > 1.0 e profit factor > 1.3. Se o XGBoost não encontra sinal, o TFT também não vai encontrar — e é muito mais difícil de debugar.

#### Objetivo
Previsão probabilística do retorno do WIN para os próximos 1–5 dias com intervalos de confiança. O TFT é justificado aqui pela existência de **known covariates** reais no dataset (eventos de calendário público).

#### Arquitetura
```
Temporal Fusion Transformer (TFT) — Lim et al., NeurIPS 2021
├── Encoder: janela de 60 dias históricos
│   ├── Variáveis contínuas: OHLCV, Rates, FX, Macro
│   └── Variáveis categóricas: Regime HMM, Dia semana, Mês
├── Variable Selection Networks (VSN) — seleciona features relevantes por timestep
├── Gated Residual Networks (GRN) — residual connections com gating
├── Multi-head Self-Attention (temporal) — capta dependências de longa janela
└── Decoder: previsão H=1..5 dias com quantis (10%, 50%, 90%)
```

#### Tipos de variáveis para o TFT

**Variáveis conhecidas no futuro (known covariates — vantagem real do dataset):**
```python
known_future = [
    'dia_da_semana', 'mes', 'trimestre',
    'evento_brasil_copom',          # data pública do COPOM
    'evento_brasil_vencimento_win', # calendário B3 público
    'evento_g6_eua_juros_fed',      # calendar Fed público
    'evento_g6_eua_payroll_nfp',    # primeira sexta do mês — público
    'dias_ate_venc',                # calculado do calendário
]
```

**Variáveis desconhecidas no futuro (observed — só disponíveis em D-1):**
```python
observed = [
    'ret_close_d1', 'vol_20d', 'sp500_change_pct',
    'vix_close', 'usd_brl_change_pct', 'di1_close',
    'us10y_close', 'dxy_close', # ... todos os retornos
]
```

**Variáveis estáticas (contexto de regime):**
```python
static = ['selic_meta_ano_pct', 'regime_juros_altos']
```

#### Target e estratégia de sinal
```python
# Previsão probabilística de retorno (regressão com quantis)
df['target_ret'] = df['winfut_close'].pct_change().shift(-1)

# Estratégia de entrada baseada em quantis
# Se P10 > 0: comprar (cenário pessimista ainda positivo → confiança alta)
# Se P90 < 0: vender (cenário otimista ainda negativo → confiança alta)
# Caso contrário: flat

sinal = np.where(proba_p10 > 0, 1,    # long com alta confiança
         np.where(proba_p90 < 0, -1,  # short com alta confiança
         0))                           # flat — incerteza alta
```

#### Implementação recomendada

```python
# Opção recomendada: pytorch-forecasting (menos código, mais estável)
pip install pytorch-forecasting pytorch-lightning

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Configuração para o WIN
dataset = TimeSeriesDataSet(
    df_treino,
    time_idx='time_idx',
    target='target_ret',
    group_ids=['serie_id'],          # único grupo (WIN)
    max_encoder_length=60,           # 60 dias de contexto
    max_prediction_length=5,         # prever até 5 dias à frente
    time_varying_known_reals=known_future,
    time_varying_unknown_reals=observed,
    static_reals=static,
    target_normalizer=None,
)

tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.2,
    hidden_continuous_size=32,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
)
```

> **Infraestrutura necessária:** GPU recomendado, ~2–4h de treino por fold. CPU é viável mas lento (~8–12h por fold).

---

## 8. Métricas de Avaliação

### 8.1 Métricas Financeiras Operacionais (Primárias)

```python
def calcular_metricas_financeiras(ret_operacoes, len_total_candles, freq_ano=252):
    """ret_operacoes: série de retornos líquidos por candle operado (já com custo)"""
    n_ops      = len(ret_operacoes)
    win_rate   = (ret_operacoes > 0).mean()
    pf         = ret_operacoes[ret_operacoes > 0].sum() / abs(ret_operacoes[ret_operacoes < 0].sum())
    sharpe     = ret_operacoes.mean() / ret_operacoes.std() * np.sqrt(freq_ano)
    sortino    = ret_operacoes.mean() / ret_operacoes[ret_operacoes < 0].std() * np.sqrt(freq_ano)
    cumret     = (1 + ret_operacoes).cumprod()
    max_dd     = (cumret / cumret.cummax() - 1).min()
    calmar     = ret_operacoes.mean() * freq_ano / abs(max_dd)
    expectancy = (win_rate * ret_operacoes[ret_operacoes > 0].mean() +
                  (1 - win_rate) * ret_operacoes[ret_operacoes < 0].mean())
    cobertura  = n_ops / len_total_candles
    return dict(n_ops=n_ops, win_rate=win_rate, profit_factor=pf,
                sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
                calmar=calmar, expectancy=expectancy, cobertura=cobertura)
```

### 8.2 Critérios Mínimos de Aceitação (OOS)

| Métrica | Mínimo promissor | Rejeitar se |
|---|---|---|
| Profit factor | > 1.3 | < 1.1 |
| Sharpe anualizado | > 0.8 | < 0.5 |
| Win rate | > 48% | < 43% (dado custo) |
| Max drawdown | < 20% | > 30% |
| N operações (por fold) | > 200 | < 100 — resultado não confiável |
| Cobertura de candles | > 15% | < 8% — estratégia muito seletiva |

### 8.3 Métricas de ML (Secundárias — para diagnóstico)

| Métrica | Notas |
|---|---|
| Balanced accuracy | Principal métrica de ML — imune ao viés de alta do WIN |
| AUC-ROC | Diagnóstico de discriminação |
| F1 macro | Balanceia precision e recall entre as classes |
| Brier score | Calibração de probabilidade |
| ECE (Expected Calibration Error) | Qualidade das probabilidades — crítico para threshold |

> **Regra de ouro:** Se o modelo não gera ao menos 30 operações por fold de teste, os resultados não são estatisticamente válidos. Ajustar o threshold ou revisar a definição do target.

### 8.4 Análise de Performance por Regime (Obrigatória para o WIN)

```python
for regime in ['bull_forte', 'bull_fraco', 'bear_fraco', 'bear_forte', 'covid', 'juros_altos']:
    df_r = df_oos[df_oos['regime_label'] == regime]
    if len(df_r) < 30: continue
    metricas_r = calcular_metricas_financeiras(df_r['ret_operacao'], len(df_r))
    print(f'Regime {regime}: Sharpe={metricas_r["sharpe"]:.2f} | PF={metricas_r["profit_factor"]:.2f}')
```

### 8.5 Custos de Transação

```python
custo_ida_volta_win = {
    'corretagem':         1.20,   # R$ por contrato (típico)
    'emolumentos_b3':     0.50,   # aproximado
    'slippage_estimado':  2.00,   # 2 pontos = R$0.20 por contrato (conservador)
    'total_r_contrato':   1.70,   # R$ por contrato por operação
}
# Convertido para retorno sobre nocional médio WIN (~R$ 1.800 por contrato × 1 ponto)
# 1 ponto do WIN = R$0.20; com WIN a 180k: R$ 36.000 nocional
# custo_retorno ≈ 1.70 / 36000 ≈ 0.0047% por trade — SEMPRE incluir nas simulações
```

---

## 9. Roadmap de Implementação

```
FASE 1 — Engenharia de Features (1–2 semanas)
├── Calcular retornos lagados (1-5 dias) para WIN e variáveis externas
├── Indicadores técnicos: RSI, MACD, Bollinger, ATR, EMA cross
├── Features de assimetria SPX (clip pos/neg, zscore, drawdown)
├── Features de regime: vol_regime, tendencia, juros, spx_bull
├── Parsing das colunas de evento texto → flags binários adicionais
├── Spread US10Y-US2Y (curva de juros)
└── AUDITORIA DE LOOK-AHEAD: toda feature em D usa apenas dados até D-1

FASE 2 — Baselines + IC (1 semana)
├── Implementar Baselines A, B, C, D, E, F com walk-forward idêntico
├── Calcular IC por feature — global e por regime
├── Remover features com IC < 0.02 ou correlação entre si > 0.95
└── Documentar piso de performance (melhor baseline por fold)

FASE 3 — Modelo 1: Logistic Regression (1 semana)
├── LogisticRegression com StandardScaler e regularização L2
├── CalibratedClassifierCV para calibrar probabilidades
├── Walk-forward purged (5 folds, região DEV 2015-2023)
├── Threshold selecionado no Holdout DEV (2024)
└── Critério de avanço: balanced_acc > 53% e PF > 1.1 no Holdout DEV

FASE 4 — Modelo 2: HMM + XGBoost (2–3 semanas)
├── HMM (2 e 3 estados) com validação de interpretabilidade
├── XGBoost por regime com Optuna (50 trials por regime)
├── SHAP values para feature importance por regime
├── Walk-forward: HMM treinado no treino, XGBoost treinado no treino
└── Critério de avanço: Sharpe > 1.0 e PF > 1.3 em ≥ 3 dos 5 folds

FASE 5 — Modelo 3: TFT (3–4 semanas)
├── Implementar via pytorch-forecasting
├── Definir known/unknown/static covariates
├── Walk-forward anual com retrain trimestral
├── Estratégia de sinal baseada em quantis P10/P50/P90
└── Comparar com Modelo 2 em OOS comum — avançar apenas se ganho mensurável

FASE 6 — Ensemble (1–2 semanas)
├── Ensemble ponderado Modelo 2 + Modelo 3 (se TFT trouxer ganho)
├── Sistema de confiança: operar apenas quando modelos concordam (≥ 0.60)
└── Análise de degradação temporal: monitorar drift de performance OOS

FASE 7 — Paper Trading + Monitoramento
├── Executar em simulação por 60 pregões antes de capital real
├── Dashboard: regime atual, sinal diário, performance acumulada
└── Critério de parada de sistema: drawdown > 15% acumulado
```

---

## 10. Riscos Técnicos Críticos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Look-ahead bias | Alta | Crítico | Audit obrigatório de todo `shift()` e rolling |
| Overfitting temporal | Alta | Alto | Walk-forward estrito, sem shuffle dos dados |
| Acurácia enganosa (viés de alta) | Alta | Alto | Usar balanced_accuracy; avaliar por regime |
| Degradação de regime | Média | Alto | Monitorar drift OOS; retrain periódico |
| Custos subestimados | Alta | Médio | Simular: spread 2pts + R$1,20/contrato sempre |
| Data snooping | Alta | Alto | Manter holdout 2025-2026 intocado até fase final |
| HMM instável | Média | Médio | Rodar com 10 seeds diferentes; validar estados |
| TFT sem ganho sobre XGBoost | Média | Baixo | Critério de entrada claro — só avançar com evidência |

---

## 11. Limitações do Dataset e Adaptações

| Limitação | Impacto | Adaptação |
|---|---|---|
| Granularidade diária (sem intraday) | Sem VWAP, microestrutura, candles 15min | Foco em drivers macro e regime; gap overnight é proxy |
| Sem dados de book/fluxo | Sem imbalance, agressores | EWZ como proxy de fluxo estrangeiro; volume relativo |
| Sem dados de opções | Sem IV, Skew, Put/Call ratio WIN | VIX como proxy de volatilidade implícita global |
| PIB 2026 = PIB 2025 | Feature imprecisa para 2026 | Usar com cautela; PIB é feature de longo prazo, baixo peso |
| Brent/EWZ/Ouro com gap Jan/2026 | ~85 dias preenchidos via ffill | Aumentar atenção nas métricas de 2026; verificar impacto |
| IPCA/SELIC: risco de look-ahead | Se usar dado divulgado após D | Garantir que feature usa valor *disponível no momento D-1* |

---

*Plano Mestre WIN v2.0 — 20/05/2026*  
*Base: 2.826 pregões D1 · 2015–2026 · 75 colunas · ZERO NaN*

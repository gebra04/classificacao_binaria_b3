# 🧠 PLANO MESTRE — Modelo Preditivo WIN (Mini-Índice Bovespa)
**Versão:** 2.0 — atualizada com dataset real  
**Dataset:** `base_concatenada_v1.csv` — diário, 2015-01-02 a 2026-05-20  
**Granularidade:** Diária (D1) — não foi possível obter 15 min

---

## 1. DIAGNÓSTICO DO DATASET

### 1.1 Estrutura Geral

| Atributo | Valor |
|---|---|
| Linhas (raw) | 4.158 |
| Linhas após limpeza | **2.826 pregões** |
| Colunas (raw) | 83 |
| Colunas após limpeza | **75** |
| Período | Jan/2015 — Mai/2026 (~11,2 anos) |
| Granularidade | Diária (D1) |

### 1.2 Grupos de Variáveis

| Grupo | Colunas | Exemplos |
|---|---|---|
| WIN OHLCV | 6 | open, high, low, close, volume, change_pct |
| FX — USD/BRL | 5 | OHLC + change_pct |
| DI1 (Juros BR) | 6 | OHLC + volume + change_pct |
| US Rates | 9 | US2Y e US10Y — close, open, high, low |
| Risk | 10 | VIX (OHLC) + DXY (OHLC) |
| Equity US | 13 | SP500, NASDAQ100, NASDAQ Composite, EWZ |
| Commodities | 7 | Brent, Ouro, Minério de Ferro |
| Macro BR | 6 | SELIC, IPCA, PIB |
| Eventos | 15 | COPOM, Focus, CPI EUA, Fed, Payroll, Vencimento WIN |

### 1.3 Estatísticas do WIN

| Métrica | Valor |
|---|---|
| Retorno médio diário | +0.058% |
| Volatilidade diária | 1.52% |
| Volatilidade anualizada | **24.1%** |
| Sharpe bruto (sem custo) | 0.60 |
| Pior ano | 2015 (-12.6%) |
| Melhor ano | 2016 (+43.3%) |
| Extremos de retorno diário | -15% (Mar/2020) / +14.6% (Mar/2020) |

---

## 2. ANÁLISE DE GAPS — DIAGNÓSTICO

### 2.1 Tipos de gaps identificados

| Tipo | Volume | Causa | Tratamento |
|---|---|---|---|
| **Fins de semana** | 1.188 linhas | Estrutural do calendário | Removidas |
| **Feriados BR** | 144 linhas | Mercado fechado | Removidas (WIN = NaN) |
| **Feriados US** | 8–15 linhas/var | NYSE/CME fechado | ffill limitado |
| **Variáveis 100% NaN** | 5 colunas | Dados nunca coletados | Dropadas |
| **Variáveis mensais** | IPCA | Divulgação mensal | ffill total (propagação de regime) |
| **Variáveis anuais** | PIB | Dado anual | Mapeamento por ano + ffill |
| **Variáveis de regime** | SELIC | Política monetária contínua | ffill total |
| **OHLC inconsistentes** | 13 linhas | Close fora do H-L | Clip para H-L |

### 2.2 Correlações Relevantes com WIN Close

| Variável | Correlação | Interpretação |
|---|---|---|
| NASDAQ Composite | +0.928 | Fortíssima correlação estrutural |
| SP500 | +0.927 | Idem |
| NASDAQ100 | +0.922 | Idem |
| Ouro | +0.853 | Correlação de longo prazo |
| EWZ | +0.777 | ETF Brasil — proxy direto |
| USD/BRL | +0.763 | Câmbio embute risco Brasil |
| Minério de Ferro | +0.646 | Commodities BR |

> ⚠️ **Atenção**: correlação de nível (preço absoluto) é espúria para séries não-estacionárias. Para modelos preditivos, usar **retornos** ou **variações**, não preços absolutos.

---

## 3. TRATAMENTOS APLICADOS

### 3.1 Remoção
- Fins de semana → removidos (não são pregões)
- Feriados brasileiros → removidos (WIN sem dados)
- Colunas 100% NaN: `us2y_volume`, `us10y_volume`, `vix_volume`, `sp500_volume`, `usd_brl_volume`, `dxy_volume`, `minerio_ferro_volume`

### 3.2 Forward Fill com limite
- **Limit=1**: USD/BRL, DI1, US Rates (US2Y, US10Y), VIX, DXY, SP500, NASDAQ100, Commodities (Brent, Ouro, EWZ, NASDAQ Composite, Minério)
  - _Raciocínio_: feriados isolados americanos; usar último valor disponível por no máximo 1 dia

### 3.3 Forward Fill sem limite
- **SELIC**: dado de regime de política monetária — válido até próxima decisão COPOM
- **IPCA**: dado mensal — válido até próxima divulgação
- **PIB**: mapeado por ano (dado estrutural anual), com ffill para 2026

### 3.4 Recálculo
- **change_pct**: todas as variações percentuais recalculadas a partir dos closes (evita inconsistências do dado original)
- **minerio_ferro_variacao_pct**: recalculado via `pct_change()`

### 3.5 Correção de OHLC
- `winfut_close` clipado para `[winfut_low, winfut_high]` nos 13 casos inconsistentes

### 3.6 Eventos
- NaN em colunas de evento → preenchidos com `0` (sem evento = 0)
- Colunas de texto (`eventos_brasil`, etc.) → `"nenhum"` quando NaN

### ✅ Resultado: Dataset 100% limpo — 2.826 linhas × 75 colunas, ZERO NaN

---

## 4. LIMITAÇÕES IMPORTANTES DO DATASET

> **Estas limitações definem diretamente o que é e não é possível modelar:**

1. **Granularidade diária, não intraday** — impossível capturar padrões de abertura, horários de maior volatilidade, reversões intraday. Toda lógica deve ser posicional (fim do dia → decisão para o dia seguinte).

2. **Sem dados de book/fluxo** — sem informação de agressores, imbalance, nem fluxo institucional vs. PF.

3. **Sem dados de opções** — sem Skew, IV, Put/Call ratio do WIN.

4. **Eventos de texto não estruturados** — as colunas `eventos_brasil` contêm strings concatenadas (ex: `"CPI_EUA;Juros_Fed"`). Precisam de parsing adicional para uso em ML.

5. **Dados 2026 incompletos** — Brent, EWZ, Ouro, NASDAQ Composite têm gap de ~85 dias em 2026 (preenchido via ffill — atenção ao uso em validação).

6. **PIB 2026 = PIB 2025** — dado estrutural, baixa utilidade como feature preditiva de curto prazo.

7. **Risco de look-ahead em IPCA/SELIC** — ao usar esses dados como features, garantir que está usando o valor *disponível no momento da decisão*, não o valor divulgado depois.

---

## 5. PLANO MESTRE — 3 MODELOS SUGERIDOS

---

### 🟢 MODELO 1 — Regressão Logística com Features Clássicas
**Nível de dificuldade: INICIANTE**

#### Objetivo
Classificar a direção do WIN no pregão seguinte: **UP (+1) ou DOWN (0)**

#### Arquitetura
```
Features → StandardScaler → LogisticRegression (C tunado) → Sinal direcional
```

#### Features sugeridas
- Retorno WIN D-1, D-2, D-3
- Retorno SP500 D-1 (proxy de humor global)
- Retorno USD/BRL D-1 (risco cambial)
- VIX close D-1 (volatilidade implícita)
- DI1 change D-1 (taxa juros BR)
- SELIC atual (nível de juros)
- Indicadores técnicos simples: RSI(14), SMA(5)/SMA(20) ratio, volume ratio
- Eventos binários: `evento_brasil_copom`, `evento_g6_eua_juros_fed`

#### Validação
- Walk-forward com janelas de 252 dias de treino / 63 dias de teste
- 8 splits mínimos para cobrir 2+ anos de OOS

#### Target
```python
df['target'] = (df['winfut_close'].shift(-1) > df['winfut_close']).astype(int)
# ATENÇÃO: shift(-1) = look-ahead! Treinar apenas com dados D já fechados
```

#### Métricas esperadas
- Acurácia OOS: 52-56% (mercado eficiente; >55% é muito bom)
- Priorizar Precision em UP (evitar falsos positivos)

#### Pontos fortes
- Interpretável: coeficientes mostram importância das features
- Rápido de treinar e auditar
- Baixo risco de overfitting com regularização L2
- Baseline excelente para comparar modelos avançados

#### Riscos
- Não captura não-linearidades (regimes de mercado)
- Pode degradar em períodos de mudança estrutural

---

### 🟡 MODELO 2 — HMM + XGBoost (Regime-Conditioned)
**Nível de dificuldade: INTERMEDIÁRIO**

#### Objetivo
Detectar regimes de mercado via HMM, e treinar um classificador separado por regime via XGBoost para sinal de direção e magnitude.

#### Arquitetura
```
Retornos + Volatilidade → HMM (2-3 estados) → Label de Regime
Features Técnicas + Macro → XGBoost (por regime) → P(UP | regime)
```

#### Fase 1 — HMM para regimes
```python
from hmmlearn.hmm import GaussianHMM

# Features para HMM:
hmm_features = ['ret_win', 'vol_21d', 'ret_sp500', 'vix_close']
# 3 estados: Bull / Bear / Lateral-Alta-Vol
model_hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=100)
```

Estados típicos identificados em séries como o WIN:
- **Estado 0 — Bull Trend**: retorno positivo, baixa vol, VIX baixo
- **Estado 1 — Bear/Crise**: retorno negativo, alta vol, VIX alto
- **Estado 2 — Lateral volátil**: retorno neutro, vol moderada

#### Fase 2 — XGBoost por regime
```python
# Para cada regime, treinar um XGBoost separado
# Features:
xgb_features = [
    # Retornos lagados (1-5 dias)
    'ret_win_d1', 'ret_win_d2', 'ret_win_d3', 'ret_win_d5',
    # Mercado externo
    'ret_sp500_d1', 'ret_nasdaq_d1', 'ret_usd_brl_d1',
    # Volatilidade
    'vol_5d', 'vol_21d', 'vol_ratio_5_21',
    # Técnico
    'rsi_14', 'macd_signal', 'bb_position',
    # Macro
    'selic_meta_ano_pct', 'ipca_acum_12m',
    # US
    'us10y_close', 'vix_close', 'dxy_close',
    # Sazonalidade
    'day_of_week', 'month',
    # Eventos
    'evento_brasil_copom', 'evento_g6_eua_juros_fed',
    'evento_brasil_vencimento_win'
]
```

#### Validação
- Walk-forward nested: HMM treinado no treino, XGBoost treinado no treino, testado no hold-out
- CRITICAL: HMM **não pode ser re-treinado** com dados futuros — usar predição online

#### Métricas adicionais
- Sharpe ratio da estratégia simulada
- Confusion matrix por regime
- Feature importance por regime

#### Pontos fortes
- Captura mudanças estruturais de mercado
- XGBoost é robusto a features heterogêneas
- Interpretável via SHAP
- Historicamente, regime-conditioning melhora substancialmente a acurácia direcional

#### Riscos
- HMM é sensível à inicialização — rodar múltiplas seeds
- Risco de overfitting no XGBoost sem regularização adequada
- Transição de regime no momento de decisão é incerta

---

### 🔴 MODELO 3 — Transformer Temporal (TFT) com Atenção Multi-Variada
**Nível de dificuldade: AVANÇADO**

#### Objetivo
Previsão probabilística do retorno do WIN para os próximos 1-5 dias, com intervalos de confiança e decomposição de atenção.

#### Arquitetura
```
Temporal Fusion Transformer (TFT) — Lim et al., 2021
├── Encoder: janela de 60 dias históricos
│   ├── Variáveis contínuas: OHLCV, Rates, FX, Macro
│   └── Variáveis categóricas: Regime HMM, Dia da semana, Mês
├── Variable Selection Networks (VSN)
├── Gated Residual Networks (GRN)
├── Multi-head Attention (temporal)
└── Decoder: previsão H=1..5 dias com quantis (10%, 50%, 90%)
```

#### Stack tecnológico
```python
# Opção 1: PyTorch Forecasting (mais fácil)
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# Opção 2: Implementação manual (mais controle)
import torch
import torch.nn as nn
```

#### Features especiais para o TFT

**Variáveis conhecidas no futuro (known covariates):**
- Dia da semana, mês, trimestre
- Flags de evento: `evento_brasil_copom` (data é conhecida antecipadamente)
- `evento_brasil_vencimento_win` (calendário B3 público)
- `evento_g6_eua_juros_fed` (calendar Fed público)

**Variáveis desconhecidas no futuro (unknown):**
- Todos os retornos e preços
- VIX, DXY, USD/BRL
- DI1

**Variáveis estáticas:**
- SELIC (muda raramente, contexto de regime)

#### Target
```python
# Previsão de retorno no próximo pregão (regressão)
df['target_ret'] = df['winfut_close'].pct_change().shift(-1)

# OU: previsão de quantis para sizing de posição
# P10, P50, P90 do retorno amanhã
```

#### Validação
- Walk-forward com expansão de janela
- Retrain a cada 63 dias (trimestral)
- Métricas: RMSE, MAE, Quantile Loss, Winkler Score (intervalos)

#### Estratégia de uso do sinal
```
Se P10 > 0:     comprar (cenário pessimista ainda positivo)
Se P90 < 0:     vender (cenário otimista ainda negativo)
Caso contrário: flat ou posição reduzida
```

#### Pontos fortes
- Captura dependências de longa janela temporal
- Atenção revela quais variáveis foram relevantes em cada regime
- Saída probabilística → permite gestão de risco baseada em incerteza
- Known covariates (eventos futuros) são uma vantagem real

#### Riscos
- Requer ~2000+ amostras para treinar bem — temos 2826 (limite aceitável)
- Muito sensível a hiperparâmetros (learning rate, dropout, n_heads)
- Difícil de debugar; risco alto de overfitting silencioso
- Infraestrutura necessária: GPU recomendado, ~2-4h de treino por fold

---

## 6. ROADMAP DE IMPLEMENTAÇÃO

```
FASE 1 — Engenharia de Features (1-2 semanas)
├── Calcular retornos lagados (1-5 dias)
├── Indicadores técnicos: RSI, MACD, Bollinger, ATR, Volume ratio
├── Volatilidade rolling: 5d, 21d, 63d
├── Spread US10Y - US2Y (curva de juros)
├── Parsing das colunas de evento texto → flags binários
└── Validação de look-ahead: TODA feature em t usa apenas dados até t-1

FASE 2 — Baseline (Modelo 1) — 1 semana
├── Logistic Regression com walk-forward
├── Estabelecer benchmark de acurácia e Sharpe
└── Análise de feature importance (coeficientes)

FASE 3 — Modelo Intermediário (Modelo 2) — 2-3 semanas
├── HMM para detecção de regimes (validar estados com plot de preços)
├── XGBoost por regime com Optuna para hyperparameter search
├── SHAP values para interpretabilidade
└── Backtest com custos: spread 2 pts + corretagem

FASE 4 — Modelo Avançado (Modelo 3) — 3-4 semanas
├── Implementar TFT via pytorch-forecasting
├── Definir known/unknown covariates
├── Treinar com walk-forward anual
└── Comparar com Modelo 2 em OOS comum

FASE 5 — Ensemble & Produção — 2 semanas
├── Ensemble ponderado Modelo 2 + Modelo 3
├── Sistema de confiança: só operar quando modelos concordam
└── Dashboard de monitoramento de regime
```

---

## 7. RISCOS TÉCNICOS CRÍTICOS

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Look-ahead bias | Alta | Crítico | Audit obrigatório de todo shift() |
| Overfitting temporal | Alta | Alto | Walk-forward estrito, sem shuffle |
| Degradação de regime | Média | Alto | Monitorar drift de performance OOS |
| Custos subestimados | Alta | Médio | Simular: spread 2 pts + R$1,20/contrato |
| Correlação espúria | Média | Médio | Usar retornos, não preços absolutos |
| Data snooping | Alta | Alto | Manter holdout de 2025-2026 intocado |

---

## 8. RECOMENDAÇÃO FINAL

> **Estado atual do projeto: Estado 1 — Exploração Inicial**  
> (dataset tratado, modelagem ainda não iniciada)

**Ordem de execução sugerida:**

1. ✅ Dataset tratado (concluído)
2. 🔲 **Próximo passo imediato**: engenharia de features + validação de look-ahead
3. 🔲 Modelo 1 como baseline (2 semanas)
4. 🔲 Modelo 2 se baseline > 53% acurácia OOS (3 semanas)
5. 🔲 Modelo 3 apenas se Modelo 2 mostrar instabilidade de regime (4 semanas)

**Critério de parada**: se nenhum modelo superar 54% de acurácia direcional consistente em walk-forward com custos reais → revisar hipótese, não adicionar complexidade.

---

*Documento gerado em: 20/05/2026*  
*Base: 2826 pregões, 2015-2026, granularidade D1*

import pandas as pd
import numpy as np
import os

# Caminhos
DATA_PATH = 'data/base_win_tratada_final.csv'
OUTPUT_CSV = 'data/base_win_features_selecionadas.csv'
OUTPUT_MD = '../.gemini/antigravity/brain/71ed50b7-9209-4fca-adfc-38ca7915de4b/scratch/resultado_features.md'

# Criar pasta scratch se não existir
os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)

def calcular_atr(df, n=14):
    high_low = df['winfut_high'] - df['winfut_low']
    high_close = np.abs(df['winfut_high'] - df['winfut_close'].shift())
    low_close = np.abs(df['winfut_low'] - df['winfut_close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(n).mean()

def calcular_rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=n-1, adjust=False).mean()
    ema_down = down.ewm(com=n-1, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    return 100 - (100 / (1 + rs))

def calcular_macd(series, n_fast=12, n_slow=26, n_sign=9):
    ema_fast = series.ewm(span=n_fast, adjust=False).mean()
    ema_slow = series.ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=n_sign, adjust=False).mean()
    return macd - signal  # Histogram

def calcular_dias_ate_vencimento_win(datas):
    # WIN vence na quarta-feira mais próxima do dia 15, meses pares (2,4,6,8,10,12)
    # Esta é uma aproximação razoável
    dias = []
    for d in datas:
        mes_venc = d.month if d.month % 2 == 0 else d.month + 1
        ano_venc = d.year
        if mes_venc > 12:
            mes_venc = 2
            ano_venc += 1
        # Aproximação: dia 15
        data_venc = pd.Timestamp(year=ano_venc, month=mes_venc, day=15)
        # Ajusta para quarta-feira
        while data_venc.dayofweek != 2:
            data_venc += pd.Timedelta(days=1)
        
        diff = (data_venc - d).days
        dias.append(max(0, diff))
    return pd.Series(dias, index=datas.index)

def gerar_features():
    print("Carregando dados...")
    df = pd.read_csv(DATA_PATH, parse_dates=['data'])
    df = df.sort_values('data').reset_index(drop=True)
    
    # === TARGET ===
    # Retorno dia D (open -> close) para evitar gap
    df['retorno_oc_futuro'] = ((df['winfut_close'] - df['winfut_open']) / df['winfut_open']).shift(-1)
    custo_estimado = 0.0004
    # Limite adaptativo
    df['retorno_oc_abs'] = ((df['winfut_close'] - df['winfut_open']) / df['winfut_open']).abs()
    limite = df['retorno_oc_abs'].rolling(252).quantile(0.30).shift(1) # usa info até D-1
    limite = limite.clip(lower=custo_estimado * 2)
    
    # 0 = Alta, 1 = Baixa
    cond_alta = df['retorno_oc_futuro'] > limite
    cond_baixa = df['retorno_oc_futuro'] < -limite
    
    df['target'] = np.nan
    df.loc[cond_alta, 'target'] = 0
    df.loc[cond_baixa, 'target'] = 1
    
    # === CAMADA 1: Candle D-1 ===
    df['ret_close_d1']       = df['winfut_close'].pct_change()
    df['body_pct_d1']        = (df['winfut_close'] - df['winfut_open']) / df['winfut_open']
    df['range_pct_d1']       = (df['winfut_high'] - df['winfut_low']) / df['winfut_open']
    df['close_position_d1']  = (df['winfut_close'] - df['winfut_low']) / (df['winfut_high'] - df['winfut_low'] + 1e-9)
    df['upper_shadow_d1']    = (df['winfut_high'] - df[['winfut_open','winfut_close']].max(axis=1)) / (df['range_pct_d1'] * df['winfut_open'] + 1e-9)
    df['lower_shadow_d1']    = (df[['winfut_open','winfut_close']].min(axis=1) - df['winfut_low']) / (df['range_pct_d1'] * df['winfut_open'] + 1e-9)
    
    df['atr_14']      = calcular_atr(df, 14)
    df['atr_ratio']   = df['range_pct_d1'] / (df['atr_14'] / df['winfut_close'])
    df['vol_20d']     = df['ret_close_d1'].rolling(20).std()
    df['rsi_14']      = calcular_rsi(df['winfut_close'], 14)
    df['macd_hist']   = calcular_macd(df['winfut_close'])
    df['ema9_21']     = (df['winfut_close'].ewm(span=9).mean() - df['winfut_close'].ewm(span=21).mean()) / df['winfut_close']
    bb_std = df['winfut_close'].rolling(20).std()
    bb_mean = df['winfut_close'].rolling(20).mean()
    df['bb_pct']      = (df['winfut_close'] - (bb_mean - 2*bb_std)) / (4*bb_std + 1e-9)
    
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'ret_lag{lag}'] = df['ret_close_d1'].shift(lag - 1)
        
    for w in [5, 10, 20]:
        df[f'ret_mean_{w}d'] = df['ret_close_d1'].rolling(w).mean()
        df[f'ret_std_{w}d']  = df['ret_close_d1'].rolling(w).std()
        df[f'ret_skew_{w}d'] = df['ret_close_d1'].rolling(w).skew()
        df[f'vol_ratio_{w}'] = df['vol_20d'] / (df['ret_close_d1'].rolling(w).std() + 1e-9)

    # === CAMADA 2: Gap ===
    df['gap_d1']          = (df['winfut_open'] - df['winfut_close'].shift(1)) / df['winfut_close'].shift(1)
    df['gap_vs_atr']      = df['gap_d1'] / (df['atr_14'] / df['winfut_close'] + 1e-9)
    df['gap_extremo']     = (df['gap_vs_atr'].abs() > 1.5).astype(int)
    df['gap_filled']      = (np.sign(df['gap_d1']) != np.sign(df['body_pct_d1'])).astype(int)
    df['gap_positivo']    = (df['gap_d1'] > 0).astype(int)
    df['gap_contra_spx']  = (np.sign(df['gap_d1']) != np.sign(df['sp500_change_pct'])).astype(int)
    
    # === CAMADA 3: SPX e Risco Global ===
    df['spx_ret_d1']      = df['sp500_change_pct']
    df['spx_ret_pos']     = df['spx_ret_d1'].clip(lower=0)
    df['spx_ret_neg']     = df['spx_ret_d1'].clip(upper=0)
    df['spx_slope_5d']    = df['sp500_close'].pct_change(5)
    df['spx_slope_20d']   = df['sp500_close'].pct_change(20)
    df['spx_zscore']      = df['spx_ret_d1'] / (df['spx_ret_d1'].rolling(20).std() + 1e-9)
    df['spx_drawdown_60'] = df['sp500_close'] / df['sp500_close'].rolling(60).max() - 1
    df['spx_acima_ma50']  = (df['sp500_close'] > df['sp500_close'].rolling(50).mean()).astype(int)
    df['spx_acima_ma200'] = (df['sp500_close'] > df['sp500_close'].rolling(200).mean()).astype(int)
    
    df['ndaq_ret_d1']   = df['nasdaq100_change_pct']
    df['ndaq_vs_spx']   = df['ndaq_ret_d1'] - df['spx_ret_d1']
    
    df['vix_nivel']     = df['vix_close']
    df['vix_ret_d1']    = df['vix_change_pct']
    df['vix_spike']     = (df['vix_ret_d1'] > 0.15).astype(int)
    df['vix_regime']    = pd.cut(df['vix_close'], bins=[-np.inf,15,20,25,35,np.inf], labels=[0,1,2,3,4]).astype(float)
    df['vix_acima_ma20']= (df['vix_close'] > df['vix_close'].rolling(20).mean()).astype(int)
    
    # === CAMADA 4: Macro Brasil e Intermercado ===
    externals = {
        'usd_brl': 'usd_brl_close', 'di1': 'di1_close',
        'us10y': 'us10y_close', 'us2y': 'us2y_close',
        'dxy': 'dxy_close', 'brent': 'petroleo_brent_close',
        'ouro': 'ouro_close', 'minerio': 'minerio_ferro_ultimo', 'ewz': 'ewz_close'
    }
    for name, col in externals.items():
        if col in df.columns:
            df[f'{name}_ret_d1']    = df[col].pct_change()
            df[f'{name}_slope_5d']  = df[col].pct_change(5)
            df[f'{name}_slope_20d'] = df[col].pct_change(20)
            df[f'{name}_vol_10d']   = df[col].pct_change().rolling(10).std()
            df[f'{name}_ma_regime'] = (df[col] > df[col].rolling(252).mean()).astype(int)
            
    df['us_yield_spread']   = df['us10y_close'] - df['us2y_close']
    df['curva_invertida']   = (df['us_yield_spread'] < 0).astype(int)
    
    df['win_vs_spx']        = df['ret_close_d1'] - df['spx_ret_d1']
    if 'ewz_ret_d1' in df.columns:
        df['win_vs_ewz']        = df['ret_close_d1'] - df['ewz_ret_d1']
    if 'brent_ret_d1' in df.columns and 'minerio_ret_d1' in df.columns:
        df['win_vs_commodities']= df['ret_close_d1'] - (0.5*df['brent_ret_d1'] + 0.5*df['minerio_ret_d1'])
        
    # === CAMADA 5: Regimes ===
    df['regime_vol_win'] = pd.cut(df['vol_20d'], 
                                  bins=[-np.inf, df['vol_20d'].quantile(0.33), df['vol_20d'].quantile(0.67), np.inf],
                                  labels=[0,1,2]).astype(float)
    
    ma50  = df['winfut_close'].rolling(50).mean()
    ma200 = df['winfut_close'].rolling(200).mean()
    df['regime_tendencia'] = np.where(df['winfut_close'] > ma50,
        np.where(ma50 > ma200, 2, 1),     
        np.where(ma50 < ma200, -2, -1))
        
    df['regime_label'] = np.where(df['regime_tendencia'] == 2, 'bull_forte',
                         np.where(df['regime_tendencia'] == 1, 'bull_fraco',
                         np.where(df['regime_tendencia'] == -1, 'bear_fraco', 'bear_forte')))
                         
    df['regime_juros_altos'] = (df['di1_close'] > 10.0).astype(int)
    df['regime_di_subindo']  = (df['di1_close'].pct_change(10) > 0).astype(int)
    df['regime_vix_alto']    = (df['vix_close'] > df['vix_close'].rolling(252).quantile(0.75)).astype(int)
    df['regime_spx_bull']    = (df['sp500_close'] > df['sp500_close'].rolling(200).mean()).astype(int)
    df['regime_dxy_forte']   = (df['dxy_close'] > df['dxy_close'].rolling(252).mean()).astype(int)
    
    df['regime_favoravel'] = ((df['regime_spx_bull'] == 1) & (df['regime_juros_altos'] == 0) & (df['regime_dxy_forte'] == 0)).astype(int)
    
    # === CAMADA 6: Calendario ===
    df['dia_da_semana']      = df['data'].dt.dayofweek
    df['mes']                = df['data'].dt.month
    df['trimestre']          = df['data'].dt.quarter
    df['semana_do_mes']      = (df['data'].dt.day - 1) // 7 + 1
    
    df['dias_ate_venc']  = calcular_dias_ate_vencimento_win(df['data'])
    df['venc_prox']      = (df['dias_ate_venc'] <= 3).astype(int)
    df['semana_ciclo']   = (df['dias_ate_venc'] // 5).clip(0, 4)
    
    # Parse de eventos existentes (já vieram 0/1 no CSV original? Vamos checar colunas evento_*)
    # Pelo dicionário e CSV tratado, existem colunas: evento_brasil_copom, evento_g6_eua_juros_fed, etc.
    event_cols = [c for c in df.columns if 'evento_' in c and c not in ['evento_brasil_focus', 'eventos_brasil', 'eventos_g6_eua', 'eventos_g6_global']]
    for c in event_cols:
        df[c] = df[c].fillna(0).astype(int)
        
    return df

def analise_selecao(df):
    print("Iniciando seleção de features (IC e Colinearidade)...")
    # Ignorar colunas originais OHLC e não-features
    nao_features = ['data', 'target', 'retorno_oc_futuro', 'retorno_oc_abs', 'regime_label']
    colunas_originais = [c for c in df.columns if '_open' in c or '_high' in c or '_low' in c or '_volume' in c]
    # Manter closes que foram usados como nível, mas é melhor focar nas calculadas
    
    features_candidates = [c for c in df.columns if c not in nao_features and c not in colunas_originais and pd.api.types.is_numeric_dtype(df[c])]
    
    # === IC Global ===
    ic_global = {}
    for col in features_candidates:
        mask = df[col].notna() & df['target'].notna()
        if mask.sum() > 100:
            corr = df.loc[mask, [col, 'target']].corr(method='spearman').iloc[0,1]
            ic_global[col] = abs(corr) if not pd.isna(corr) else 0

    # === IC por Regime ===
    ic_por_regime = {}
    for regime in ['bull_forte', 'bull_fraco', 'bear_fraco', 'bear_forte']:
        df_r = df[df['regime_label'] == regime]
        ic_regime = {}
        for col in features_candidates:
            mask = df_r[col].notna() & df_r['target'].notna()
            if mask.sum() >= 30:
                corr = df_r.loc[mask, [col, 'target']].corr(method='spearman').iloc[0,1]
                ic_regime[col] = abs(corr) if not pd.isna(corr) else 0
        ic_por_regime[regime] = ic_regime
        
    # === Filtragem IC >= 0.02 ===
    features_validas = [f for f, ic in ic_global.items() if ic >= 0.02]
    
    # === Multicolinearidade ===
    corr_m  = df[features_validas].corr(method='pearson').abs()
    upper   = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    features_final = [f for f in features_validas if f not in to_drop]
    
    # === Relatório ===
    relatorio = []
    relatorio.append("# Relatório de Engenharia e Seleção de Features WIN")
    relatorio.append("\n## Top 20 Features (IC Global)")
    top_global = pd.Series(ic_global).sort_values(ascending=False).head(20)
    relatorio.append("| Feature | IC Spearman |")
    relatorio.append("|---|---|")
    for f, v in top_global.items():
        relatorio.append(f"| {f} | {v:.4f} |")
        
    relatorio.append("\n## Top 5 Features por Regime")
    for regime in ic_por_regime.keys():
        relatorio.append(f"### Regime: {regime}")
        top_r = pd.Series(ic_por_regime[regime]).sort_values(ascending=False).head(5)
        relatorio.append("| Feature | IC Spearman |")
        relatorio.append("|---|---|")
        for f, v in top_r.items():
            relatorio.append(f"| {f} | {v:.4f} |")
            
    relatorio.append("\n## Multicolinearidade")
    relatorio.append(f"**Removidas ({len(to_drop)}):** {', '.join(to_drop)}")
    
    relatorio.append("\n## Resultado Final")
    relatorio.append(f"De {len(features_candidates)} features originais, restaram **{len(features_final)}** features finais robustas.")
    
    with open(OUTPUT_MD, 'w') as f:
        f.write('\n'.join(relatorio))
        
    # Salvar CSV final
    colunas_salvar = ['data', 'target', 'regime_label'] + features_final
    # Preencher NaNs residuais com ffill/bfill antes de salvar ou salvar com NaN mesmo (XGBoost lida bem)
    df_final = df[colunas_salvar]
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"Relatório salvo em: {OUTPUT_MD}")
    print(f"Dataset final salvo em: {OUTPUT_CSV}")

if __name__ == "__main__":
    df_features = gerar_features()
    analise_selecao(df_features)

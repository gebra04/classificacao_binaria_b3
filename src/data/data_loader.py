"""
Data Loader — Carregamento, normalização e split temporal dos dados WIN.
Centraliza toda a lógica de preparação de dados para os 3 produtos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# Caminhos relativos à raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Splits temporais fixos (não mudar sem justificativa)
SPLITS = {
    "treino": ("2015-01-01", "2022-12-31"),
    "validacao": ("2023-01-01", "2023-12-31"),
    "holdout_dev": ("2024-01-01", "2024-12-31"),
    "holdout_final": ("2025-01-01", "2026-12-31"),
}

# Colunas que NÃO são features (metadados, target, labels)
META_COLS = ["data", "target", "regime_label"]

# Colunas de preço bruto que não devem entrar como features diretas
RAW_PRICE_COLS_PREFIXES = [
    "winfut_open", "winfut_high", "winfut_low", "winfut_volume",
    "usd_brl_open", "usd_brl_high", "usd_brl_low",
    "di1_open", "di1_high", "di1_low", "di1_volume",
    "us2y_open", "us2y_high", "us2y_low",
    "us10y_open", "us10y_high", "us10y_low",
    "dxy_open", "dxy_high", "dxy_low",
    "vix_open", "vix_high", "vix_low",
    "nasdaq100_open", "nasdaq100_high", "nasdaq100_low", "nasdaq100_volume",
    "sp500_open", "sp500_high", "sp500_low",
    "minerio_ferro_abertura", "minerio_ferro_maxima", "minerio_ferro_minima",
]


def load_raw_data() -> pd.DataFrame:
    """Carrega base tratada final com parse de datas."""
    path = DATA_DIR / "base_win_tratada_final.csv"
    df = pd.read_csv(path, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def load_features_data() -> pd.DataFrame:
    """Carrega base com features já selecionadas (v1)."""
    path = DATA_DIR / "base_win_features_selecionadas.csv"
    df = pd.read_csv(path, parse_dates=["data"])
    df = df.sort_values("data").reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna apenas colunas numéricas que são features (exclui meta e preços brutos)."""
    feature_cols = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if col in RAW_PRICE_COLS_PREFIXES:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def temporal_split(
    df: pd.DataFrame,
    split_config: dict = None,
) -> dict[str, pd.DataFrame]:
    """
    Divide o DataFrame em splits temporais.
    
    Returns:
        dict com chaves 'treino', 'validacao', 'holdout_dev', 'holdout_final'
    """
    if split_config is None:
        split_config = SPLITS

    splits = {}
    for name, (start, end) in split_config.items():
        mask = (df["data"] >= start) & (df["data"] <= end)
        splits[name] = df[mask].copy().reset_index(drop=True)

    return splits


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler = None,
    fit: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Extrai e normaliza features numéricas.
    
    Args:
        df: DataFrame com os dados
        feature_cols: lista de colunas de features
        scaler: scaler pré-treinado (None para criar novo)
        fit: se True, faz fit_transform; se False, apenas transform
    
    Returns:
        (array normalizado, scaler usado)
    """
    X = df[feature_cols].values.astype(np.float32)

    # Substituir NaN/inf por 0 (XGBoost lida nativo, mas AE precisa)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X).astype(np.float32)
    else:
        X_scaled = scaler.transform(X).astype(np.float32)

    return X_scaled, scaler


def prepare_autoencoder_data(
    feature_cols: list[str] = None,
    source: str = "features",
) -> dict:
    """
    Pipeline completo de preparação de dados para o autoencoder.
    
    Args:
        feature_cols: colunas de features (None = auto-detectar)
        source: 'features' para base_win_features_selecionadas, 'raw' para base_win_tratada_final
    
    Returns:
        dict com X_train, X_val, X_holdout, scaler, feature_cols, splits
    """
    if source == "features":
        df = load_features_data()
    else:
        df = load_raw_data()

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    splits = temporal_split(df)

    # Fit scaler apenas no treino
    X_train, scaler = prepare_features(
        splits["treino"], feature_cols, fit=True
    )
    X_val, _ = prepare_features(
        splits["validacao"], feature_cols, scaler=scaler, fit=False
    )
    X_holdout, _ = prepare_features(
        splits["holdout_dev"], feature_cols, scaler=scaler, fit=False
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_holdout": X_holdout,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "splits": splits,
    }

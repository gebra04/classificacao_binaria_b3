"""
Feature Selection Pipeline — 4 estágios de seleção robusta.

Estágio 1: Filtragem estatística (IC, MI, estabilidade temporal, colinearidade)
Estágio 2: Feature importance via modelo (XGBoost permutation importance)
Estágio 3: Autoencoder-based selection (reconstrução por feature)
Estágio 4: Curadoria + exportação final

Dica do amigo: "faz um feature selection bom, acho que vai dar bom"
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from pathlib import Path

from src.data.data_loader import (
    load_features_data,
    load_raw_data,
    get_feature_columns,
    temporal_split,
    META_COLS,
    DATA_DIR,
)


def estagio_1_filtragem_estatistica(
    df: pd.DataFrame,
    feature_cols: list[str],
    ic_threshold: float = 0.02,
    mi_threshold: float = 0.001,
    corr_threshold: float = 0.90,
    ic_window: int = 252,
) -> dict:
    """
    Estágio 1: Filtragem estatística.
    
    - IC Spearman global ≥ ic_threshold
    - Mutual Information com target
    - Estabilidade temporal do IC (não muda de sinal)
    - Remoção de multicolinearidade (r > corr_threshold)
    """
    results = {"estagio": 1, "detalhes": {}}

    # Apenas linhas com target válido
    df_valid = df.dropna(subset=["target"]).copy()
    target = df_valid["target"].values

    # --- IC Spearman global ---
    ic_global = {}
    for col in feature_cols:
        vals = df_valid[col].values
        mask = np.isfinite(vals)
        if mask.sum() < 100:
            continue
        corr, _ = spearmanr(vals[mask], target[mask])
        ic_global[col] = abs(corr) if not np.isnan(corr) else 0.0

    # --- Mutual Information ---
    X_mi = df_valid[feature_cols].fillna(0).values
    mi_scores = mutual_info_classif(
        X_mi, target.astype(int), random_state=42, n_neighbors=5
    )
    mi_dict = dict(zip(feature_cols, mi_scores))

    # --- Estabilidade temporal do IC ---
    ic_stable = {}
    for col in feature_cols:
        if col not in ic_global:
            continue
        # Calcular IC rolling
        ic_rolling = []
        for start in range(0, len(df_valid) - ic_window, ic_window // 2):
            window = df_valid.iloc[start : start + ic_window]
            vals = window[col].values
            tgt = window["target"].values
            mask = np.isfinite(vals) & np.isfinite(tgt)
            if mask.sum() < 50:
                continue
            corr, _ = spearmanr(vals[mask], tgt[mask])
            if not np.isnan(corr):
                ic_rolling.append(corr)

        if len(ic_rolling) >= 3:
            # Estável = mesmo sinal em ≥ 70% das janelas
            signs = np.sign(ic_rolling)
            dominant_sign = np.sign(np.mean(ic_rolling))
            stability = np.mean(signs == dominant_sign)
            ic_stable[col] = stability
        else:
            ic_stable[col] = 0.5  # incerto

    # --- Filtrar por IC + MI ---
    features_ic_ok = {f for f, ic in ic_global.items() if ic >= ic_threshold}
    features_mi_ok = {f for f, mi in mi_dict.items() if mi >= mi_threshold}
    features_stable = {f for f, s in ic_stable.items() if s >= 0.6}

    # Interseção: passar nos 3 filtros
    features_stage1 = list(features_ic_ok & features_mi_ok & features_stable)

    # --- Remoção de multicolinearidade ---
    if len(features_stage1) > 2:
        corr_m = df_valid[features_stage1].corr(method="pearson").abs()
        upper = corr_m.where(
            np.triu(np.ones(corr_m.shape), k=1).astype(bool)
        )
        # Quando par tem corr > threshold, remover a feature com menor IC
        to_drop = set()
        for col in upper.columns:
            correlated = upper.index[upper[col] > corr_threshold].tolist()
            for corr_col in correlated:
                # Manter a de maior IC
                ic_col = ic_global.get(col, 0)
                ic_corr = ic_global.get(corr_col, 0)
                drop = corr_col if ic_col >= ic_corr else col
                to_drop.add(drop)

        features_stage1 = [f for f in features_stage1 if f not in to_drop]
    else:
        to_drop = set()

    results["features_selecionadas"] = sorted(features_stage1)
    results["n_total"] = len(feature_cols)
    results["n_ic_ok"] = len(features_ic_ok)
    results["n_mi_ok"] = len(features_mi_ok)
    results["n_stable"] = len(features_stable)
    results["n_apos_colinearidade"] = len(features_stage1)
    results["removidas_colinearidade"] = sorted(to_drop)
    results["detalhes"]["ic_global"] = ic_global
    results["detalhes"]["mi_scores"] = mi_dict
    results["detalhes"]["ic_stability"] = ic_stable

    return results


def estagio_2_importance_modelo(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int = 3,
    top_n: int = 30,
    min_folds_present: int = 2,
) -> dict:
    """
    Estágio 2: Feature importance via XGBoost + permutation importance.
    Treina em folds temporais e mantém features estáveis.
    """
    results = {"estagio": 2, "detalhes": {}}

    df_valid = df.dropna(subset=["target"]).copy()
    X = df_valid[feature_cols].fillna(0).values
    y = df_valid["target"].values.astype(int)
    n = len(df_valid)

    # Folds temporais simples
    fold_size = n // (n_folds + 1)
    importance_counts = {f: 0 for f in feature_cols}
    fold_importances = {}

    for i in range(n_folds):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Permutation importance no test set
        perm_result = permutation_importance(
            model, X_test, y_test,
            n_repeats=10, random_state=42, scoring="balanced_accuracy",
        )

        # Top-N features deste fold
        top_idx = np.argsort(perm_result.importances_mean)[::-1][:top_n]
        top_features = [feature_cols[j] for j in top_idx]
        fold_importances[f"fold_{i}"] = {
            feature_cols[j]: float(perm_result.importances_mean[j])
            for j in top_idx
        }

        for f in top_features:
            importance_counts[f] += 1

    # Manter features que aparecem no top-N em ≥ min_folds_present folds
    features_stage2 = [
        f for f, count in importance_counts.items()
        if count >= min_folds_present
    ]

    results["features_selecionadas"] = sorted(features_stage2)
    results["n_selecionadas"] = len(features_stage2)
    results["detalhes"]["fold_importances"] = fold_importances
    results["detalhes"]["counts"] = {
        f: c for f, c in importance_counts.items() if c > 0
    }

    return results


def estagio_3_autoencoder_selection(
    df: pd.DataFrame,
    feature_cols: list[str],
    latent_dim: int = 12,
    epochs: int = 200,
    redundancy_threshold: float = 0.01,
) -> dict:
    """
    Estágio 3: Autoencoder-based feature selection.
    
    Treina AE preliminar e analisa erro de reconstrução por feature.
    Features com erro ≈ 0 são redundantes. Features com erro altíssimo são ruidosas.
    """
    import torch
    import torch.nn as nn

    results = {"estagio": 3, "detalhes": {}}

    df_valid = df.dropna(subset=["target"]).copy()
    X = df_valid[feature_cols].fillna(0).values.astype(np.float32)

    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split simples para treino/val
    split_idx = int(len(X_scaled) * 0.8)
    X_train = torch.tensor(X_scaled[:split_idx])
    X_val = torch.tensor(X_scaled[split_idx:])

    n_features = X_train.shape[1]

    # AE simples para seleção (não é o backbone final)
    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, n_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    model = SimpleAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")

    # Treinar
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        x_recon = model(X_train)
        loss = loss_fn(x_recon, X_train).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val)
            val_loss = loss_fn(val_recon, X_val).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    # Analisar erro de reconstrução por feature
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(X_scaled))
        per_feature_mse = (
            (recon - torch.tensor(X_scaled)) ** 2
        ).mean(dim=0).numpy()

    feature_errors = dict(zip(feature_cols, per_feature_mse))

    # Features redundantes: erro muito baixo (informação já capturada por outras)
    median_error = np.median(per_feature_mse)
    redundant = [
        f for f, e in feature_errors.items()
        if e < redundancy_threshold * median_error
    ]

    # Features ruidosas: erro muito alto (> 3× mediana)
    noisy = [
        f for f, e in feature_errors.items()
        if e > 3.0 * median_error
    ]

    # Manter tudo que não é redundante (ruidosas ficam, podem ter informação)
    features_stage3 = [f for f in feature_cols if f not in redundant]

    results["features_selecionadas"] = sorted(features_stage3)
    results["n_selecionadas"] = len(features_stage3)
    results["redundantes"] = sorted(redundant)
    results["ruidosas"] = sorted(noisy)
    results["detalhes"]["per_feature_mse"] = feature_errors
    results["detalhes"]["best_val_loss"] = best_val_loss

    return results


def run_full_pipeline(
    source: str = "raw",
    output_csv: str = "features_curadas_v2.csv",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Executa pipeline completo de feature selection (Estágios 1-3).
    
    Args:
        source: 'raw' para base_win_tratada_final, 'features' para base_win_features_selecionadas
        output_csv: nome do arquivo de saída
        verbose: se True, imprime progresso
    
    Returns:
        (DataFrame final, lista de features selecionadas, relatório completo)
    """
    if source == "features":
        df = load_features_data()
    else:
        df = load_raw_data()

    all_features = get_feature_columns(df)

    if verbose:
        print(f"=== Feature Selection Pipeline ===")
        print(f"Dataset: {len(df)} linhas, {len(all_features)} features candidatas\n")

    # Estágio 1
    if verbose:
        print("--- Estágio 1: Filtragem Estatística ---")
    r1 = estagio_1_filtragem_estatistica(df, all_features)
    features_after_1 = r1["features_selecionadas"]
    if verbose:
        print(f"  IC ok: {r1['n_ic_ok']} | MI ok: {r1['n_mi_ok']} | "
              f"Estáveis: {r1['n_stable']} | Após colinearidade: {r1['n_apos_colinearidade']}")
        print(f"  → {len(features_after_1)} features sobreviveram\n")

    # Estágio 2
    if verbose:
        print("--- Estágio 2: Importance via Modelo ---")
    r2 = estagio_2_importance_modelo(df, features_after_1)
    features_after_2 = r2["features_selecionadas"]
    if verbose:
        print(f"  → {r2['n_selecionadas']} features no top-30 em ≥2 folds\n")

    # Estágio 3
    if verbose:
        print("--- Estágio 3: Autoencoder-based Selection ---")

    # Usar interseção dos estágios 1 e 2, mas dar chance extra ao estágio 1
    # Features que passaram no estágio 1 mas não no 2 podem ter valor para o AE
    features_for_ae = list(set(features_after_1) | set(features_after_2))
    r3 = estagio_3_autoencoder_selection(df, features_for_ae)
    features_after_3 = r3["features_selecionadas"]
    if verbose:
        print(f"  Redundantes removidas: {len(r3['redundantes'])}")
        print(f"  Ruidosas identificadas: {len(r3['ruidosas'])}")
        print(f"  → {r3['n_selecionadas']} features finais\n")

    # Interseção final: features que passaram nos estágios 1+2 E não foram
    # removidas pelo estágio 3
    features_final = sorted(
        set(features_after_2) & set(features_after_3)
    )

    # Garantir representação mínima das camadas
    # (verificação informativa, não remove features)
    camada_check = {
        "preco": any("ret_" in f or "body_" in f or "range_" in f or "close_" in f for f in features_final),
        "gap": any("gap_" in f for f in features_final),
        "global_risk": any("spx_" in f or "vix_" in f or "ndaq_" in f for f in features_final),
        "macro": any("usd_" in f or "di1_" in f or "us10y_" in f or "dxy_" in f or "brent_" in f for f in features_final),
        "regime": any("regime_" in f for f in features_final),
        "calendario": any("dia_" in f or "venc" in f or "evento_" in f for f in features_final),
    }
    if verbose:
        print(f"=== RESULTADO FINAL: {len(features_final)} features ===")
        print(f"Cobertura de camadas: {camada_check}\n")
        print(f"Features: {features_final}")

    # Salvar CSV
    output_path = DATA_DIR / output_csv
    cols_to_save = ["data", "target", "regime_label"] + features_final
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    df[cols_to_save].to_csv(output_path, index=False)
    if verbose:
        print(f"\nSalvo em: {output_path}")

    report = {
        "estagio_1": r1,
        "estagio_2": r2,
        "estagio_3": r3,
        "features_final": features_final,
        "n_final": len(features_final),
        "camada_check": camada_check,
    }

    return df, features_final, report


if __name__ == "__main__":
    df, features, report = run_full_pipeline(source="features", verbose=True)

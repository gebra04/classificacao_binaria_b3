"""
Script de execução do Produto 3 — Detector de Dia Atípico.

Pipeline completo:
1. Carrega DVAE backbone treinado
2. Calibra o detector de anomalias no período de treino
3. Roda detecção em todo o dataset
4. Valida contra eventos conhecidos (COVID, circuit breakers, etc.)
5. Gera relatório com métricas, visualizações e exemplos
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.data_loader import (
    load_features_data,
    get_feature_columns,
    temporal_split,
    prepare_features,
)
from src.models.autoencoder import DVAE, DVAETrainer
from src.products.anomaly_detector import AnomalyDetector

MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


def load_trained_dvae():
    """Carrega DVAE backbone treinado + scaler + feature_cols."""
    trainer = DVAETrainer.load(MODELS_DIR / "dvae_backbone.pt")
    
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open(MODELS_DIR / "feature_cols.json", "r") as f:
        feature_cols = json.load(f)
    
    return trainer.model, scaler, feature_cols


def run_anomaly_detection():
    """Pipeline completo do Produto 3."""
    print("=" * 60)
    print("PRODUTO 3 — DETECTOR DE DIA ATÍPICO")
    print("=" * 60)

    # --- 1. Carregar modelo e dados ---
    print("\n📦 Carregando modelo e dados...")
    dvae, scaler, feature_cols = load_trained_dvae()
    
    df = load_features_data()
    splits = temporal_split(df)
    
    X_train, _ = prepare_features(splits["treino"], feature_cols, scaler=scaler, fit=False)
    X_val, _ = prepare_features(splits["validacao"], feature_cols, scaler=scaler, fit=False)
    X_holdout, _ = prepare_features(splits["holdout_dev"], feature_cols, scaler=scaler, fit=False)
    
    # Dataset completo para análise temporal
    X_all, _ = prepare_features(df, feature_cols, scaler=scaler, fit=False)
    
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Validação: {X_val.shape[0]} amostras")
    print(f"  Holdout: {X_holdout.shape[0]} amostras")
    print(f"  Total: {X_all.shape[0]} amostras")

    # --- 2. Calibrar detector ---
    print("\n🔧 Calibrando detector no período de treino (2015-2022)...")
    detector = AnomalyDetector(dvae, scaler, feature_cols)
    detector.fit(X_train)
    
    # Salvar calibração
    detector.save(MODELS_DIR / "anomaly_detector_calibration.pkl")

    # --- 3. Rodar detecção em todo o dataset ---
    print("\n🔍 Rodando detecção em todo o dataset...")
    all_scores = detector.score(X_all)
    if isinstance(all_scores, dict):
        all_scores = [all_scores]
    
    # Adicionar scores ao DataFrame
    df_results = df.copy()
    df_results["anomaly_score"] = [s["anomaly_score"] for s in all_scores]
    df_results["anomaly_nivel"] = [s["nivel"] for s in all_scores]
    df_results["raw_error"] = [s["raw_error"] for s in all_scores]
    df_results["top_contributor_1"] = [
        s["top_contributors"][0]["feature"] if s["top_contributors"] else "N/A"
        for s in all_scores
    ]
    df_results["top_contributor_1_z"] = [
        s["top_contributors"][0]["z_score"] if s["top_contributors"] else 0.0
        for s in all_scores
    ]
    df_results["explicacao"] = [s["explicacao"] for s in all_scores]

    # --- 4. Estatísticas gerais ---
    print("\n📊 ESTATÍSTICAS GERAIS")
    print("-" * 40)
    nivel_counts = df_results["anomaly_nivel"].value_counts()
    for nivel in ["NORMAL", "ATENCAO", "ATIPICO", "CRITICO"]:
        n = nivel_counts.get(nivel, 0)
        pct = n / len(df_results) * 100
        print(f"  {nivel:10s}: {n:5d} dias ({pct:.1f}%)")
    
    print(f"\n  Score médio:   {df_results['anomaly_score'].mean():.4f}")
    print(f"  Score mediana: {df_results['anomaly_score'].median():.4f}")
    print(f"  Score Q90:     {df_results['anomaly_score'].quantile(0.90):.4f}")
    print(f"  Score Q99:     {df_results['anomaly_score'].quantile(0.99):.4f}")

    # --- 5. Validação contra eventos conhecidos ---
    print("\n\n🎯 VALIDAÇÃO CONTRA EVENTOS CONHECIDOS")
    print("=" * 60)
    
    eventos_conhecidos = {
        # COVID-19 crash
        "2020-03-09": "COVID Circuit Breaker #1 — Ibovespa -12%",
        "2020-03-11": "COVID OMS declara pandemia — VIX > 50",
        "2020-03-12": "COVID Circuit Breaker #2 — Ibovespa -14%",
        "2020-03-16": "COVID Circuit Breaker #3 — Ibovespa -13%",
        "2020-03-18": "COVID VIX > 80 — máxima histórica",
        "2020-03-23": "COVID Fundo do crash — Ibovespa na mínima",
        # Joesley Day
        "2017-05-18": "Joesley Day — gravação Temer, Ibovespa -8.8%",
        # Greve dos caminhoneiros
        "2018-05-28": "Greve dos caminhoneiros — caos logístico",
        # Eleições 2018
        "2018-10-08": "Pós-1º turno 2018 — Bolsonaro líder, Ibovespa +4.6%",
        # Fed
        "2022-06-16": "Fed sobe juros 75bps (surpresa hawkish)",
        "2023-03-13": "Colapso Silicon Valley Bank — contágio financeiro",
        # Brasil 2024-2025
        "2024-01-03": "Primeiro pregão 2024 — ajuste de posições",
        "2025-04-07": "Tariffaço Trump reciprocal tariffs",
    }
    
    validacao_resultados = []
    for data_str, descricao in eventos_conhecidos.items():
        match = df_results[df_results["data"] == data_str]
        if len(match) > 0:
            row = match.iloc[0]
            score = row["anomaly_score"]
            nivel = row["anomaly_nivel"]
            top1 = row["top_contributor_1"]
            status = "✅" if score >= 0.80 else ("⚠️" if score >= 0.70 else "❌")
            print(f"  {status} {data_str} | Score: {score:.3f} [{nivel}]")
            print(f"     {descricao}")
            print(f"     Top contributor: {top1}")
            print()
            validacao_resultados.append({
                "data": data_str,
                "descricao": descricao,
                "score": float(score),
                "nivel": str(nivel),
                "detectado": bool(score >= 0.80),
            })
        else:
            print(f"  ⚪ {data_str} — não encontrado no dataset")
            print(f"     {descricao}\n")
    
    # Taxa de detecção
    detectados = sum(1 for v in validacao_resultados if v["detectado"])
    total_eventos = len(validacao_resultados)
    print(f"\n  Taxa de detecção: {detectados}/{total_eventos} "
          f"({detectados/total_eventos*100:.0f}%) eventos com score ≥ 0.80")

    # --- 6. Top 20 dias mais atípicos ---
    print("\n\n📈 TOP 20 DIAS MAIS ATÍPICOS (todo o dataset)")
    print("-" * 60)
    top20 = df_results.nlargest(20, "anomaly_score")
    for _, row in top20.iterrows():
        data_str = row["data"].strftime("%Y-%m-%d")
        print(f"  {data_str} | Score: {row['anomaly_score']:.3f} [{row['anomaly_nivel']}]")
        print(f"     Explicação: {row['explicacao']}")
        print()

    # --- 7. Análise por período ---
    print("\n📅 SCORE MÉDIO POR ANO")
    print("-" * 40)
    df_results["ano"] = df_results["data"].dt.year
    yearly = df_results.groupby("ano").agg(
        score_medio=("anomaly_score", "mean"),
        score_max=("anomaly_score", "max"),
        n_atipicos=("anomaly_nivel", lambda x: (x.isin(["ATIPICO", "CRITICO"])).sum()),
    )
    for ano, row in yearly.iterrows():
        print(f"  {ano}: μ={row['score_medio']:.3f} | max={row['score_max']:.3f} | "
              f"atípicos={row['n_atipicos']}")

    # --- 8. Visualizações ---
    print("\n\n🎨 Gerando visualizações...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 8a. Score ao longo do tempo
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    dates = df_results["data"].values
    scores = df_results["anomaly_score"].values
    
    # Anomaly score timeline
    ax = axes[0]
    ax.fill_between(dates, 0, scores, alpha=0.3, color="steelblue")
    ax.plot(dates, scores, linewidth=0.5, color="steelblue", alpha=0.8)
    ax.axhline(y=0.90, color="orange", linestyle="--", alpha=0.7, label="Threshold Q90")
    ax.axhline(y=0.99, color="red", linestyle="--", alpha=0.7, label="Threshold Q99")
    
    # Marcar eventos conhecidos
    for data_str in eventos_conhecidos:
        match = df_results[df_results["data"] == data_str]
        if len(match) > 0:
            ax.axvline(x=match.iloc[0]["data"], color="red", alpha=0.3, linewidth=0.5)
    
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Produto 3 — Score de Dia Atípico (2015-2026)")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.05)
    
    # Retorno WIN
    ax = axes[1]
    if "winfut_change_pct" in df_results.columns:
        rets = df_results["winfut_change_pct"].values * 100
        colors_bar = np.where(scores > 0.90, "red", np.where(scores > 0.75, "orange", "gray"))
        ax.bar(dates, rets, width=1.5, color=colors_bar, alpha=0.6)
        ax.set_ylabel("Retorno WIN (%)")
        ax.set_title("Retorno diário colorido por nível de anomalia (vermelho = atípico)")
    
    # Distribuição de scores por ano
    ax = axes[2]
    df_results["ano"] = df_results["data"].dt.year
    bp_data = [df_results[df_results["ano"] == y]["anomaly_score"].values 
               for y in sorted(df_results["ano"].unique())]
    bp = ax.boxplot(bp_data, tick_labels=sorted(df_results["ano"].unique()), 
                    patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.5)
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Distribuição do Score por Ano")
    ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "produto3_anomaly_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Timeline salva")

    # 8b. Heatmap: top contributors em dias atípicos
    atypical_days = df_results[df_results["anomaly_nivel"].isin(["ATIPICO", "CRITICO"])]
    if len(atypical_days) > 0:
        import torch
        
        X_atypical, _ = prepare_features(atypical_days, feature_cols, scaler=scaler, fit=False)
        
        dvae.eval()
        with torch.no_grad():
            per_feat_err = dvae.get_reconstruction_error(
                torch.tensor(X_atypical, dtype=torch.float32), per_feature=True
            ).numpy()
        
        # Normalizar pelo baseline de treino
        per_feat_normalized = (per_feat_err - detector.per_feature_mean) / detector.per_feature_std
        
        # Top 15 features mais frequentemente anômalas
        mean_contrib = np.abs(per_feat_normalized).mean(axis=0)
        top15_idx = np.argsort(mean_contrib)[::-1][:15]
        top15_names = [feature_cols[i] for i in top15_idx]
        
        # Heatmap das últimas 30 anomalias
        n_show = min(30, len(atypical_days))
        heatmap_data = per_feat_normalized[-n_show:, top15_idx]
        heatmap_dates = atypical_days["data"].iloc[-n_show:].dt.strftime("%Y-%m-%d").values
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            heatmap_data,
            xticklabels=top15_names,
            yticklabels=heatmap_dates,
            cmap="RdYlBu_r",
            center=0,
            vmin=-3, vmax=3,
            ax=ax,
        )
        ax.set_title("Contribuição por Feature — Dias Atípicos (z-score do erro de reconstrução)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "produto3_heatmap_contributors.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✅ Heatmap de contributors salva")

    # 8c. Scatter: anomaly score vs retorno absoluto
    fig, ax = plt.subplots(figsize=(10, 6))
    if "winfut_change_pct" in df_results.columns:
        abs_ret = df_results["winfut_change_pct"].abs().values * 100
        ax.scatter(scores, abs_ret, alpha=0.3, s=5, color="steelblue")
        ax.axvline(x=0.90, color="red", linestyle="--", alpha=0.5, label="Q90")
        
        # Correlation
        mask = np.isfinite(abs_ret) & np.isfinite(scores)
        corr = np.corrcoef(scores[mask], abs_ret[mask])[0, 1]
        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("|Retorno WIN| (%)")
        ax.set_title(f"Anomaly Score vs Magnitude do Retorno (corr = {corr:.3f})")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "produto3_score_vs_return.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Scatter score vs retorno salva")

    # 8d. Análise: dias atípicos têm retornos maiores?
    print("\n\n📊 ANÁLISE: DIAS ATÍPICOS TÊM RETORNOS MAIORES?")
    print("-" * 50)
    if "winfut_change_pct" in df_results.columns:
        for nivel in ["NORMAL", "ATENCAO", "ATIPICO", "CRITICO"]:
            mask = df_results["anomaly_nivel"] == nivel
            if mask.sum() > 0:
                abs_ret_nivel = df_results.loc[mask, "winfut_change_pct"].abs() * 100
                print(f"  {nivel:10s}: |ret| médio = {abs_ret_nivel.mean():.3f}% | "
                      f"max = {abs_ret_nivel.max():.3f}% | n = {mask.sum()}")

    # --- 9. Exemplo de output do Produto 3 (último dia) ---
    print("\n\n🎯 EXEMPLO DE OUTPUT — ÚLTIMO DIA DISPONÍVEL")
    print("=" * 60)
    
    ultimo = all_scores[-1]
    ultima_data = df_results.iloc[-1]["data"].strftime("%Y-%m-%d")
    
    print(f"  Data: {ultima_data}")
    print(f"  Anomaly Score: {ultimo['anomaly_score']}")
    print(f"  Nível: {ultimo['nivel']}")
    print(f"  Erro bruto: {ultimo['raw_error']:.6f}")
    print(f"\n  Top contributors:")
    for contrib in ultimo["top_contributors"][:3]:
        print(f"    • {contrib['feature']}: z={contrib['z_score']:.2f} "
              f"(contribuição: {contrib['contribution_pct']:.1f}%)")
    print(f"\n  Explicação: {ultimo['explicacao']}")

    # --- 10. Salvar resultados ---
    output_path = ROOT / "outputs" / "reports" / "produto3_resultados.csv"
    cols_save = ["data", "anomaly_score", "anomaly_nivel", "raw_error",
                 "top_contributor_1", "top_contributor_1_z", "explicacao"]
    df_results[cols_save].to_csv(output_path, index=False)
    print(f"\n\n💾 Resultados salvos em: {output_path}")

    # Salvar validação
    validation_path = ROOT / "outputs" / "reports" / "produto3_validacao.json"
    with open(validation_path, "w") as f:
        json.dump({
            "eventos_validados": validacao_resultados,
            "taxa_deteccao": detectados / total_eventos if total_eventos > 0 else 0,
            "estatisticas": {
                "score_medio": float(df_results["anomaly_score"].mean()),
                "n_atipicos": int(nivel_counts.get("ATIPICO", 0)),
                "n_criticos": int(nivel_counts.get("CRITICO", 0)),
            },
            "gerado_em": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print(f"💾 Validação salva em: {validation_path}")
    print("\n✅ Produto 3 completo!")


if __name__ == "__main__":
    run_anomaly_detection()

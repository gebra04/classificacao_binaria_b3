"""
Script de treino do Autoencoder backbone.

Executa:
1. Feature Selection Pipeline (se features_curadas_v2.csv não existir)
2. Treinamento do DVAE
3. Diagnósticos (reconstruction error, latent space visualization)
4. Salva modelo, scaler, e métricas
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo
import matplotlib.pyplot as plt
from pathlib import Path

# Adicionar raiz ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.data_loader import (
    load_features_data,
    prepare_autoencoder_data,
    get_feature_columns,
    DATA_DIR,
)
from src.data.feature_selection import run_full_pipeline
from src.models.autoencoder import DVAE, DVAETrainer


OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"


def step_1_feature_selection():
    """Executa feature selection se necessário."""
    curated_path = DATA_DIR / "features_curadas_v2.csv"

    if curated_path.exists():
        print(f"✅ Features curadas já existem: {curated_path}")
        print("   Para re-executar, delete o arquivo e rode novamente.\n")
        return

    print("🔍 Executando Feature Selection Pipeline...\n")
    df, features, report = run_full_pipeline(source="features", verbose=True)

    # Salvar relatório
    report_path = REPORTS_DIR / "feature_selection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Converter para serializável
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(list(obj))
        return obj

    with open(report_path, "w") as f:
        json.dump(make_serializable(report), f, indent=2, ensure_ascii=False)
    print(f"\n📄 Relatório salvo em: {report_path}")


def step_2_train_autoencoder(
    latent_dim: int = 12,
    beta: float = 0.1,
    epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Treina o DVAE backbone."""
    print("\n🧠 Treinando DVAE Backbone...\n")

    # Carregar dados
    data = prepare_autoencoder_data(source="features")
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_holdout = data["X_holdout"]
    scaler = data["scaler"]
    feature_cols = data["feature_cols"]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Treino: {X_train.shape[0]} amostras")
    print(f"  Validação: {X_val.shape[0]} amostras")
    print(f"  Holdout: {X_holdout.shape[0]} amostras")
    print(f"  Latent dim: {latent_dim}")
    print(f"  β: {beta}\n")

    # Criar modelo
    model = DVAE(
        n_features=len(feature_cols),
        latent_dim=latent_dim,
        beta=beta,
    )
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    # Treinar
    trainer = DVAETrainer(model, lr=lr, patience=20, min_epochs=50)
    result = trainer.train(X_train, X_val, epochs=epochs, batch_size=batch_size)

    print(f"\n  ✅ Treino completo: {result['epochs_trained']} epochs")
    print(f"  Best val loss: {result['best_val_loss']:.6f}")

    # Salvar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "dvae_backbone.pt"
    trainer.save(model_path)

    # Salvar scaler e feature_cols
    import pickle
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    return trainer, data


def step_3_diagnostics(trainer: DVAETrainer, data: dict):
    """Gera visualizações de diagnóstico."""
    import torch

    print("\n📊 Gerando diagnósticos...\n")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model = trainer.model
    model.eval()

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_holdout = data["X_holdout"]
    feature_cols = data["feature_cols"]
    splits = data["splits"]

    # --- 1. Learning curves ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(trainer.history["train_loss"], label="Train", alpha=0.7)
    axes[0].plot(trainer.history["val_loss"], label="Val", alpha=0.7)
    axes[0].set_title("Loss Total")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(trainer.history["recon_loss"], label="Recon", color="green")
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(trainer.history["kl_loss"], label="KL", color="red")
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "dvae_learning_curves.png", dpi=150)
    plt.close()
    print("  ✅ Learning curves salvas")

    # --- 2. Reconstruction error distribution ---
    with torch.no_grad():
        err_train = model.get_reconstruction_error(
            torch.tensor(X_train, dtype=torch.float32)
        ).numpy()
        err_val = model.get_reconstruction_error(
            torch.tensor(X_val, dtype=torch.float32)
        ).numpy()
        err_holdout = model.get_reconstruction_error(
            torch.tensor(X_holdout, dtype=torch.float32)
        ).numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(err_train, bins=50, alpha=0.5, label=f"Treino (μ={err_train.mean():.4f})", density=True)
    ax.hist(err_val, bins=50, alpha=0.5, label=f"Val (μ={err_val.mean():.4f})", density=True)
    ax.hist(err_holdout, bins=50, alpha=0.5, label=f"Holdout (μ={err_holdout.mean():.4f})", density=True)
    ax.set_title("Distribuição do Erro de Reconstrução")
    ax.set_xlabel("MSE")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "dvae_recon_error_dist.png", dpi=150)
    plt.close()
    print("  ✅ Reconstruction error distribution salva")

    # --- 3. Per-feature reconstruction error ---
    with torch.no_grad():
        per_feat_err = model.get_reconstruction_error(
            torch.tensor(X_val, dtype=torch.float32), per_feature=True
        ).mean(dim=0).numpy()

    fig, ax = plt.subplots(figsize=(14, 5))
    sorted_idx = np.argsort(per_feat_err)[::-1]
    top_n = min(30, len(feature_cols))
    ax.barh(
        range(top_n),
        per_feat_err[sorted_idx[:top_n]],
        color="steelblue",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx[:top_n]], fontsize=8)
    ax.set_xlabel("MSE por Feature")
    ax.set_title("Erro de Reconstrução por Feature (Val)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "dvae_per_feature_error.png", dpi=150)
    plt.close()
    print("  ✅ Per-feature error salva")

    # --- 4. Latent space visualization (UMAP) ---
    try:
        import umap

        # Juntar todos os dados com labels de período
        X_all = np.vstack([X_train, X_val, X_holdout])
        with torch.no_grad():
            latent_all = model.get_latent(
                torch.tensor(X_all, dtype=torch.float32)
            ).numpy()

        labels = (
            ["Treino"] * len(X_train)
            + ["Val"] * len(X_val)
            + ["Holdout"] * len(X_holdout)
        )

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(latent_all)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {"Treino": "#4a9eff", "Val": "#22c55e", "Holdout": "#f59e0b"}
        for label in ["Treino", "Val", "Holdout"]:
            mask = np.array(labels) == label
            ax.scatter(
                embedding[mask, 0], embedding[mask, 1],
                c=colors[label], label=label, alpha=0.5, s=10,
            )
        ax.set_title("UMAP do Espaço Latente do DVAE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "dvae_latent_umap.png", dpi=150)
        plt.close()
        print("  ✅ UMAP do latent space salva")

        # --- 5. Latent space colorido por regime ---
        regimes_train = splits["treino"]["regime_label"].values
        regimes_val = splits["validacao"]["regime_label"].values
        regimes_holdout = splits["holdout_dev"]["regime_label"].values
        regimes_all = np.concatenate([regimes_train, regimes_val, regimes_holdout])

        fig, ax = plt.subplots(figsize=(10, 8))
        regime_colors = {
            "bull_forte": "#22c55e",
            "bull_fraco": "#86efac",
            "bear_fraco": "#fca5a5",
            "bear_forte": "#ef4444",
        }
        for regime, color in regime_colors.items():
            mask = regimes_all == regime
            if mask.sum() > 0:
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    c=color, label=regime, alpha=0.5, s=10,
                )
        ax.set_title("UMAP do Espaço Latente — Colorido por Regime")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "dvae_latent_regimes.png", dpi=150)
        plt.close()
        print("  ✅ UMAP por regime salva")

    except ImportError:
        print("  ⚠️ umap-learn não disponível, pulando visualização de latent space")

    # --- 6. Métricas resumo ---
    ratio = err_holdout.mean() / err_train.mean()
    metrics = {
        "recon_mse_train": float(err_train.mean()),
        "recon_mse_val": float(err_val.mean()),
        "recon_mse_holdout": float(err_holdout.mean()),
        "holdout_train_ratio": float(ratio),
        "kl_final": float(trainer.history["kl_loss"][-1]),
        "epochs_trained": len(trainer.history["train_loss"]),
        "status": "OK" if ratio < 2.0 else "ATENÇÃO: overfitting possível",
    }

    with open(REPORTS_DIR / "dvae_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n📈 Métricas:")
    print(f"  MSE Treino:  {metrics['recon_mse_train']:.6f}")
    print(f"  MSE Val:     {metrics['recon_mse_val']:.6f}")
    print(f"  MSE Holdout: {metrics['recon_mse_holdout']:.6f}")
    print(f"  Ratio H/T:   {metrics['holdout_train_ratio']:.2f}x")
    print(f"  KL final:    {metrics['kl_final']:.6f}")
    print(f"  Status:      {metrics['status']}")


def main():
    # Passo 1: Feature Selection
    step_1_feature_selection()

    # Passo 2: Treinar DVAE
    trainer, data = step_2_train_autoencoder(
        latent_dim=12,
        beta=0.1,
        epochs=500,
        batch_size=64,
        lr=1e-3,
    )

    # Passo 3: Diagnósticos
    step_3_diagnostics(trainer, data)

    print("\n✅ Pipeline completo!")
    print(f"  Modelo: {MODELS_DIR / 'dvae_backbone.pt'}")
    print(f"  Reports: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()

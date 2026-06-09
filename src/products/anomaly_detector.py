"""
Produto 3 — Detector de Dia Atípico / Anomaly Score.

Usa o erro de reconstrução do DVAE para gerar:
1. Score de anomalia 0-1 (normalizado via CDF empírica)
2. Threshold dinâmico (Q90/Q99 rolling)
3. Interpretabilidade por feature (contribution score)
4. Texto explicativo automático

"anomaly detection é tarefa para a qual autoencoder é literalmente desenhado"
"""

import numpy as np
import torch
from scipy import stats
from pathlib import Path


class AnomalyDetector:
    """
    Detector de dias atípicos via reconstruction error do DVAE.
    """

    def __init__(self, dvae_model, scaler, feature_cols: list[str]):
        self.dvae = dvae_model
        self.scaler = scaler
        self.feature_cols = feature_cols

        # Calibração (definida no fit)
        self.train_errors = None
        self.train_per_feature_errors = None
        self.per_feature_mean = None
        self.per_feature_std = None

    def fit(self, X_train_scaled: np.ndarray):
        """
        Calibra o detector usando dados de treino.
        
        Calcula distribuição de referência do erro de reconstrução.
        """
        self.dvae.eval()
        X_t = torch.tensor(X_train_scaled, dtype=torch.float32)

        with torch.no_grad():
            # Erro global por amostra
            self.train_errors = self.dvae.get_reconstruction_error(X_t).numpy()

            # Erro por feature
            self.train_per_feature_errors = self.dvae.get_reconstruction_error(
                X_t, per_feature=True
            ).numpy()

        # Estatísticas por feature para contribution score
        self.per_feature_mean = self.train_per_feature_errors.mean(axis=0)
        self.per_feature_std = self.train_per_feature_errors.std(axis=0) + 1e-9

        print(f"  Calibração: {len(self.train_errors)} amostras")
        print(f"  Erro médio treino: {self.train_errors.mean():.6f}")
        print(f"  Q90: {np.quantile(self.train_errors, 0.90):.6f}")
        print(f"  Q99: {np.quantile(self.train_errors, 0.99):.6f}")

    def score(self, X_scaled: np.ndarray) -> list[dict]:
        """
        Calcula anomaly score para um ou mais dias.
        
        Returns:
            lista de dicts com score, nível, contributors, explicação
        """
        if self.train_errors is None:
            raise ValueError("Detector não calibrado. Execute fit() primeiro.")

        self.dvae.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            errors = self.dvae.get_reconstruction_error(X_t).numpy()
            per_feature_errors = self.dvae.get_reconstruction_error(
                X_t, per_feature=True
            ).numpy()

        results = []
        for i in range(len(X_scaled)):
            # Score normalizado via CDF empírica (percentile rank)
            raw_error = errors[i]
            anomaly_score = float(
                stats.percentileofscore(self.train_errors, raw_error) / 100.0
            )

            # Nível
            if anomaly_score >= 0.99:
                nivel = "CRITICO"
            elif anomaly_score >= 0.90:
                nivel = "ATIPICO"
            elif anomaly_score >= 0.75:
                nivel = "ATENCAO"
            else:
                nivel = "NORMAL"

            # Contribution score por feature (z-score do erro, clipped for interpretability)
            feat_errors = per_feature_errors[i]
            contributions = (feat_errors - self.per_feature_mean) / self.per_feature_std
            contributions = np.clip(contributions, -10.0, 10.0)  # Prevent absurd z-scores from data drift

            # Top contributors (features mais anômalas)
            top_idx = np.argsort(contributions)[::-1][:5]
            top_contributors = []
            total_contribution = np.abs(contributions).sum()

            for idx in top_idx:
                if contributions[idx] > 1.0:  # Só features significativamente anômalas
                    feat_name = self.feature_cols[idx]
                    feat_value = float(X_scaled[i, idx])

                    # Desnormalizar valor para contexto
                    original_mean = float(self.scaler.mean_[idx]) if hasattr(self.scaler, 'mean_') else 0
                    original_std = float(self.scaler.scale_[idx]) if hasattr(self.scaler, 'scale_') else 1
                    original_value = feat_value * original_std + original_mean

                    top_contributors.append({
                        "feature": feat_name,
                        "contribution_pct": float(
                            np.abs(contributions[idx]) / total_contribution * 100
                        ),
                        "z_score": float(contributions[idx]),
                        "valor_normalizado": feat_value,
                        "valor_original": original_value,
                    })

            # Gerar explicação textual
            explicacao = self._gerar_explicacao(top_contributors)

            results.append({
                "anomaly_score": round(anomaly_score, 4),
                "raw_error": float(raw_error),
                "nivel": nivel,
                "top_contributors": top_contributors,
                "explicacao": explicacao,
            })

        return results[0] if len(results) == 1 else results

    def _gerar_explicacao(self, contributors: list[dict]) -> str:
        """Gera texto explicativo da anomalia."""
        if not contributors:
            return "Dia dentro do padrão histórico."

        # Mapeamento de features para nomes legíveis
        feature_names = {
            "vix_change_pct": "VIX",
            "sp500_change_pct": "SP500",
            "spx_slope_20d": "tendência do SP500 (20d)",
            "spx_drawdown_60": "drawdown do SP500",
            "dxy_slope_5d": "DXY (5d)",
            "dxy_slope_20d": "DXY (20d)",
            "us_yield_spread": "spread de juros EUA",
            "us10y_slope_5d": "US10Y (5d)",
            "usd_brl_slope_5d": "USD/BRL (5d)",
            "di1_close": "DI1",
            "brent_ret_d1": "Petróleo Brent",
            "ewz_ret_d1": "EWZ",
            "regime_juros_altos": "regime de juros altos",
        }

        partes = []
        for contrib in contributors[:3]:
            feat = contrib["feature"]
            z = contrib["z_score"]
            nome = feature_names.get(feat, feat)
            direcao = "subiu" if z > 0 else "caiu"
            intensidade = "muito" if abs(z) > 2.5 else "significativamente"
            partes.append(f"{nome} {direcao} {intensidade} (z={z:.1f})")

        return ". ".join(partes) + "."

    def score_with_history(
        self, X_scaled_history: np.ndarray, window: int = 60
    ) -> dict:
        """
        Score com threshold dinâmico baseado no histórico recente.
        
        Args:
            X_scaled_history: últimos N dias normalizados (último = dia atual)
            window: janela para threshold dinâmico
        """
        # Score de todos os dias da janela
        all_scores = self.score(X_scaled_history)
        if isinstance(all_scores, dict):
            all_scores = [all_scores]

        current = all_scores[-1]

        # Threshold dinâmico (Q90 da janela recente)
        recent_scores = [s["anomaly_score"] for s in all_scores[-window:]]
        dynamic_q90 = np.quantile(recent_scores, 0.90)
        dynamic_q99 = np.quantile(recent_scores, 0.99)

        current["threshold_dinamico_q90"] = float(dynamic_q90)
        current["threshold_dinamico_q99"] = float(dynamic_q99)
        current["acima_threshold_dinamico"] = current["anomaly_score"] > dynamic_q90

        return current

    def save(self, path: str | Path):
        """Salva calibração do detector."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "train_errors": self.train_errors,
            "per_feature_mean": self.per_feature_mean,
            "per_feature_std": self.per_feature_std,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path):
        """Carrega calibração."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.train_errors = data["train_errors"]
        self.per_feature_mean = data["per_feature_mean"]
        self.per_feature_std = data["per_feature_std"]

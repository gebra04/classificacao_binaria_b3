"""
Produto 4 — Sistema de Probabilidades Condicionais.

Dado o contexto de D-1 (features + latente do DVAE + regime do Produto 2),
gera distribuição de probabilidade do retorno de D.

Output:
- P(ret em cada faixa)
- Quantis (Q05, Q25, Q50, Q75, Q95)
- Cenário modal
- Métricas de calibração (ECE)
"""

import numpy as np
import torch
from scipy import stats as scipy_stats
from pathlib import Path


class ConditionalProbSystem:
    """
    Sistema de probabilidades condicionais via MDN.
    """

    def __init__(self, mdn_model, dvae_model, scaler, feature_cols: list[str]):
        self.mdn = mdn_model
        self.dvae = dvae_model
        self.scaler = scaler
        self.feature_cols = feature_cols

        # Bins de retorno para as probabilidades
        self.return_bins = [
            (-np.inf, -0.015, "WIN < -1.5%"),
            (-0.015, -0.005, "WIN entre -1.5% e -0.5%"),
            (-0.005, 0.005, "WIN entre -0.5% e +0.5%"),
            (0.005, 0.015, "WIN entre +0.5% e +1.5%"),
            (0.015, np.inf, "WIN > +1.5%"),
        ]

        self.quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]

    def predict(
        self, X_scaled: np.ndarray, regime_info: dict = None
    ) -> list[dict]:
        """
        Gera probabilidades condicionais.
        
        Args:
            X_scaled: features normalizadas (pode incluir latente + regime já concatenados)
            regime_info: dict do Produto 2 (opcional, para contexto)
        
        Returns:
            lista de dicts com distribuição, quantis, cenário
        """
        self.mdn.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            pi, mu, sigma = self.mdn(X_t)

        pi = pi.numpy()
        mu = mu.numpy()
        sigma = sigma.numpy()

        results = []
        for i in range(len(X_scaled)):
            # Calcular probabilidades por faixa via integração da mistura
            distribuicao = {}
            for low, high, label in self.return_bins:
                prob = self._mixture_cdf_range(
                    pi[i], mu[i], sigma[i], low, high
                )
                distribuicao[label] = round(float(prob), 4)

            # Calcular quantis via inversão numérica
            quantis = {}
            for q in self.quantile_levels:
                quantis[f"Q{int(q * 100):02d}"] = round(
                    float(self._mixture_quantile(pi[i], mu[i], sigma[i], q)), 6
                )

            # Cenário modal (faixa mais provável)
            cenario_modal = max(distribuicao, key=distribuicao.get)
            prob_modal = distribuicao[cenario_modal]

            # Estatísticas da mistura
            mixture_mean = float(np.sum(pi[i] * mu[i]))
            mixture_var = float(
                np.sum(pi[i] * (sigma[i] ** 2 + mu[i] ** 2)) - mixture_mean ** 2
            )
            mixture_std = float(np.sqrt(max(mixture_var, 1e-10)))

            result = {
                "distribuicao": distribuicao,
                "quantis": quantis,
                "cenario_modal": cenario_modal,
                "prob_cenario_modal": prob_modal,
                "media_esperada": round(mixture_mean, 6),
                "volatilidade_esperada": round(mixture_std, 6),
                "componentes": {
                    "pi": pi[i].tolist(),
                    "mu": mu[i].tolist(),
                    "sigma": sigma[i].tolist(),
                },
            }

            if regime_info:
                result["regime_contexto"] = regime_info.get("regime_nome", "N/A")

            results.append(result)

        return results[0] if len(results) == 1 else results

    def _mixture_cdf_range(self, pi, mu, sigma, low, high):
        """CDF da mistura de gaussianas no intervalo [low, high]."""
        prob = 0.0
        for k in range(len(pi)):
            cdf_high = scipy_stats.norm.cdf(high, loc=mu[k], scale=sigma[k])
            cdf_low = scipy_stats.norm.cdf(low, loc=mu[k], scale=sigma[k])
            prob += pi[k] * (cdf_high - cdf_low)
        return prob

    def _mixture_quantile(self, pi, mu, sigma, q, n_points=1000):
        """Quantil da mistura via grid search."""
        # Range razoável para retornos diários
        x_grid = np.linspace(-0.10, 0.10, n_points)
        cdf_vals = np.zeros(n_points)

        for k in range(len(pi)):
            cdf_vals += pi[k] * scipy_stats.norm.cdf(x_grid, loc=mu[k], scale=sigma[k])

        # Encontrar o ponto onde CDF ≈ q
        idx = np.searchsorted(cdf_vals, q)
        idx = min(idx, n_points - 1)
        return x_grid[idx]

    def calibration_metrics(
        self, X_scaled: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> dict:
        """
        Calcula métricas de calibração.
        
        Args:
            X_scaled: features normalizadas
            y_true: retornos realizados
        
        Returns:
            dict com ECE, Brier, cobertura empírica
        """
        predictions = self.predict(X_scaled)
        if isinstance(predictions, dict):
            predictions = [predictions]

        # --- ECE (Expected Calibration Error) ---
        # Para cada bin de probabilidade prevista, comparar com frequência real
        all_probs = []
        all_realized = []
        for pred, actual in zip(predictions, y_true):
            for low, high, label in self.return_bins:
                prob = pred["distribuicao"][label]
                realized = 1.0 if low < actual <= high else 0.0
                all_probs.append(prob)
                all_realized.append(realized)

        all_probs = np.array(all_probs)
        all_realized = np.array(all_realized)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (all_probs >= bin_edges[b]) & (all_probs < bin_edges[b + 1])
            if mask.sum() > 0:
                avg_prob = all_probs[mask].mean()
                avg_real = all_realized[mask].mean()
                ece += mask.sum() / len(all_probs) * abs(avg_prob - avg_real)

        # --- Cobertura empírica dos intervalos de confiança ---
        coverage = {}
        for q_low, q_high, target_cov in [(0.05, 0.95, 0.90), (0.25, 0.75, 0.50)]:
            in_interval = 0
            for pred, actual in zip(predictions, y_true):
                q_l = pred["quantis"][f"Q{int(q_low * 100):02d}"]
                q_h = pred["quantis"][f"Q{int(q_high * 100):02d}"]
                if q_l <= actual <= q_h:
                    in_interval += 1
            empirical = in_interval / len(y_true) if len(y_true) > 0 else 0
            coverage[f"[Q{int(q_low*100):02d}-Q{int(q_high*100):02d}]"] = {
                "target": target_cov,
                "empirical": round(empirical, 4),
                "gap": round(abs(empirical - target_cov), 4),
            }

        return {
            "ece": round(float(ece), 4),
            "coverage": coverage,
            "n_samples": len(y_true),
        }

    def prepare_mdn_input(
        self,
        X_features_scaled: np.ndarray,
        regime_onehot: np.ndarray = None,
    ) -> np.ndarray:
        """
        Prepara input completo para o MDN:
        features curadas + latente do DVAE + regime one-hot.
        """
        # Latente do DVAE
        self.dvae.eval()
        with torch.no_grad():
            latent = self.dvae.get_latent(
                torch.tensor(X_features_scaled, dtype=torch.float32)
            ).numpy()

        # Concatenar
        parts = [X_features_scaled, latent]
        if regime_onehot is not None:
            parts.append(regime_onehot)

        return np.hstack(parts).astype(np.float32)

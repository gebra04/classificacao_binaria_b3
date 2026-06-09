"""
Produto 2 — Classificador de Regime de Mercado.

Detecta (não prevê) o regime atual do mercado via:
1. Clustering GMM no espaço latente do DVAE
2. Nomeação automática dos clusters via análise de características
3. Matriz de transição de Markov → probabilidade de persistência

Output: regime label + confiança + duração + recomendação operacional
"""

import json
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from pathlib import Path


class RegimeClassifier:
    """
    Classificador de regimes de mercado via clustering no espaço latente do DVAE.
    """

    def __init__(self, dvae_model, scaler, feature_cols: list[str]):
        self.dvae = dvae_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.gmm = None
        self.n_regimes = None
        self.regime_names = {}
        self.regime_profiles = {}
        self.transition_matrix = None

    def fit(
        self,
        X_raw: np.ndarray,
        feature_df=None,
        k_range: tuple = (3, 7),
    ) -> dict:
        """
        Ajusta o classificador de regimes.
        
        Args:
            X_raw: features normalizadas (output do scaler)
            feature_df: DataFrame original com colunas de features (para nomeação)
            k_range: range de clusters a testar
        """
        # Extrair latentes
        self.dvae.eval()
        with torch.no_grad():
            latent = self.dvae.get_latent(
                torch.tensor(X_raw, dtype=torch.float32)
            ).numpy()

        # Testar diferentes k via BIC + silhouette
        results = {}
        for k in range(k_range[0], k_range[1] + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=5,
                random_state=42,
            )
            labels = gmm.fit_predict(latent)
            bic = gmm.bic(latent)
            sil = silhouette_score(latent, labels) if k > 1 else 0
            results[k] = {"bic": bic, "silhouette": sil, "gmm": gmm, "labels": labels}

        # Selecionar k: melhor silhouette com BIC razoável
        best_k = max(results, key=lambda k: results[k]["silhouette"])
        self.gmm = results[best_k]["gmm"]
        self.n_regimes = best_k
        best_labels = results[best_k]["labels"]

        print(f"  Melhor k={best_k} (silhouette={results[best_k]['silhouette']:.3f}, "
              f"BIC={results[best_k]['bic']:.0f})")

        # Nomear regimes automaticamente via perfil de features
        if feature_df is not None:
            self._name_regimes(feature_df, best_labels)

        # Calcular matriz de transição
        self._compute_transition_matrix(best_labels)

        return {
            "k": best_k,
            "silhouette": results[best_k]["silhouette"],
            "bic": results[best_k]["bic"],
            "all_results": {
                k: {"bic": v["bic"], "silhouette": v["silhouette"]}
                for k, v in results.items()
            },
            "labels": best_labels,
        }

    def _name_regimes(self, df, labels):
        """
        Nomeia regimes automaticamente baseado no perfil de features relevantes.
        
        Analisa médias de features-chave por cluster para definir nomes interpretativos.
        """
        # Features diagnósticas para nomeação
        diagnostic_features = {
            "vol": ["vol_20d", "ret_std_5d", "ret_std_10d", "atr_14"],
            "trend": ["spx_slope_20d", "spx_acima_ma50", "spx_acima_ma200"],
            "risk": ["vix_change_pct", "vix_spike", "spx_drawdown_60"],
            "macro": ["di1_close", "regime_juros_altos", "us_yield_spread"],
        }

        df_analysis = df.copy()
        df_analysis["_regime_label"] = labels

        for regime_id in range(self.n_regimes):
            mask = df_analysis["_regime_label"] == regime_id
            regime_data = df_analysis[mask]
            profile = {}

            for category, features in diagnostic_features.items():
                for feat in features:
                    if feat in df_analysis.columns:
                        profile[feat] = {
                            "mean": float(regime_data[feat].mean()),
                            "std": float(regime_data[feat].std()),
                        }

            # Heurística de nomeação
            name_parts = []

            # Tendência
            spx_slope = profile.get("spx_slope_20d", {}).get("mean", 0)
            if spx_slope > 0.02:
                name_parts.append("Trending-Up")
            elif spx_slope < -0.02:
                name_parts.append("Trending-Down")
            else:
                name_parts.append("Range")

            # Risco
            vix_mean = profile.get("vix_change_pct", {}).get("mean", 0)
            drawdown = profile.get("spx_drawdown_60", {}).get("mean", 0)
            if drawdown < -0.08 or vix_mean > 0.05:
                name_parts.append("Risk-Off")
            else:
                name_parts.append("Risk-On")

            # Volatilidade
            vol_keys = [k for k in profile if "vol" in k or "std" in k or "atr" in k]
            if vol_keys:
                avg_vol = np.mean([profile[k]["mean"] for k in vol_keys])
                # Normalizar por referência (aproximado)
                if avg_vol > 0.02:
                    name_parts.append("High-Vol")
                elif avg_vol < 0.008:
                    name_parts.append("Low-Vol")
                else:
                    name_parts.append("Mid-Vol")

            self.regime_names[regime_id] = " / ".join(name_parts)
            self.regime_profiles[regime_id] = {
                "name": self.regime_names[regime_id],
                "n_days": int(mask.sum()),
                "pct": float(mask.mean()),
                "profile": profile,
            }

        # Print resumo
        print(f"\n  Regimes identificados:")
        for rid, info in self.regime_profiles.items():
            print(f"    {rid}: {info['name']} ({info['n_days']} dias, {info['pct']:.1%})")

    def _compute_transition_matrix(self, labels):
        """Calcula matriz de transição de Markov entre regimes."""
        n = self.n_regimes
        trans = np.zeros((n, n))

        for i in range(len(labels) - 1):
            trans[labels[i], labels[i + 1]] += 1

        # Normalizar por linha
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Evitar divisão por zero
        self.transition_matrix = trans / row_sums

    def predict(self, X_raw: np.ndarray) -> dict:
        """
        Classifica um ou mais dias em regimes.
        
        Args:
            X_raw: features normalizadas (1 ou mais linhas)
        
        Returns:
            dict com regime, probabilidades, etc.
        """
        if self.gmm is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")

        # Extrair latente
        self.dvae.eval()
        with torch.no_grad():
            latent = self.dvae.get_latent(
                torch.tensor(X_raw, dtype=torch.float32)
            ).numpy()

        # Predizer regime
        labels = self.gmm.predict(latent)
        probas = self.gmm.predict_proba(latent)

        results = []
        for i in range(len(X_raw)):
            regime_id = int(labels[i])
            regime_name = self.regime_names.get(regime_id, f"Regime_{regime_id}")

            # Probabilidade de persistência
            persist_prob = float(self.transition_matrix[regime_id, regime_id])

            # Regimes vizinhos (ordenados por probabilidade de transição)
            trans_probs = self.transition_matrix[regime_id].copy()
            trans_probs[regime_id] = 0  # Excluir auto-transição
            vizinhos = {}
            for rid in np.argsort(trans_probs)[::-1][:3]:
                if trans_probs[rid] > 0.05:
                    vizinhos[self.regime_names.get(rid, f"Regime_{rid}")] = float(
                        trans_probs[rid]
                    )

            # Recomendação operacional
            recomendacao = self._gerar_recomendacao(regime_name)

            results.append({
                "regime_id": regime_id,
                "regime_nome": regime_name,
                "confianca": float(probas[i, regime_id]),
                "probabilidade_persistencia": persist_prob,
                "regimes_vizinhos": vizinhos,
                "recomendacao_operacional": recomendacao,
            })

        return results[0] if len(results) == 1 else results

    def _gerar_recomendacao(self, regime_name: str) -> str:
        """Gera recomendação operacional baseada no nome do regime."""
        name_lower = regime_name.lower()

        recs = []
        if "risk-off" in name_lower:
            recs.append("Evitar posições long agressivas")
        if "high-vol" in name_lower:
            recs.append("Stops mais largos, posição menor")
        if "trending-down" in name_lower:
            recs.append("Viés vendedor, proteções ativas")
        if "trending-up" in name_lower and "risk-on" in name_lower:
            recs.append("Ambiente favorável para long")
        if "range" in name_lower and "low-vol" in name_lower:
            recs.append("Mercado lateral, operar reversão à média")

        return ". ".join(recs) if recs else "Sem recomendação específica"

    def save(self, path: str | Path):
        """Salva classificador de regimes."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "gmm": self.gmm,
            "n_regimes": self.n_regimes,
            "regime_names": self.regime_names,
            "regime_profiles": self.regime_profiles,
            "transition_matrix": self.transition_matrix,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str | Path):
        """Carrega classificador de regimes."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.gmm = data["gmm"]
        self.n_regimes = data["n_regimes"]
        self.regime_names = data["regime_names"]
        self.regime_profiles = data["regime_profiles"]
        self.transition_matrix = data["transition_matrix"]

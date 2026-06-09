"""
Mixture Density Network (MDN) — Modelo para Produto 4.

Output: mistura de K gaussianas → distribuição condicional do retorno.
Captura distribuições multimodais (ex: dias pré-FOMC com bimodalidade).

Loss: Negative Log-Likelihood da mistura
"""

import torch
import torch.nn as nn
import numpy as np


class MDN(nn.Module):
    """
    Mixture Density Network.
    
    Output: K gaussianas com parâmetros (π, μ, σ) cada.
    
    Args:
        n_input: dimensão do input
        n_gaussians: número de componentes da mistura (K)
        hidden_dims: dimensões das camadas ocultas
        dropout: dropout rate
    """

    def __init__(
        self,
        n_input: int,
        n_gaussians: int = 3,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.n_input = n_input
        self.n_gaussians = n_gaussians

        # Backbone
        layers = []
        in_dim = n_input
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # MDN heads
        last_dim = hidden_dims[-1]
        self.pi_head = nn.Linear(last_dim, n_gaussians)   # mixing coefficients
        self.mu_head = nn.Linear(last_dim, n_gaussians)   # means
        self.sigma_head = nn.Linear(last_dim, n_gaussians) # std devs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            (pi, mu, sigma) — cada com shape (batch, K)
            pi: mixing coefficients (somam 1)
            mu: means
            sigma: standard deviations (positivos)
        """
        h = self.backbone(x)

        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma = torch.nn.functional.softplus(self.sigma_head(h)) + 1e-6

        return pi, mu, sigma


def mdn_loss(pi, mu, sigma, target):
    """
    Negative Log-Likelihood da mistura de gaussianas.
    
    Args:
        pi: (batch, K) mixing coefficients
        mu: (batch, K) means
        sigma: (batch, K) std devs
        target: (batch,) target values
    
    Returns:
        scalar loss
    """
    target = target.unsqueeze(1)  # (batch, 1)

    # Log-probabilidade de cada gaussiana
    # log N(x | μ, σ) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
    log_normal = (
        -0.5 * np.log(2 * np.pi)
        - torch.log(sigma)
        - 0.5 * ((target - mu) / sigma) ** 2
    )

    # Log-sum-exp para estabilidade numérica
    log_pi = torch.log(pi + 1e-10)
    log_prob = torch.logsumexp(log_pi + log_normal, dim=1)

    return -log_prob.mean()


class MDNTrainer:
    """Treinador do MDN com early stopping."""

    def __init__(
        self,
        model: MDN,
        lr: float = 1e-3,
        patience: int = 20,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.patience = patience
        self.history = {"train_loss": [], "val_loss": []}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 500,
        batch_size: int = 64,
    ) -> dict:
        """Treina o MDN."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for X_batch, y_batch in loader:
                pi, mu, sigma = self.model(X_batch)
                loss = mdn_loss(pi, mu, sigma, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = np.mean(epoch_losses)

            self.model.eval()
            with torch.no_grad():
                pi_v, mu_v, sigma_v = self.model(X_val_t)
                val_loss = mdn_loss(pi_v, mu_v, sigma_v, y_val_t).item()

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch >= 50 and patience_counter >= self.patience:
                print(f"  Early stopping na epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    def save(self, path):
        """Salva modelo MDN."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "n_input": self.model.n_input,
                "n_gaussians": self.model.n_gaussians,
            },
            "history": self.history,
        }, path)


from pathlib import Path

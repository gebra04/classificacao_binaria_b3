"""
Denoising Variational Autoencoder (DVAE) — Backbone compartilhado.

Combina:
- VAE: espaço latente regularizado (ideal para clustering do Produto 2)
- Denoising: robustez a ruído (melhora detecção de anomalias do Produto 3)

Alimenta os 3 produtos:
- Produto 2: vetores latentes μ → clustering GMM → regimes
- Produto 3: erro de reconstrução por feature → anomaly score
- Produto 4: features latentes enriquecidas → input para MDN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class DVAE(nn.Module):
    """
    Denoising Variational Autoencoder.
    
    Arquitetura:
        Input (n_features) → Dropout ruído → Encoder → μ, log_σ² → z → Decoder → Reconstrução
    
    Args:
        n_features: número de features de entrada
        latent_dim: dimensão do espaço latente (default: 12)
        hidden_dims: dimensões das camadas ocultas (default: [64, 32])
        noise_dropout: probabilidade de dropout de ruído no input (denoising)
        dropout: dropout nas camadas ocultas
        beta: peso da KL divergence (β-VAE, default: 0.1)
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 12,
        hidden_dims: list[int] = None,
        noise_dropout: float = 0.1,
        dropout: float = 0.2,
        beta: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.n_features = n_features
        self.latent_dim = latent_dim
        self.beta = beta

        # Denoising: dropout no input
        self.noise = nn.Dropout(p=noise_dropout)

        # --- Encoder ---
        encoder_layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Cabeças para μ e log_σ²
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # --- Decoder ---
        decoder_layers = []
        decoder_dims = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h_dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(decoder_dims[-1], n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna μ e log_σ² do espaço latente."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε·σ"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Em inferência, usar μ diretamente (determinístico)
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstrói a entrada a partir do latente."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo.
        
        Returns:
            (reconstrução, μ, log_σ²)
        """
        # Denoising: corromper input durante treino
        x_noisy = self.noise(x)
        mu, logvar = self.encode(x_noisy)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai vetor latente μ (sem ruído, determinístico)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def get_reconstruction_error(
        self, x: torch.Tensor, per_feature: bool = False
    ) -> torch.Tensor:
        """
        Calcula erro de reconstrução.
        
        Args:
            x: input tensor
            per_feature: se True, retorna erro por feature; se False, média
        
        Returns:
            tensor de erros
        """
        self.eval()
        with torch.no_grad():
            recon, _, _ = self(x)
            error = (recon - x) ** 2
            if per_feature:
                return error
            return error.mean(dim=1)


def dvae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss do DVAE = MSE_reconstruction + β * KL_divergence.
    
    Returns:
        (loss_total, loss_recon, loss_kl)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")

    # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


class DVAETrainer:
    """
    Treinador do DVAE com early stopping e logging.
    """

    def __init__(
        self,
        model: DVAE,
        lr: float = 1e-3,
        patience: int = 20,
        min_epochs: int = 50,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.patience = patience
        self.min_epochs = min_epochs
        self.history = {"train_loss": [], "val_loss": [], "kl_loss": [], "recon_loss": []}

    def train(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        epochs: int = 500,
        batch_size: int = 64,
    ) -> dict:
        """
        Treina o DVAE.
        
        Returns:
            dict com histórico de treino e melhor modelo
        """
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        dataset = torch.utils.data.TensorDataset(X_train_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            epoch_losses = []
            for (batch,) in loader:
                recon, mu, logvar = self.model(batch)
                loss, recon_loss, kl_loss = dvae_loss(
                    recon, batch, mu, logvar, beta=self.model.beta
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = np.mean(epoch_losses)

            # --- Validation ---
            self.model.eval()
            with torch.no_grad():
                val_recon, val_mu, val_logvar = self.model(X_val_t)
                val_loss, val_recon_l, val_kl_l = dvae_loss(
                    val_recon, X_val_t, val_mu, val_logvar,
                    beta=self.model.beta,
                )
                val_loss = val_loss.item()

            self.scheduler.step(val_loss)

            # Logging
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["recon_loss"].append(val_recon_l.item())
            self.history["kl_loss"].append(val_kl_l.item())

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch >= self.min_epochs and patience_counter >= self.patience:
                print(f"Early stopping na epoch {epoch + 1}")
                break

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"Train: {train_loss:.6f} | "
                    f"Val: {val_loss:.6f} | "
                    f"Recon: {val_recon_l.item():.6f} | "
                    f"KL: {val_kl_l.item():.6f}"
                )

        # Restaurar melhor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "history": self.history,
        }

    def save(self, path: str | Path):
        """Salva modelo e config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": {
                    "n_features": self.model.n_features,
                    "latent_dim": self.model.latent_dim,
                    "beta": self.model.beta,
                },
                "history": self.history,
            },
            path,
        )
        print(f"Modelo salvo em: {path}")

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> "DVAETrainer":
        """Carrega modelo salvo."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = DVAE(
            n_features=config["n_features"],
            latent_dim=config["latent_dim"],
            beta=config["beta"],
        )
        model.load_state_dict(checkpoint["model_state"])
        trainer = DVAETrainer(model, device=device)
        trainer.history = checkpoint.get("history", {})
        return trainer

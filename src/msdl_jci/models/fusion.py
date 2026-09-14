"""Multi-Source Deep Learning Models and Adaptive Soft Gating Fusion.

Implements the exact model architectures specified in Table 5 and Section 2.4:
1. TechnicalLSTMBranch: Hidden Size 64, Lookback 28, Dropout 0.2
2. MacroMLPBranch: 2 Layers (3 -> 32 -> 16), ReLU
3. NewsProjectionBranch: Linear Layer (768 -> 64), LayerNorm, ReLU
4. SoftGatingNetwork: Input 144 -> 64 -> 3 Softmax weights (alpha, beta, gamma)
5. AdaptiveSoftGatingFusionModel: Full proposed architecture with dynamic gating
6. Baseline Models for Ablation Study:
   - PureLSTMModel (Technical only)
   - LSTMMacroModel (Technical + Macro)
   - LSTMNewsModel (Technical + News)
   - StaticFusionModel (Technical + Macro + News without Soft Gating)
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TechnicalLSTMBranch(nn.Module):
    """LSTM feature extractor for sequential technical price indicators.

    Table 5: Hidden Size 64, Lookback Window 28, Dropout 0.2
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, seq_len=28, input_dim=7]

        Returns:
            h_tech: Tensor of shape [batch_size, 64]
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # [batch_size, 64]
        out = self.layer_norm(last_hidden)
        out = self.dropout(out)
        return out


class MacroMLPBranch(nn.Module):
    """Multi-Layer Perceptron encoder for macroeconomic variables.

    Table 5: 2 Layers (3 -> 32 -> 16), ReLU -> 16-dimensional embedding

    Audit Phase 3: BatchNorm1d replaced with LayerNorm for stability with
    small/uneven batch sizes (BatchNorm's running stats behave erratically
    in eval with tiny batches and couple train/eval outputs).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim1: int = 32,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, 3] (BI-Rate, Inflation, USD/IDR)

        Returns:
            h_macro: Tensor of shape [batch_size, 16]
        """
        return self.net(x)


class NewsProjectionBranch(nn.Module):
    """Linear projection layer for Frozen IndoBERT 768-dim embeddings.

    Table 5: Linear Layer (768 -> 64)
    """

    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, 768]

        Returns:
            h_news: Tensor of shape [batch_size, 64]
        """
        return self.proj(x)


class SoftGatingNetwork(nn.Module):
    """Adaptive Soft Gating Network (Dynamic Weight Generator).

    Table 5:
    - Input: 144 (64 [Tech] + 16 [Macro] + 64 [News])
    - Output: 3 dynamic weights (alpha, beta, gamma) via Softmax
    """

    def __init__(
        self,
        input_dim: int = 144,
        hidden_dim: int = 64,
        num_experts: int = 3,
        temperature: float = 1.0,
        min_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )
        # Balanced init: near-zero final bias/weights => ~uniform [1/3,1/3,1/3].
        last_layer = self.gating[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.bias)
            nn.init.xavier_uniform_(last_layer.weight, gain=0.1)

    def forward(self, h_concat: torch.Tensor) -> torch.Tensor:
        """Compute dynamic weighting coefficients.

        Args:
            h_concat: Tensor of shape [batch_size, 144]

        Returns:
            weights: Tensor of shape [batch_size, 3] where weights sum to 1.0 (Softmax)
        """
        logits = self.gating(h_concat)
        weights = F.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        if self.min_weight > 0.0:
            # Floor each branch weight: w_eff = eps + (1 - K*eps) * w.
            k = weights.shape[-1]
            eps = min(self.min_weight, 0.99 / k)
            weights = eps + (1.0 - k * eps) * weights
        return weights

    def gating_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Mean Shannon entropy of gating weights (higher = more balanced)."""
        return -(weights * (weights.clamp_min(1e-9)).log()).sum(-1).mean()


class AdaptiveSoftGatingFusionModel(nn.Module):
    """Proposed Multi-Source Deep Learning Model with Adaptive Soft Gating.

    Fuses Technical LSTM (64-dim), Macro MLP (16-dim), and IndoBERT (64-dim)
    using dynamic Soft Gating weights [alpha, beta, gamma] to predict t+5 direction.

    Audit Phase 3 stabilizers:
    - ``pos_rate`` initializes the final classifier bias to log(p/(1-p)) so the
      model starts calibrated instead of collapsing to the majority class.
    - ``modality_dropout`` randomly zeroes whole branches during training.
    - Branch LayerNorm before gating prevents scale dominance.
    - ``temperature`` softens gating softmax; ``min_weight`` floors each branch.
    - Optional auxiliary per-branch heads (``aux_loss``) for deep supervision.
    """

    def __init__(
        self,
        tech_input_dim: int = 7,
        macro_input_dim: int = 3,
        news_input_dim: int = 768,
        tech_hidden_dim: int = 64,
        macro_latent_dim: int = 16,
        news_proj_dim: int = 64,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.2,
        pos_rate: float | None = None,
        modality_dropout: float = 0.0,
        temperature: float = 1.0,
        min_weight: float = 0.0,
        aux_loss: bool = False,
    ) -> None:
        super().__init__()
        self.tech_branch = TechnicalLSTMBranch(
            input_dim=tech_input_dim,
            hidden_dim=tech_hidden_dim,
            dropout=dropout,
        )
        self.macro_branch = MacroMLPBranch(
            input_dim=macro_input_dim,
            latent_dim=macro_latent_dim,
        )
        self.news_branch = NewsProjectionBranch(
            input_dim=news_input_dim,
            proj_dim=news_proj_dim,
        )

        total_dim = tech_hidden_dim + macro_latent_dim + news_proj_dim  # 144
        self.soft_gating = SoftGatingNetwork(
            input_dim=total_dim, num_experts=3,
            temperature=temperature, min_weight=min_weight,
        )

        # Per-branch normalization before gating (prevent scale dominance).
        self.norm_tech = nn.LayerNorm(tech_hidden_dim)
        self.norm_macro = nn.LayerNorm(macro_latent_dim)
        self.norm_news = nn.LayerNorm(news_proj_dim)
        self.modality_dropout = float(modality_dropout)
        self.aux_loss = bool(aux_loss)
        if self.aux_loss:
            self.aux_tech = nn.Linear(tech_hidden_dim, 1)
            self.aux_macro = nn.Linear(macro_latent_dim, 1)
            self.aux_news = nn.Linear(news_proj_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )
        # Calibrated init: bias = logit(pos_rate) so initial p ≈ base rate.
        if pos_rate is not None and 0.0 < pos_rate < 1.0:
            import math
            out_layer = self.classifier[-1]
            if isinstance(out_layer, nn.Linear):
                with torch.no_grad():
                    out_layer.bias.fill_(math.log(pos_rate / (1.0 - pos_rate)))
                    nn.init.xavier_uniform_(out_layer.weight, gain=0.5)

    def _maybe_drop_modality(self, h: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0.0:
            return h
        if torch.rand(1, device=h.device).item() < self.modality_dropout:
            return torch.zeros_like(h)
        return h

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor,
        x_news: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dynamic soft gating.

        Args:
            x_tech: [batch_size, 28, 7]
            x_macro: [batch_size, 3]
            x_news: [batch_size, 768]

        Returns:
            logits: [batch_size, 1] raw prediction logits
            weights: [batch_size, 3] dynamic gating weights (alpha, beta, gamma)
        """
        h_tech = self.norm_tech(self._maybe_drop_modality(self.tech_branch(x_tech)))      # [B, 64]
        h_macro = self.norm_macro(self._maybe_drop_modality(self.macro_branch(x_macro)))    # [B, 16]
        h_news = self.norm_news(self._maybe_drop_modality(self.news_branch(x_news)))      # [B, 64]

        h_concat = torch.cat([h_tech, h_macro, h_news], dim=1)  # [B, 144]

        # Dynamic gating weights: alpha, beta, gamma
        weights = self.soft_gating(h_concat)  # [B, 3]
        alpha = weights[:, 0:1]               # [B, 1]
        beta = weights[:, 1:2]                # [B, 1]
        gamma = weights[:, 2:3]               # [B, 1]

        # Gated representation
        h_gated = torch.cat([alpha * h_tech, beta * h_macro, gamma * h_news], dim=1)  # [B, 144]

        logits = self.classifier(h_gated)  # [B, 1]
        return logits, weights

    def aux_logits(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor,
        x_news: torch.Tensor,
    ) -> torch.Tensor:
        """Per-branch auxiliary logits [B, 3] for deep supervision (train only)."""
        if not self.aux_loss:
            raise RuntimeError("aux_logits requires aux_loss=True")
        h_tech = self.norm_tech(self.tech_branch(x_tech))
        h_macro = self.norm_macro(self.macro_branch(x_macro))
        h_news = self.norm_news(self.news_branch(x_news))
        return torch.cat([self.aux_tech(h_tech), self.aux_macro(h_macro),
                          self.aux_news(h_news)], dim=1)


# ==============================================================================
# ABLATION STUDY BASELINE MODELS (Thesis Section 2.6)
# ==============================================================================

class PureLSTMModel(nn.Module):
    """Baseline 1: Pure LSTM on technical features only."""

    def __init__(
        self,
        tech_input_dim: int = 7,
        hidden_dim: int = 64,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tech_branch = TechnicalLSTMBranch(
            input_dim=tech_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor | None = None,
        x_news: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_tech = self.tech_branch(x_tech)
        logits = self.classifier(h_tech)
        return logits, None


class LSTMMacroModel(nn.Module):
    """Baseline 2: Technical LSTM + Macroeconomic MLP."""

    def __init__(
        self,
        tech_input_dim: int = 7,
        macro_input_dim: int = 3,
        tech_hidden_dim: int = 64,
        macro_latent_dim: int = 16,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tech_branch = TechnicalLSTMBranch(
            input_dim=tech_input_dim,
            hidden_dim=tech_hidden_dim,
            dropout=dropout,
        )
        self.macro_branch = MacroMLPBranch(
            input_dim=macro_input_dim,
            latent_dim=macro_latent_dim,
        )
        total_dim = tech_hidden_dim + macro_latent_dim  # 80
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor,
        x_news: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_tech = self.tech_branch(x_tech)
        h_macro = self.macro_branch(x_macro)
        h_concat = torch.cat([h_tech, h_macro], dim=1)
        logits = self.classifier(h_concat)
        return logits, None


class LSTMNewsModel(nn.Module):
    """Baseline 3: Technical LSTM + News IndoBERT."""

    def __init__(
        self,
        tech_input_dim: int = 7,
        news_input_dim: int = 768,
        tech_hidden_dim: int = 64,
        news_proj_dim: int = 64,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tech_branch = TechnicalLSTMBranch(
            input_dim=tech_input_dim,
            hidden_dim=tech_hidden_dim,
            dropout=dropout,
        )
        self.news_branch = NewsProjectionBranch(
            input_dim=news_input_dim,
            proj_dim=news_proj_dim,
        )
        total_dim = tech_hidden_dim + news_proj_dim  # 128
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor | None = None,
        x_news: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_tech = self.tech_branch(x_tech)
        h_news = self.news_branch(x_news)
        h_concat = torch.cat([h_tech, h_news], dim=1)
        logits = self.classifier(h_concat)
        return logits, None


class StaticFusionModel(nn.Module):
    """Baseline 4: Static multimodal fusion without Soft Gating Network."""

    def __init__(
        self,
        tech_input_dim: int = 7,
        macro_input_dim: int = 3,
        news_input_dim: int = 768,
        tech_hidden_dim: int = 64,
        macro_latent_dim: int = 16,
        news_proj_dim: int = 64,
        classifier_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tech_branch = TechnicalLSTMBranch(
            input_dim=tech_input_dim,
            hidden_dim=tech_hidden_dim,
            dropout=dropout,
        )
        self.macro_branch = MacroMLPBranch(
            input_dim=macro_input_dim,
            latent_dim=macro_latent_dim,
        )
        self.news_branch = NewsProjectionBranch(
            input_dim=news_input_dim,
            proj_dim=news_proj_dim,
        )
        total_dim = tech_hidden_dim + macro_latent_dim + news_proj_dim  # 144
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(
        self,
        x_tech: torch.Tensor,
        x_macro: torch.Tensor,
        x_news: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h_tech = self.tech_branch(x_tech)
        h_macro = self.macro_branch(x_macro)
        h_news = self.news_branch(x_news)
        h_concat = torch.cat([h_tech, h_macro, h_news], dim=1)
        logits = self.classifier(h_concat)
        return logits, None

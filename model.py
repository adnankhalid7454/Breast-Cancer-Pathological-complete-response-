"""
Multi-modal TabNet-style architecture with gated late fusion.

Pipeline per modality (clinical / tumor-level / tumor-breast-ratio):
    raw features -> TabNetEncoder (sparse attention, N steps) -> hidden_dim vector

All modality vectors are combined with GatedFusion (a learned per-modality
gate, lighter-weight than cross-attention) and passed to a linear classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sparsemax(nn.Module):
    """Sparsemax activation (Martins & Astudillo, 2016) — used for TabNet's
    feature attention masks so that irrelevant features get exactly zero weight."""

    def forward(self, input):
        original_size = input.size()
        input = input.view(input.size(0), -1)
        dim = 1
        number_of_logits = input.size(dim)
        input_sorted, _ = torch.sort(input, descending=True, dim=dim)
        input_cumsum = input_sorted.cumsum(dim) - 1
        range_values = torch.arange(1, number_of_logits + 1, device=input.device).float()
        is_gt = (input_sorted - input_cumsum / range_values) > 0
        k = is_gt.float().sum(dim).unsqueeze(dim)
        tau_sum = input_cumsum.gather(dim, (k - 1).long())
        tau = tau_sum / k.float()
        output = torch.clamp(input - tau, min=0)
        return output.view(original_size)


class FeatureTransformer(nn.Module):
    """GLU-gated linear block with a residual connection."""

    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(num_features, hidden_dim)
        self.fc_gate = nn.Linear(num_features, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(num_features, hidden_dim)
        nn.init.xavier_uniform_(self.residual.weight, gain=0.1)

    def forward(self, x):
        gate = torch.sigmoid(self.fc_gate(x))
        out = self.bn(self.fc(x) * gate)
        out = F.relu(out)
        out = self.fc_out(out)
        return out + self.residual(x)


class TabNetEncoder(nn.Module):
    """N-step sparse-attention encoder for a single modality."""

    def __init__(self, num_features, n_steps=3, hidden_dim=64, gamma=1.3):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.sparsemax = Sparsemax()
        self.input_bn = nn.BatchNorm1d(num_features)
        self.attention_layers = nn.ModuleList(
            [nn.Linear(num_features, num_features) for _ in range(n_steps)]
        )
        self.transformers = nn.ModuleList(
            [FeatureTransformer(num_features, hidden_dim) for _ in range(n_steps)]
        )
        self.step_bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features) for _ in range(n_steps)]
        )

    def forward(self, x):
        x = self.input_bn(x)
        prior = torch.ones_like(x)
        aggregated = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for i in range(self.n_steps):
            attn_scores = self.attention_layers[i](x)
            mask = self.sparsemax(attn_scores * prior)
            x_masked = self.step_bns[i](x * mask)
            transformed = self.transformers[i](x_masked)
            aggregated = aggregated + F.relu(transformed)
            prior = prior * (self.gamma - mask)
        return aggregated


class GatedFusion(nn.Module):
    """Learns a per-modality scalar gate (0-1) from the concatenated
    modality encodings, then fuses the gated encodings through an MLP."""

    def __init__(self, hidden_dim, num_modalities, dropout=0.4):
        super().__init__()
        self.num_modalities = num_modalities
        total_dim = hidden_dim * num_modalities

        self.gate = nn.Sequential(
            nn.Linear(total_dim, num_modalities),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim // 2

    def forward(self, encoded_list):
        stacked = torch.stack(encoded_list, dim=1)          # [B, M, H]
        combined = stacked.view(stacked.size(0), -1)         # [B, M*H]
        gates = self.gate(combined).unsqueeze(-1)             # [B, M, 1]
        gated = (stacked * gates).view(stacked.size(0), -1)   # [B, M*H]
        return self.mlp(gated)


class TabNet(nn.Module):
    """Multi-modal classifier: one TabNetEncoder per modality + GatedFusion + linear head."""

    def __init__(self, feature_dims, num_classes,
                 n_steps=3, hidden_dim=64, gamma=1.3, dropout=0.4):
        super().__init__()
        self.encoders = nn.ModuleList(
            [TabNetEncoder(dim, n_steps, hidden_dim, gamma) for dim in feature_dims]
        )
        self.fusion = GatedFusion(hidden_dim, len(feature_dims), dropout)
        self.classifier = nn.Linear(self.fusion.out_dim, num_classes)

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        encoded = [enc(inp) for enc, inp in zip(self.encoders, inputs)]
        fused = self.fusion(encoded)
        return self.classifier(fused)

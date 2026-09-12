import torch
from torch import nn
from transformers import AutoModel


class SymbolicRAGModel(nn.Module):
    MODES = {"full", "text_only", "retrieval_only", "symbolic_only"}

    def __init__(
        self,
        backbone_name="sentence-transformers/all-MiniLM-L6-v2",
        mode="full",
        attention_heads=6,
        dropout=0.1
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode = mode
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden = self.encoder.config.hidden_size
        if hidden % attention_heads:
            raise ValueError("hidden size must be divisible by attention_heads")

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden)
        self.two_stream_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)
        self.regressor = nn.Linear(hidden, 1)

    def forward(
        self,
        dream_input_ids,
        dream_attention_mask,
        symbol_input_ids,
        symbol_attention_mask
    ):
        dream_states = self.encoder(
            input_ids=dream_input_ids,
            attention_mask=dream_attention_mask
        ).last_hidden_state
        symbol_states = self.encoder(
            input_ids=symbol_input_ids,
            attention_mask=symbol_attention_mask
        ).last_hidden_state

        dream_pool = masked_mean(dream_states, dream_attention_mask)
        symbol_pool = masked_mean(symbol_states, symbol_attention_mask)

        if self.mode == "text_only":
            fused = dream_pool
            attention_weights = None
        elif self.mode == "symbolic_only":
            fused = symbol_pool
            attention_weights = None
        elif self.mode == "retrieval_only":
            fused = self.two_stream_projection(torch.cat((dream_pool, symbol_pool), dim=-1))
            attention_weights = None
        else:
            attended, attention_weights = self.cross_attention(
                query=dream_states,
                key=symbol_states,
                value=symbol_states,
                key_padding_mask=~symbol_attention_mask.bool(),
                need_weights=True,
                average_attn_weights=False
            )
            attended = self.cross_norm(dream_states + attended)
            attended_pool = masked_mean(attended, dream_attention_mask)
            fused = self.two_stream_projection(torch.cat((dream_pool, attended_pool), dim=-1))

        fused = self.dropout(fused)
        class_logits = self.classifier(fused).squeeze(-1)
        psqi_prediction = torch.sigmoid(self.regressor(fused).squeeze(-1)) * 21.0
        return {
            "class_logits": class_logits,
            "psqi_prediction": psqi_prediction,
            "attention_weights": attention_weights
        }


def multitask_loss(outputs, labels, psqi, alpha=1.0):
    classification = nn.functional.binary_cross_entropy_with_logits(outputs["class_logits"], labels)
    regression = nn.functional.mse_loss(outputs["psqi_prediction"] / 21.0, psqi / 21.0)
    total = classification + alpha * regression
    return total, {"classification_loss": classification.detach(), "regression_loss": regression.detach()}


def masked_mean(states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(states.dtype)
    return (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

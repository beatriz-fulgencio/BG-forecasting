"""
Transformer models for blood glucose forecasting.

Implementations of attention-based approaches including:
- Recurrent self-attention encoder, ported from Cui et al.
- Cross-attention over insulin and carbohydrate signals TODO
- Encoder-decoder architectures TODO

Model addapted from "Personalised Short-Term Glucose Prediction via Recurrent Self-Attention Network" (Cui et al.), https://github.com/r-cui/GluPred. 
"""

import math
from .base_model import BasePyTorchBGModel

#pyTorch implementation
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    def positional_encoding(length, d_model):
        """ Sinusoidal positional encoding.
        """
        pe = torch.zeros(length, d_model)
        pos = torch.arange(length).unsqueeze(1)

        pe[:, 0::2] = torch.sin(pos / torch.pow(10000, torch.arange(0, d_model, step=2, dtype=torch.float32) / d_model))
        pe[:, 1::2] = torch.cos(pos / torch.pow(10000, torch.arange(1, d_model, step=2, dtype=torch.float32) / d_model))

        return pe

    def get_past_mask(length):
        """ Past mask: row is query, column is key, True is not accessible. """
        return torch.triu(torch.ones((length, length)), diagonal=1).bool()

    class MultiHeadedAttention(nn.Module):
        """ Multi-head dot-product attention """

        def __init__(self, n_heads, d_model, dropout):
            super().__init__()
            assert d_model % n_heads == 0
            self.dim_per_head = d_model // n_heads
            self.n_heads = n_heads

            self.linear_keys = nn.Linear(d_model, n_heads * self.dim_per_head)
            self.linear_values = nn.Linear(d_model, n_heads * self.dim_per_head)
            self.linear_query = nn.Linear(d_model, n_heads * self.dim_per_head)

            self.softmax = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)
            self.final_linear = nn.Linear(d_model, d_model)

        def forward(self, key, value, query, mask=None):
            batch_size = key.shape[0]

            def shape(x):
                return x.view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

            def unshape(x):
                return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_per_head)

            key = shape(self.linear_keys(key))
            value = shape(self.linear_values(value))
            query = shape(self.linear_query(query))

            query = query / math.sqrt(self.dim_per_head)
            scores = torch.matmul(query, key.transpose(2, 3)).float()
            if mask is not None:
                scores = scores.masked_fill(mask.unsqueeze(1), -1e18)

            attention = self.softmax(scores).to(query.dtype)
            context = unshape(torch.matmul(self.dropout(attention), value))
            return self.final_linear(context)

    class PositionwiseFeedForward(nn.Module):
        """ Two layer feed-forward with pre-norm and a residual connection """

        def __init__(self, d_model, d_ff, dropout):
            super().__init__()
            self.w_1 = nn.Linear(d_model, d_ff)
            self.w_2 = nn.Linear(d_ff, d_model)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
            self.dropout_1 = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.dropout_2 = nn.Dropout(dropout)

        def forward(self, x):
            inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
            output = self.dropout_2(self.w_2(inter))
            return output + x

    class EncoderLayer(nn.Module):
        """ Pre-norm self-attention block """

        def __init__(self, d_model, heads, d_ff, dropout, attention_dropout):
            super().__init__()
            self.self_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            input_norm = self.layer_norm(x)
            context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
            out = self.dropout(context) + x
            return self.feed_forward(out)

    class Encoder(nn.Module):
        """ Stack of pre-norm blocks, with positions added to the input """

        def __init__(self, num_layers, d_model, heads, d_ff, dropout, attention_dropout):
            super().__init__()
            self.layers = nn.ModuleList(
                [EncoderLayer(d_model, heads, d_ff, dropout, attention_dropout) for _ in range(num_layers)]
            )
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        def forward(self, x, mask=None):
            out = x + positional_encoding(x.shape[1], x.shape[2]).to(x.device)
            for layer in self.layers:
                out = layer(out, mask)
            out = self.layer_norm(out)

            return out.contiguous()

    class TransformerBGModel(BasePyTorchBGModel):
        """ Recurrent self-attention model for blood glucose forecasting """

        def _build_model(self):
            d_model = self.hyperparameters.get("d_model", 128)
            nhead = self.hyperparameters.get("nhead", 4)
            num_layers = self.hyperparameters.get("num_layers", 3)
            dim_feedforward = self.hyperparameters.get("dim_feedforward", 512)
            dropout = self.hyperparameters.get("dropout", 0.1)
            attention_dropout = self.hyperparameters.get("attention_dropout", 0.1)

            # Create Model
            class TransformerModel(nn.Module):
                def __init__(self, input_size, d_model, heads, num_layers, d_ff, dropout,
                             attention_dropout, output_size):
                    super().__init__()
                    # Continuous features, so a linear projection takes the place of an embedding
                    self.emb = nn.Linear(input_size, d_model)
                    self.encoder = Encoder(num_layers, d_model, heads, d_ff, dropout, attention_dropout)
                    self.final_linear = nn.Linear(d_model, input_size)

                    self.output_size = output_size

                def _transformer_forward(self, x):
                    embedded = self.emb(x)
                    mask = get_past_mask(embedded.shape[1]).unsqueeze(0).expand(
                        embedded.shape[0], -1, -1
                    ).to(embedded.device)
                    return self.final_linear(self.encoder(embedded, mask=mask))

                def forward(self, x):
                    # One step at a time: each prediction is appended to the window and fed
                    # back in, so step h is conditioned on steps 1..h-1 as Cui et al. do.
                    sequence = x
                    steps = []
                    for _ in range(self.output_size):
                        step = self._transformer_forward(sequence)[:, -1, :]
                        steps.append(step[:, 0])
                        sequence = torch.cat((sequence, step.unsqueeze(1)), dim=1)

                    predictions = torch.stack(steps, dim=1)
                    return predictions

            # assign Transformer model
            self.model = TransformerModel(input_size=self.feature_dim,
                                          d_model=d_model,
                                          heads=nhead,
                                          num_layers=num_layers,
                                          d_ff=dim_feedforward,
                                          dropout=dropout,
                                          attention_dropout=attention_dropout,
                                          output_size=self.prediction_horizon)

except ImportError:

    # Placeholder class when PyTorch is not available
    class TransformerBGModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for TransformerBGModel")

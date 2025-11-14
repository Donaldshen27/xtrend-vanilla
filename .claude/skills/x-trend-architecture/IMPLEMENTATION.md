# X-Trend Implementation Details

## Table of Contents

1. [Input Features](#input-features)
2. [Variable Selection Network (VSN)](#variable-selection-network-vsn)
3. [Entity Embeddings](#entity-embeddings)
4. [Sequence Encoder](#sequence-encoder)
5. [Cross-Attention Mechanism](#cross-attention-mechanism)
6. [Self-Attention Over Context](#self-attention-over-context)
7. [X-Trend Encoder](#x-trend-encoder)
8. [Decoder](#decoder)
9. [Complete X-Trend Model](#complete-x-trend-model)

---

## Input Features

```python
def create_input_features(prices):
    """
    Create feature vector x[t] for X-Trend model.

    Features (total dimension |X| ≈ 8):
    - Normalized returns at multiple timescales (5 features)
    - MACD indicators at multiple (S,L) pairs (3 features)
    """
    # Use EWMA for volatility calculation
    volatility = prices.pct_change().ewm(span=60).std()

    # Normalized returns: r_hat[t-t', t] = r[t-t',t] / (σ[t] * sqrt(t'))
    norm_returns = []
    for t in [1, 21, 63, 126, 252]:
        ret = (prices - prices.shift(t)) / prices.shift(t)
        r_norm = ret / (volatility * np.sqrt(t))
        norm_returns.append(r_norm)

    # MACD features
    macds = [
        macd_factor(prices, S=8, L=24),
        macd_factor(prices, S=16, L=28),
        macd_factor(prices, S=32, L=96)
    ]

    # Concatenate all features
    x_t = np.column_stack(norm_returns + macds)

    return x_t  # Shape: (T, |X|)
```

## Variable Selection Network (VSN)

Learns to weight different input features dynamically:

```python
class VariableSelectionNetwork(nn.Module):
    """
    Learn feature importance weights for each time step.

    For each feature j at time t:
    1. Transform with dedicated FFN: v[j,t] = FFN_j(x[j,t])
    2. Compute weights: w[t] = softmax(FFN_weight(x[t]))
    3. Weighted sum: VSN(x[t]) = Σ w[j,t] * v[j,t]
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim

        # Feature-specific transformations
        self.feature_ffns = nn.ModuleList([
            nn.Linear(1, hidden_dim)
            for _ in range(input_dim)
        ])

        # Weight network
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            vsn_output: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Compute feature weights
        weights = self.weight_net(x)  # (batch, seq_len, input_dim)

        # Transform each feature
        transformed = []
        for j in range(self.input_dim):
            v_j = self.feature_ffns[j](x[..., j:j+1])  # (batch, seq_len, hidden)
            transformed.append(v_j)

        transformed = torch.stack(transformed, dim=-2)  # (batch, seq, input_dim, hidden)

        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)  # (batch, seq, input_dim, 1)
        output = (transformed * weights_expanded).sum(dim=-2)  # (batch, seq, hidden)

        return output
```

## Entity Embeddings

Handle asset-specific dynamics:

```python
class EntityEmbedding(nn.Module):
    """
    Learn embeddings for each asset (ticker).

    Automatically groups similar assets in embedding space.
    Examples: Equities cluster together, commodities cluster together, etc.
    """

    def __init__(self, num_assets, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_assets, embedding_dim)

    def forward(self, asset_id):
        """
        Args:
            asset_id: Integer index for asset (0 to num_assets-1)

        Returns:
            embedding: (embedding_dim,) vector representation
        """
        return self.embedding(asset_id)
```

## Sequence Encoder

LSTM-based sequence representation with skip connections:

```python
class SequenceEncoder(nn.Module):
    """
    Encode input sequence into hidden representation.

    Architecture:
    1. VSN for feature selection
    2. LSTM for temporal modeling
    3. LayerNorm + skip connections
    4. FFN for final transformation
    """

    def __init__(self, input_dim, hidden_dim, num_assets):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.entity_embedding = EntityEmbedding(num_assets, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, asset_id):
        """
        Args:
            x: (batch, seq_len, input_dim) - input features
            asset_id: Integer asset index

        Returns:
            encoding: (batch, seq_len, hidden_dim)
        """
        # 1. Variable selection
        x_selected = self.vsn(x)  # (batch, seq_len, hidden_dim)

        # 2. LSTM encoding
        h_lstm, _ = self.lstm(x_selected)

        # 3. Skip connection + LayerNorm
        h_skip = self.layer_norm1(h_lstm + x_selected)

        # 4. FFN with entity embedding
        entity_emb = self.entity_embedding(asset_id)
        h_ffn = self.ffn(h_skip) + entity_emb.unsqueeze(1)

        # 5. Final skip connection + LayerNorm
        output = self.layer_norm2(h_ffn + h_skip)

        return output
```

## Cross-Attention Mechanism

Target attends to context sequences:

```python
class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: target queries context.

    Attention(Q, K, V) = softmax(QK^T / √d) V

    where:
    - Q (queries): From target sequence
    - K (keys): From context sequences
    - V (values): From context sequences
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, keys, values):
        """
        Args:
            query: (batch, seq_len_q, hidden_dim) - target encoding
            keys: (batch, num_contexts, hidden_dim) - context encodings
            values: (batch, num_contexts, hidden_dim) - context encodings

        Returns:
            attended: (batch, seq_len_q, hidden_dim)
            attention_weights: (batch, num_heads, seq_len_q, num_contexts)
        """
        batch_size = query.size(0)

        # Linear projections + reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim)
        K = self.W_k(keys).view(batch_size, -1, self.num_heads, self.head_dim)
        V = self.W_v(values).view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose for attention: (batch, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.hidden_dim)

        # Output projection
        output = self.W_o(attended)

        return output, attention_weights
```

## Self-Attention Over Context

Context sequences attend to each other:

```python
class SelfAttention(nn.Module):
    """
    Self-attention within context set.

    Helps identify similarities between different context sequences.
    """

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = CrossAttention(hidden_dim, num_heads)

    def forward(self, context_encodings):
        """
        Args:
            context_encodings: (batch, num_contexts, hidden_dim)

        Returns:
            attended_context: (batch, num_contexts, hidden_dim)
        """
        # Self-attention: Q=K=V=context
        attended, _ = self.attention(
            query=context_encodings,
            keys=context_encodings,
            values=context_encodings
        )

        return attended
```

## X-Trend Encoder

Complete encoder combining all components:

```python
class XTrendEncoder(nn.Module):
    """
    X-Trend encoder: processes target and context.

    Steps:
    1. Encode each context sequence
    2. Self-attention over context
    3. Encode target sequence
    4. Cross-attention: target ← context
    """

    def __init__(self, input_dim, hidden_dim, num_assets, num_heads=4):
        super().__init__()
        self.context_encoder = SequenceEncoder(input_dim, hidden_dim, num_assets)
        self.target_encoder = SequenceEncoder(input_dim, hidden_dim, num_assets)

        self.self_attention = SelfAttention(hidden_dim, num_heads)
        self.cross_attention = CrossAttention(hidden_dim, num_heads)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, target_features, target_asset_id,
               context_features_list, context_asset_ids):
        """
        Args:
            target_features: (batch, seq_len, input_dim)
            target_asset_id: Integer
            context_features_list: List of (seq_len_c, input_dim) tensors
            context_asset_ids: List of integers

        Returns:
            encoding: (batch, seq_len, hidden_dim)
        """
        # 1. Encode all context sequences (take final hidden state)
        context_encodings = []
        for ctx_features, ctx_asset in zip(context_features_list, context_asset_ids):
            ctx_enc = self.context_encoder(ctx_features.unsqueeze(0), ctx_asset)
            # Take final time step as summary
            ctx_enc_final = ctx_enc[:, -1, :]  # (1, hidden_dim)
            context_encodings.append(ctx_enc_final)

        context_encodings = torch.cat(context_encodings, dim=0).unsqueeze(0)
        # Shape: (1, num_contexts, hidden_dim)

        # 2. Self-attention over context
        context_attended = self.self_attention(context_encodings)

        # 3. Encode target
        target_enc = self.target_encoder(target_features, target_asset_id)
        # Shape: (batch, seq_len, hidden_dim)

        # 4. Cross-attention: target queries context
        cross_attended, attention_weights = self.cross_attention(
            query=target_enc,
            keys=context_attended,
            values=context_attended
        )

        # 5. FFN + residual
        output = self.layer_norm(self.ffn(cross_attended) + cross_attended)

        return output, attention_weights
```

## Decoder

Combines encoder output with target features to produce predictions:

```python
class XTrendDecoder(nn.Module):
    """
    Decoder: produces trading signals + forecasts.

    Inputs: Target features + Encoder output
    Outputs: Position z[t] + Forecast (μ, σ) or quantiles
    """

    def __init__(self, input_dim, hidden_dim, num_assets, forecast_type='gaussian'):
        super().__init__()
        self.forecast_type = forecast_type

        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)  # *2 for concat
        self.entity_embedding = EntityEmbedding(num_assets, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Forecast head
        if forecast_type == 'gaussian':
            self.forecast_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 2)  # (μ, σ)
            )
            self.ptp = nn.Sequential(  # Predictive To Position
                nn.Linear(2, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()  # Position in [-1, 1]
            )

        elif forecast_type == 'quantile':
            num_quantiles = 13
            self.forecast_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, num_quantiles)
            )
            self.ptp = nn.Sequential(
                nn.Linear(num_quantiles, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh()
            )

        # Direct Sharpe head (alternative to PTP)
        self.sharpe_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, target_features, encoder_output, asset_id, use_ptp=True):
        """
        Args:
            target_features: (batch, seq_len, input_dim)
            encoder_output: (batch, seq_len, hidden_dim) from encoder
            asset_id: Integer
            use_ptp: If True, use PTP head; else use direct Sharpe head

        Returns:
            position: (batch, seq_len, 1)
            forecast: (batch, seq_len, 2 or num_quantiles)
        """
        # 1. VSN for target features
        x_selected = self.vsn(target_features)

        # 2. Concatenate with encoder output
        combined = torch.cat([x_selected, encoder_output], dim=-1)

        # 3. LSTM
        h_lstm, _ = self.lstm(combined)

        # 4. Skip connection + LayerNorm
        h_skip = self.layer_norm1(h_lstm + combined[:, :, :self.hidden_dim])

        # 5. FFN with entity embedding (for few-shot only, not zero-shot)
        entity_emb = self.entity_embedding(asset_id)
        h_ffn = self.ffn(h_skip) + entity_emb.unsqueeze(1)

        # 6. Final representation
        representation = self.layer_norm2(h_ffn + h_skip)

        # 7. Generate forecast
        forecast = self.forecast_head(representation)

        # 8. Generate position
        if use_ptp:
            # Use forecast to inform position
            position = self.ptp(forecast)
        else:
            # Direct position from representation
            position = self.sharpe_head(representation)

        return position, forecast
```

## Complete X-Trend Model

```python
class XTrendModel(nn.Module):
    """
    Complete X-Trend architecture.

    Variants:
    - X-Trend: Direct Sharpe optimization (no forecast)
    - X-Trend-G: Joint Gaussian MLE + Sharpe
    - X-Trend-Q: Joint Quantile Regression + Sharpe
    """

    def __init__(self, input_dim=8, hidden_dim=64, num_assets=50,
                 forecast_type='gaussian', num_heads=4):
        super().__init__()
        self.encoder = XTrendEncoder(input_dim, hidden_dim, num_assets, num_heads)
        self.decoder = XTrendDecoder(input_dim, hidden_dim, num_assets, forecast_type)

    def forward(self, target_features, target_asset_id,
               context_features, context_asset_ids, use_ptp=True):
        """
        Forward pass.

        Args:
            target_features: (batch, seq_len, input_dim)
            target_asset_id: Integer
            context_features: List of (seq_len_c, input_dim) tensors
            context_asset_ids: List of integers
            use_ptp: Use PTP head or direct Sharpe head

        Returns:
            position: (batch, seq_len, 1)
            forecast: (batch, seq_len, 2 or num_quantiles)
            attention_weights: (batch, num_heads, seq_len, num_contexts)
        """
        # Encode
        encoder_output, attention_weights = self.encoder(
            target_features, target_asset_id,
            context_features, context_asset_ids
        )

        # Decode
        position, forecast = self.decoder(
            target_features, encoder_output,
            target_asset_id, use_ptp
        )

        return position, forecast, attention_weights
```

## Interpretation

### Attention Weights

```python
def interpret_attention(model, target, context):
    """
    Visualize which context sequences model attends to.

    Useful for understanding model decisions.
    """
    _, _, attention_weights = model(target, context)
    # Shape: (1, num_heads, seq_len, num_contexts)

    # Average across heads and time
    avg_attention = attention_weights.mean(dim=(0, 1, 2))  # (num_contexts,)

    # Top 3 most important contexts
    top_k = torch.topk(avg_attention, k=3)

    for rank, (weight, idx) in enumerate(zip(top_k.values, top_k.indices)):
        ctx = context[idx]
        print(f"Rank {rank+1}: {ctx['asset']} ({weight:.2%} attention)")
```

---

**Last Updated**: March 2024
**Reference Type**: Implementation Details

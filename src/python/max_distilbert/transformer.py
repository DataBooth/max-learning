"""DistilBERT Transformer implementation for MAX Graph.

DistilBERT uses different weight naming than BERT:
- Attention: q_lin, k_lin, v_lin, out_lin (not query, key, value, dense)
- Layer norms: sa_layer_norm, output_layer_norm
- FFN: lin1, lin2
"""

from max.dtype import DType
from max.graph import ops, TensorValue
from max.nn import Module


class DistilBertAttention(Module):
    """Multi-head self-attention for DistilBERT.
    
    Weight structure:
    - q_lin.weight, q_lin.bias
    - k_lin.weight, k_lin.bias
    - v_lin.weight, v_lin.bias
    - out_lin.weight, out_lin.bias
    """
    
    def __init__(self, weights, config, dtype, device):
        """Initialize attention layer.
        
        Args:
            weights: Weight accessor for this layer (e.g., weights.attention)
            config: Model configuration with hidden_size, num_attention_heads
            dtype: Data type for computations
            device: Device to allocate tensors on
        """
        self.hidden_size = config.dim  # DistilBERT uses 'dim' not 'hidden_size'
        self.num_attention_heads = config.n_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        # Query, key, value projections
        self.q_lin_weight = weights.q_lin.weight.allocate(DType.float32).cast(dtype)
        self.q_lin_bias = weights.q_lin.bias.allocate(DType.float32).cast(dtype)
        self.k_lin_weight = weights.k_lin.weight.allocate(DType.float32).cast(dtype)
        self.k_lin_bias = weights.k_lin.bias.allocate(DType.float32).cast(dtype)
        self.v_lin_weight = weights.v_lin.weight.allocate(DType.float32).cast(dtype)
        self.v_lin_bias = weights.v_lin.bias.allocate(DType.float32).cast(dtype)
        
        # Output projection
        self.out_lin_weight = weights.out_lin.weight.allocate(DType.float32).cast(dtype)
        self.out_lin_bias = weights.out_lin.bias.allocate(DType.float32).cast(dtype)
    
    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        """Forward pass of attention.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Attention output: [batch_size, seq_len, hidden_size]
        """
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        
        # Project to Q, K, V: [batch, seq_len, hidden_size]
        # Linear: hidden_states @ weight^T + bias
        query = ops.matmul(hidden_states, ops.transpose(self.q_lin_weight, 1, 0)) + self.q_lin_bias
        key = ops.matmul(hidden_states, ops.transpose(self.k_lin_weight, 1, 0)) + self.k_lin_bias
        value = ops.matmul(hidden_states, ops.transpose(self.v_lin_weight, 1, 0)) + self.v_lin_bias
        
        # Reshape to [batch, num_heads, seq_len, head_size]
        query = ops.reshape(
            query,
            [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        )
        query = ops.permute(query, [0, 2, 1, 3])  # [batch, num_heads, seq_len, head_size]
        
        key = ops.reshape(
            key,
            [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        )
        key = ops.permute(key, [0, 2, 1, 3])
        
        value = ops.reshape(
            value,
            [batch_size, seq_length, self.num_attention_heads, self.attention_head_size]
        )
        value = ops.permute(value, [0, 2, 1, 3])
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # [batch, num_heads, seq_len, seq_len]
        key_transposed = ops.permute(key, [0, 1, 3, 2])
        attention_scores = ops.matmul(query, key_transposed)
        
        # Scale by sqrt(attention_head_size)
        scale = ops.constant(self.attention_head_size ** 0.5, dtype=attention_scores.dtype, device=attention_scores.device)
        attention_scores = attention_scores / scale
        
        # Apply attention mask (convert 0/1 mask to additive mask)
        if attention_mask is not None:
            # Expand mask: [batch, 1, 1, seq_len]
            mask_expanded = ops.reshape(attention_mask, [batch_size, 1, 1, seq_length])
            mask_expanded = ops.cast(mask_expanded, attention_scores.dtype)
            
            # Create additive mask: (1 - mask) * -10000
            ones = ops.constant(1.0, dtype=mask_expanded.dtype, device=mask_expanded.device)
            inverted_mask = ones - mask_expanded
            additive_mask = inverted_mask * ops.constant(-10000.0, dtype=inverted_mask.dtype, device=inverted_mask.device)
            
            attention_scores = ops.add(attention_scores, additive_mask)
        
        # Softmax to get attention probabilities
        attention_probs = ops.softmax(attention_scores, axis=-1)
        
        # Apply attention to values: [batch, num_heads, seq_len, head_size]
        context = ops.matmul(attention_probs, value)
        
        # Reshape back: [batch, seq_len, hidden_size]
        context = ops.permute(context, [0, 2, 1, 3])
        context = ops.reshape(context, [batch_size, seq_length, self.hidden_size])
        
        # Output projection
        output = ops.matmul(context, ops.transpose(self.out_lin_weight, 1, 0)) + self.out_lin_bias
        
        return output


class DistilBertFeedForward(Module):
    """Feed-forward network for DistilBERT.
    
    Weight structure:
    - lin1.weight, lin1.bias (hidden_size -> intermediate_size)
    - lin2.weight, lin2.bias (intermediate_size -> hidden_size)
    """
    
    def __init__(self, weights, config, dtype, device):
        """Initialize feed-forward network.
        
        Args:
            weights: Weight accessor for FFN (e.g., weights.ffn)
            config: Model configuration
            dtype: Data type
            device: Device
        """
        # First linear layer
        self.lin1_weight = weights.lin1.weight.allocate(DType.float32).cast(dtype)
        self.lin1_bias = weights.lin1.bias.allocate(DType.float32).cast(dtype)
        
        # Second linear layer
        self.lin2_weight = weights.lin2.weight.allocate(DType.float32).cast(dtype)
        self.lin2_bias = weights.lin2.bias.allocate(DType.float32).cast(dtype)
    
    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Forward pass.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        
        Returns:
            FFN output: [batch_size, seq_len, hidden_size]
        """
        # First projection with GELU activation
        hidden_states = ops.matmul(hidden_states, ops.transpose(self.lin1_weight, 1, 0)) + self.lin1_bias
        hidden_states = ops.gelu(hidden_states)
        
        # Second projection
        hidden_states = ops.matmul(hidden_states, ops.transpose(self.lin2_weight, 1, 0)) + self.lin2_bias
        
        return hidden_states


class DistilBertTransformerBlock(Module):
    """Single DistilBERT transformer block.
    
    Architecture:
    1. Multi-head self-attention
    2. Layer norm (sa_layer_norm)
    3. Feed-forward network
    4. Layer norm (output_layer_norm)
    
    Note: DistilBERT uses pre-norm architecture (layer norm before sublayer).
    """
    
    def __init__(self, weights, layer_idx, config, dtype, device):
        """Initialize transformer block.
        
        Args:
            weights: Weight accessor for this layer (e.g., weights.transformer.layer[i])
            layer_idx: Layer index (for debugging)
            config: Model configuration
            dtype: Data type
            device: Device
        """
        self.layer_idx = layer_idx
        
        # Attention
        self.attention = DistilBertAttention(weights.attention, config, dtype, device)
        
        # Self-attention layer norm
        self.sa_layer_norm_weight = weights.sa_layer_norm.weight.allocate(DType.float32).cast(dtype)
        self.sa_layer_norm_bias = weights.sa_layer_norm.bias.allocate(DType.float32).cast(dtype)
        
        # Feed-forward network
        self.ffn = DistilBertFeedForward(weights.ffn, config, dtype, device)
        
        # Output layer norm
        self.output_layer_norm_weight = weights.output_layer_norm.weight.allocate(DType.float32).cast(dtype)
        self.output_layer_norm_bias = weights.output_layer_norm.bias.allocate(DType.float32).cast(dtype)
        
        self.layer_norm_eps = 1e-12
    
    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        """Forward pass.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Block output: [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = ops.add(attention_output, hidden_states)  # Residual
        attention_output = ops.layer_norm(
            attention_output,
            self.sa_layer_norm_weight,
            self.sa_layer_norm_bias,
            self.layer_norm_eps,
        )
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(attention_output)
        ffn_output = ops.add(ffn_output, attention_output)  # Residual
        ffn_output = ops.layer_norm(
            ffn_output,
            self.output_layer_norm_weight,
            self.output_layer_norm_bias,
            self.layer_norm_eps,
        )
        
        return ffn_output


class DistilBertTransformerEncoder(Module):
    """DistilBERT transformer encoder (stack of transformer blocks)."""
    
    def __init__(self, weights, config, dtype, device):
        """Initialize encoder.
        
        Args:
            weights: Weight accessor (e.g., weights.transformer)
            config: Model configuration
            dtype: Data type
            device: Device
        """
        self.num_layers = config.n_layers
        
        # Create all transformer blocks
        self.layers = []
        for i in range(self.num_layers):
            layer = DistilBertTransformerBlock(
                weights.layer[i],
                layer_idx=i,
                config=config,
                dtype=dtype,
                device=device
            )
            self.layers.append(layer)
    
    def __call__(self, hidden_states: TensorValue, attention_mask: TensorValue) -> TensorValue:
        """Forward pass through all layers.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            Encoder output: [batch_size, seq_len, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states

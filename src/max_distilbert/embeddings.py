"""DistilBERT-specific embedding layer (no token type embeddings)."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.weights import Weights
from max.nn import Module


class DistilBertEmbeddings(Module):
    """DistilBERT embedding layer without token type embeddings."""

    def __init__(
        self,
        weights: Weights,
        config,
        dtype: DType,
        device: DeviceRef,
    ):
        """Initialize DistilBERT embeddings.
        
        Args:
            weights: Embedding weights from the model
            config: HuggingFace model configuration
            dtype: Data type for computations
            device: Device to run on
        """
        super().__init__()
        
        # Word embeddings
        self.word_embeddings = weights.word_embeddings.weight.allocate(
            DType.float32,
            [config.vocab_size, config.hidden_size],
        ).cast(dtype)
        
        # Position embeddings
        self.position_embeddings = weights.position_embeddings.weight.allocate(
            DType.float32,
            [config.max_position_embeddings, config.hidden_size],
        ).cast(dtype)
        
        # LayerNorm
        self.LayerNorm_weight = weights.LayerNorm.weight.allocate(
            DType.float32, [config.hidden_size]
        ).cast(dtype)
        self.LayerNorm_bias = weights.LayerNorm.bias.allocate(
            DType.float32, [config.hidden_size]
        ).cast(dtype)
        
        self.layer_norm_eps = config.layer_norm_eps if hasattr(config, 'layer_norm_eps') else 1e-12

    def __call__(
        self,
        input_ids: TensorValue,
    ) -> TensorValue:
        """Forward pass.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]
            
        Returns:
            Embeddings, shape [batch_size, seq_len, hidden_size]
        """
        # Get word embeddings
        embeddings = ops.gather(self.word_embeddings, input_ids, axis=0)
        
        # Create position IDs (0, 1, 2, ..., seq_len-1)
        seq_length = input_ids.shape[1]
        position_ids = ops.range(
            0,
            seq_length,
            1,
            dtype=DType.int64,
            device=input_ids.device,
        )
        
        # Expand position_ids to match batch size
        # Shape: [1, seq_len] -> [batch_size, seq_len]
        position_ids = ops.unsqueeze(position_ids, 0)
        position_ids = ops.broadcast_to(
            position_ids,
            ("batch_size", seq_length),
        )
        
        # Get position embeddings
        position_embeddings = ops.gather(self.position_embeddings, position_ids, axis=0)
        
        # Add embeddings (no token type embeddings for DistilBERT)
        embeddings = embeddings + position_embeddings
        
        # Layer normalization
        embeddings = ops.layer_norm(
            embeddings,
            self.LayerNorm_weight,
            self.LayerNorm_bias,
            self.layer_norm_eps,
        )
        
        return embeddings

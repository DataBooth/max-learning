"""DistilBERT graph implementation for sentiment classification using MAX Graph API.

This implementation adapts the BERT example from Modular to create a sentiment
classifier. Key differences from BERT:
- 6 transformer layers instead of 12 (DistilBERT)
- No token_type_ids (DistilBERT doesn't use segment embeddings)
- Classification head instead of pooling layer (for sentiment classification)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import Weights
from max.nn import Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

# Import our custom DistilBERT components
from .embeddings import DistilBertEmbeddings
from .transformer import DistilBertTransformerEncoder


class DistilBertClassifier(Module):
    """DistilBERT model for sentiment classification."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        
        # DistilBERT uses "distilbert" prefix for weights
        model_weights = weights.distilbert if hasattr(weights, "distilbert") else weights
        
        # Use custom DistilBERT embeddings (no token type embeddings)
        self.embeddings = DistilBertEmbeddings(
            weights=model_weights.embeddings,
            config=huggingface_config,
            dtype=dtype,
            device=device,
        )
        
        # Transformer encoder (6 layers for DistilBERT)
        self.encoder = DistilBertTransformerEncoder(
            weights=model_weights.transformer,
            config=huggingface_config,
            dtype=dtype,
            device=device,
        )
        
        # Classification head (sentiment: 2 classes)
        # DistilBERT sequence classification uses: pre_classifier → ReLU → dropout → classifier
        # We skip dropout in inference mode
        self.pre_classifier_weight = weights.pre_classifier.weight.allocate(DType.float32).cast(dtype)
        self.pre_classifier_bias = weights.pre_classifier.bias.allocate(DType.float32).cast(dtype)
        self.classifier_weight = weights.classifier.weight.allocate(DType.float32).cast(dtype)
        self.classifier_bias = weights.classifier.bias.allocate(DType.float32).cast(dtype)
        self.hidden_size = huggingface_config.hidden_size

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, seq_len]
            
        Returns:
            logits: Classification logits, shape [batch_size, num_labels]
        """
        # Get embeddings (no token_type_ids for DistilBERT)
        embedding_output = self.embeddings(input_ids)
        
        # Pass through transformer encoder
        # Note: Our custom encoder expects 2D attention mask [batch, seq_len]
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        
        # Extract [CLS] token representation (first token)
        # Shape: [batch_size, hidden_size]
        cls_output = encoder_outputs[:, 0, :]
        
        # Classification head: pre_classifier → ReLU → classifier
        # (dropout is skipped in inference mode)
        pooled_output = ops.matmul(cls_output, ops.transpose(self.pre_classifier_weight, 1, 0)) + self.pre_classifier_bias
        pooled_output = ops.relu(pooled_output)
        
        # Final classification layer
        logits = ops.matmul(pooled_output, ops.transpose(self.classifier_weight, 1, 0)) + self.classifier_bias
        
        return logits


def build_graph(
    pipeline_config: PipelineConfig,
    weights: Weights,
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
) -> Graph:
    """Build the DistilBERT classification graph.
    
    Args:
        pipeline_config: Pipeline configuration
        weights: Model weights loaded from HuggingFace
        huggingface_config: HuggingFace model configuration
        dtype: Data type for computations
        input_device: Device to run inference on
        
    Returns:
        Compiled MAX Graph ready for inference
    """
    # Define input tensor types
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=input_device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=input_device
    )
    
    # Build the graph
    with Graph(
        "distilbert_classifier",
        input_types=[input_ids_type, attention_mask_type],
    ) as graph:
        # Instantiate the model
        distilbert = DistilBertClassifier(
            pipeline_config,
            weights,
            huggingface_config,
            dtype,
            device=input_device,
        )
        
        # Get inputs
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        
        # Run forward pass and output logits
        logits = distilbert(input_ids, attention_mask)
        graph.output(logits)
    
    return graph

"""DistilBERT model configuration for sentiment classification."""

from __future__ import annotations

from max.dtype import DType
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib import KVCacheConfig
from transformers import AutoConfig


class DistilBertConfig:
    """Configuration helper for DistilBERT models."""

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """Get the number of transformer layers."""
        return huggingface_config.n_layers

    @staticmethod
    def get_kv_params(
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        """Get KV cache parameters.
        
        Note: DistilBERT doesn't use KV cache (not a generative model),
        but we return dummy params for compatibility with the pipeline infrastructure.
        """
        return KVCacheParams(
            dtype=cache_dtype,
            n_kv_heads=0,  # No KV cache
            head_size=0,
            n_layers=DistilBertConfig.get_num_layers(huggingface_config),
        )

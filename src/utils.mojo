"""
Utility functions for the inference service.
"""

from logger import Logger, Level
from math import exp


fn sigmoid(x: Float64) -> Float64:
    """
    Sigmoid activation function.
    
    Args:
        x: Input value.
    
    Returns:
        Sigmoid(x) in range (0, 1).
    """
    return 1.0 / (1.0 + exp(-x))


fn normalize_text(text: String) -> String:
    """
    Normalize input text.
    
    Args:
        text: Input text.
    
    Returns:
        Normalized text (lowercased, trimmed).
    """
    # TODO: Implement text normalization
    # - Convert to lowercase
    # - Trim whitespace
    # - Remove special characters (optional)
    return text


fn truncate_text(text: String, max_length: Int) -> String:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text.
        max_length: Maximum character length.
    
    Returns:
        Truncated text.
    """
    # TODO: Implement truncation
    if len(text) <= max_length:
        return text
    return text  # Placeholder

"""Shared fixtures for model tests."""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_features():
    """Sample input features (batch, sequence, features)."""
    batch_size, seq_len, num_features = 32, 126, 8
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, num_features)


@pytest.fixture
def sample_entity_ids():
    """Sample entity IDs for embedding lookup."""
    batch_size = 32
    num_entities = 50
    torch.manual_seed(42)
    return torch.randint(0, num_entities, (batch_size,))


@pytest.fixture
def model_config():
    """Default model configuration."""
    from xtrend.models.types import ModelConfig
    return ModelConfig()

import torch
import numpy as np
import pytest

from ppo_pytorch.models.policy_layers import CategoricalPolicy


@pytest.mark.parametrize("input_dim", [10])
@pytest.mark.parametrize("output_dim", [5])
@pytest.mark.parametrize("sample_size", [1, 2])
def test_categorical_policy(input_dim, output_dim, sample_size):
    inp_tensor = torch.randn((sample_size, input_dim))
    policy = CategoricalPolicy(input_dim, output_dim)
    with torch.no_grad():
        forward_out = policy.forward(inp_tensor)
        assert forward_out.shape == (sample_size, output_dim)
        assert (forward_out > 0).all()
        assert np.allclose(np.asarray(forward_out).sum(axis=1), 1)

        sample_action, sample_logprob = policy.sample(inp_tensor)

        assert len(sample_action.shape) == 1
        assert sample_action.shape[0] == sample_size
        assert sample_action.dtype in {torch.int, torch.long}
        assert all([action in torch.arange(output_dim) for action in sample_action])

        assert sample_logprob.dtype == torch.float32
        assert sample_logprob.shape == sample_action.shape
        assert (sample_logprob <= 0).all()

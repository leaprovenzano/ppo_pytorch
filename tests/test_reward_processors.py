import torch
import numpy as np
import pytest

from ppo_pytorch.reward_processors import RewardProcessor, ShapedRewardProcessor, ScaledPositiveRewardProcessor


@pytest.mark.parametrize("kwargs, inp, expected", [({}, [0., 1., -1], [0., 2., -1]),
                                                   ({'clip': (-1, 1)}, [0., 1., -1], [0., 1., -1])])
def test_scale_positive_processor(kwargs, inp, expected):
    processor = ScaledPositiveRewardProcessor(positive_scale=2., **kwargs)
    result = processor.shape(torch.FloatTensor(inp))
    assert type(result) is torch.Tensor
    assert result.tolist() == expected


@pytest.mark.parametrize("kwargs, inp, expected", [({'scale': 2}, [0., 1., -1], [0., 2., -2]),
                                                   ({'clip': (-1, 1)}, [0., 2., -1], [0., 1., -1]),
                                                   ({'clip': (-1, 1), 'scale': .5}, [0., 4., -1], [0., 1., -.5])])
def test_shaped_processor(kwargs, inp, expected):
    processor = ShapedRewardProcessor(**kwargs)
    result = processor.shape(torch.FloatTensor(inp))
    assert type(result) is torch.Tensor
    assert result.tolist() == expected



@pytest.mark.parametrize("inp, expected", [([-1., 0,  2., -4., 1.], [ -0.9375 , 0.125, .25 , -3.5, 1.0])])
def test_compute_discount_returns(inp, expected):
    processor = RewardProcessor(gamma=.5)
    result = processor.compute_discount_returns(torch.FloatTensor(inp))
    assert type(result) is torch.Tensor
    assert result.tolist() == expected
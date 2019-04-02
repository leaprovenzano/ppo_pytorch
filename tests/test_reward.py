import torch

import pytest

from ppo_pytorch.reward import discount, masked_discount


class Case():

    def __init__(self, inp, expected, bootstrap=0., gamma=.99):
        self.input = torch.FloatTensor(inp)
        self.expected = torch.FloatTensor(expected)
        self.bootstrap = bootstrap
        self.non_terminals = torch.ones(len(inp))
        if not self.bootstrap:
            self.non_terminals[-1] = 0.

        self.gamma = gamma


@pytest.mark.fixture
def simplecase():
    return Case([0., 0., 1], [0.9801, 0.99, 1.])


@pytest.mark.fixture
def bootstrapcase():
    return Case([0., 0., 0], [0.9703, 0.9801, 0.9900], bootstrap=1)


@pytest.mark.fixture
def maskedcase():
    case = Case([0., 0., 1, 0, 0, 1], [0.9801, 0.99, 1., 0.9801, 0.99, 1.])
    case.non_terminals = torch.FloatTensor([1., 1., 0, 1, 1, 0])
    return case


@pytest.mark.fixture
def multidimmaskedcase():
    case = Case([[0., 0., 1, 0, 0, 1]] * 3, [[0.9801, 0.99, 1., 0.9801, 0.99, 1.]] * 3)
    case.input = case.input.t()
    case.expected = case.expected.t()
    case.non_terminals = torch.FloatTensor([[1., 1., 0, 1, 1, 0]] * 3).t()
    case.bootstrap = torch.FloatTensor([[1.]] * 3).t()
    return case


@pytest.mark.fixture
def maskedbootstrapcase():
    case = Case([0., 0., 1, 0, 0, 0], [0.9801, 0.99, 1., 0.9703, 0.9801, 0.9900], bootstrap=1)
    case.non_terminals = torch.FloatTensor([1., 1., 0, 1, 1, 1])
    return case


@pytest.mark.parametrize('case', [simplecase, bootstrapcase])
def test_discount(case):
    case = case()
    result = discount(case.input, bootstrap=case.bootstrap, gamma=case.gamma)
    torch.testing.assert_allclose(result, case.expected)


@pytest.mark.parametrize('case', [simplecase, bootstrapcase, maskedcase, maskedbootstrapcase, multidimmaskedcase])
def test_masked_discount(case):
    case = case()
    result = masked_discount(case.input, non_terminals=case.non_terminals, bootstrap=case.bootstrap, gamma=case.gamma)
    torch.testing.assert_allclose(result, case.expected)

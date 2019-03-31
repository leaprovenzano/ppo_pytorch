import torch
from .network_head import NetworkHead
from ppo_pytorch.utils import inherit_docstring


@inherit_docstring
class ValueHead(NetworkHead):

    """Value head
    """

    def __init__(self, output_dim=1, *args, **kwargs):
        super(ValueHead, self).__init__(output_dim=output_dim, *args, **kwargs)
        self._loss = torch.nn.MSELoss()

    def get_value(self, inp: torch.Tensor) -> torch.Tensor:
        return self(inp)

    def loss(self, x: torch.Tensor, y_true: torch.Tensor, weight=1.) -> torch.Tensor:
        y_pred = self(x)
        return self._loss(y_pred, y_true) * weight

    def advantage(self, states: torch.Tensor, discounted_returns: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            v = self(states)
        return discounted_returns - v

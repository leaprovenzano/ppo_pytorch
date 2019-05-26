from torch import nn

from .utils import inspect_input_dim, inspect_output_dim


class NetworkHead(nn.Module):
    """NetworkHead: Base class for policy and value network heads.

    Attributes:
        output_dim (int): outputs of this NetworkHead will have this shape
        hidden (nn.Module): hidden layers of this module, should take one input and
            return a single output... so nn.Sequential is often an easy choice
        output_layer (nn.Linear): linear layer applied to output of hidden
    """

    def __init__(self, output_dim: int, hidden: nn.Module, lr=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = inspect_input_dim(hidden)
        self.lr = lr
        self.hidden = hidden
        self._hidden_output_dim = inspect_output_dim(hidden)
        self.output_layer = nn.Linear(self._hidden_ouput_dim, self.output_dim)

    @property
    def info(self):
        return dict(cls=self.__class__.__name__,
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    lr=self.lr)

    def output_activation(self, x):
        return x

    def forward(self, inp):
        x = self.hidden(inp)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    # noinspection PyUnusedFunction
    def get_optimizer_parameters(self) -> list:
        """useful when building an optimizer with multiple
        learning rates etc for multiple network heads

        Returns:
            (list): list containing a single dictionary
                of {'params', 'lr'}

        """
        p_dict = {'params': self.parameters()}
        if self.lr is not None:
            p_dict['lr'] = self.lr
        return [p_dict]

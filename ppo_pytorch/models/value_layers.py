from torch import nn
from ppo_pytorch.models.shared import NetworkHead

class ValueLayer(NetworkHead):

    def __init__(self, output_dim=1, *args, **kwargs):
        super(ValueLayer, self).__init__(output_dim=output_dim, *args, **kwargs)



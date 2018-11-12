from torch import nn


class ValueLayer(nn.Module):

    def __init__(self, input_dim, output_dim=1, activation=None):
        super(ValueLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_layer = nn.Linear(self.input_dim, self.output_dim)
        self.activation = activation() if activation else lambda x: x

    def forward(self, x):
        x = self.output_layer(x)
        x = self.activation(x)
        return x

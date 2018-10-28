from torch import nn



class SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, hidden_act=nn.ReLU, output_act=None):
        super(SimpleMLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.hidden_act = hidden_act
        self.output_dim = output_dim
        self.output_act = output_act

        self.hidden = self.build_hidden()
        self.output_head = self.build_output_layer()


    def build_hidden(self):
        hidden = []
        input_dim = self.input_dim
        for d in self.hidden_dims:
            hidden.append(nn.Linear(input_dim, d))
            hidden.append(self.hidden_act())
            input_dim = d
        return nn.Sequential(*hidden)

    def build_output_layer(self):
        layers = []
        input_dim = self.hidden[-2].out_features
        layers.append(nn.Linear(input_dim, self.output_dim))
        if self.output_act:
            layers.append(self.output_act())
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.hidden(x)
        x = self.output_head(x)
        return x
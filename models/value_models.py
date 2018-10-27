from torch import nn


class ValueModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, hidden_act = nn.ReLU):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)
        self.w3 = nn.Linear(hidden_size, output_dim)
        self.hidden_act = hidden_act
        
    def forward(self, x):
        x = self.hidden_act()(self.w1(x))
        x = self.hidden_act()(self.w2(x))
        return self.w3(x)
        
    def predict(self, states):
        return self.forward(states)      
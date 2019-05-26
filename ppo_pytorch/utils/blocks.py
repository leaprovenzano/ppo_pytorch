from torch import nn


class LinearBlock(nn.Module):

    @classmethod
    def from_block(cls, block, out_features, **kwargs):
        in_feats = block.out_features
        return cls(in_feats, out_features, **kwargs)

    def __init__(self, in_features, out_features, bias=True, activation=None, dropout=None, normalization=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for name, mod in (('normalization', normalization), ('activation', activation), ('dropout', dropout)):
            if mod is not None:
                self.add_module(name, mod)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

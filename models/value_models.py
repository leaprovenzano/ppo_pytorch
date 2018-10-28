from torch import nn
from ppo_pytorch.models.base import SimpleMLP

class ValueModel(SimpleMLP):

    def __init__(self, *args, **kwargs):
        super(ValueModel, self).__init__(*args, **kwargs)
        
        
    def predict(self, states):
        return self.forward(states)
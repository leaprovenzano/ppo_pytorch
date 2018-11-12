from torch import nn


class PolicyValueModel(nn.Module):

    """combined on policy actor critic 
    model. as generic as possible

    Attributes:
        hidden (torch.nn.Module): some kind of torch module this 
            is the guts of the shared model can be whatever so long 
            as it outputs something flat!

        policy (ppo_pytorch.models.policy_layers.PolicyLayer ): policy layer module 
            this should act as the policy head and have a sample, and evaluate_action method
            see ppo_pytorch.policy_layers ....

        value_model (torch.nn.Moudle): value head see ppo_pytorch.models.value_layers
    """

    def __init__(self, hidden, policy_layer, value_layer):
        super().__init__()
        self.hidden = hidden
        self.policy = policy_layer
        self.value_model = value_layer

    def forward(self, x):
        x = self.hidden(x)
        p = self.policy(x)
        v = self.value(x)
        return p, v

    def sample_action(self, x):
        x = self.hidden(x)
        action, logprob = self.policy.sample(x)
        value = self.value_model(x)
        return action, logprob, value

    def evaluate_actions(self, x, action):
        x = self.hidden(x)
        logprob = self.policy.evaluate_action(x, action)
        value = self.value_model(x)
        return logprob, value

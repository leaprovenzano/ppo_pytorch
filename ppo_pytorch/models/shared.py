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
        v = self.value_model(x)
        return p, v

    def sample_action(self, x):
        x = self.hidden(x)
        action, logprob = self.policy.sample(x)
        value = self.value_model(x)
        return action, logprob, value

    def evaluate_actions(self, x, action):
        x = self.hidden(x)
        logprob, entropy = self.policy.evaluate_action(x, action)
        value = self.value_model(x)
        return logprob, entropy, value

    def build_optimizer(self, optimizer, default_lr=1e-4, policy_std_lr=1e-7, *args, **kwargs):
        if hasattr(self.policy, 'log_std'):
            return optimizer([{'params': self.hidden.parameters()},
                              {'params': self.value_model.parameters()},
                              {'params': self.policy.log_std, 'lr': policy_std_lr},
                              {'params': map(lambda p: p[1], filter(lambda p: p[0] != 'log_std', self.policy.named_parameters()))}], 
                              default_lr, *args, **kwargs)
        return optimizer(self.parameters(), lr=default_lr, *args, **kwargs)

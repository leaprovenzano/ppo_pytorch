from torch import nn


class NetworkHead(nn.Module):

    """baseclass for policy and value heads

    Attributes:
        hidden (nn.Module, Default=None): if provided these
            will be used as the hidden and input layers for the model
            (data goes through them before output_layer)

        input_dim (int, Default=None): input dimensions for the final output layer
            should be a single integer. If not provided at init and hidden
            layers are provided this attribute will be set automatically from 
            the last layer in self.hidden.

        lr (float, Default=None): if provided the get_optimizer_params method will
            return this value for the networkhead's learning rate

        output_activation (torch.optim.Activation, default=None): if provided this activation
            will be applied to the output head's final layer

        output_dim (int): this is the final size of the output (required for instantiation)

        output_layer (torch.nn.Linear): linear layer final layer for output, accepts inputs of size input_dim
            and returns output of size  output_dim

    """

    @classmethod
    def get_head_input_dim_from_hidden(cls, hidden):
        return list(hidden.state_dict().items())[-1][-1].shape[0]

    @classmethod
    def from_shared_hidden(cls, shared_hidden, *args, **kwargs):
        input_dim = cls.get_head_input_dim_from_hidden(shared_hidden)
        return cls(input_dim=input_dim, *args, **kwargs)

    def __init__(self, output_dim, hidden=None, input_dim=None, activation=None, lr=None):
        super(NetworkHead, self).__init__()
        if hidden is not None:
            self.hidden = hidden
            self.input_dim = self.get_head_input_dim_from_hidden(self.hidden)
        else:
            try:
                assert input_dim is not None
                self.input_dim = input_dim
                self.hidden = lambda x: x
            except AssertionError:
                raise ValueError('You must provide either hidden or input_dim in order to initilize {}'.format(self.__class__.__name__))

        self.output_dim = output_dim
        self.output_layer = nn.Linear(self.input_dim, self.output_dim)
        self.output_activation = activation() if activation else lambda x: x
        self.lr = lr

    def forward(self, inp):
        x = self.hidden(inp)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

    def get_optimizer_parameters(self):
        p_dict = {'params': self.parameters()}
        if self.lr is not None:
            p_dict['lr'] = self.lr
        return [p_dict]


class PolicyValueModel(nn.Module):

    """combined on policy actor critic model. as generic as possible

    Attributes:
        policy (ppo_pytorch.models.policy_layers.PolicyLayer): policy layer module 
            this should act as the policy head and have a sample, and evaluate_action method
            see ppo_pytorch.policy_layers ....

        shared (torch.nn.Module, Default=None): module for shared hidden layers between policy and value model

        shared_parameters (bool): if true the policy and value model share parameters

        value_model (torch.nn.Module): value head see ppo_pytorch.models.value_layers


    """

    def __init__(self, policy_model, value_model, shared=None):
        super().__init__()
        self.shared_parameters = False
        if shared is not None:
            self.shared = shared
            self.shared_parameters = True
        else:
            self.shared = lambda x: x

        self.policy = policy_model
        self.value_model = value_model

    def forward(self, x):
        if self.shared_parameters:
            x = self.shared(x)
        p = self.policy(x)
        v = self.value_model(x)
        return p, v

    def sample_action(self, x):
        if self.shared_parameters:
            x = self.shared(x)
        action, logprob = self.policy.sample(x)
        value = self.value_model(x)
        return action, logprob, value

    def evaluate_actions(self, x, action):
        if self.shared_parameters:
            x = self.shared(x)
        logprob, entropy = self.policy.evaluate_action(x, action)
        value = self.value_model(x)
        return logprob, entropy, value

    def build_optimizer(self, optimizer, lr, *args, **kwargs):

        paramlist = []
        paramlist += self.policy.get_optimizer_parameters()
        paramlist += self.value_model.get_optimizer_parameters()
        if self.shared_parameters:
            paramlist.append({'params': self.shared.parameters()})

        return optimizer(paramlist, lr=lr, *args, **kwargs)

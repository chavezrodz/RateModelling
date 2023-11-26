import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers,
        output_dim
    ):

        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
            )
        for i in range(n_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        return self.mlp(x)

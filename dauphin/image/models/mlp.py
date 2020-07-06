from torch import nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, has_fc=True):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.has_fc = has_fc

        self.linear = nn.Linear(input_dim, hidden_dim)
        if self.has_fc:
            self.fc = nn.Linear(hidden_dim, has_fc)

    def __repr__(self):
        return f"{type(self).__name__} {self.hidden_dim}"

    def forward(self, x):
        out = self.linear(x.view(-1, self.input_dim))
        out = F.relu(out)
        if self.has_fc:
            out = self.fc(out)

        return out

import torch

import torch.nn as nn


class q_network(nn.Module):
    def __init__(self, lr, input_size, output_size, num_hidden_layers=1, hidden_size=4):
        super(q_network, self).__init__()
        
        self.inp = nn.Linear(input_size, hidden_size)
        self.hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers-1)])
        self.out = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, grad=True):
        x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = nn.functional.elu(self.inp(x))
        for layer in self.hidden:
            x = nn.functional.elu(layer(x))
        x = self.out(x)
        if not grad:
            x = x.detach()
        return x

    def backward(self, state, target):
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(state, grad=True), target)
        loss.backward()
        self.optimizer.step()
        return loss.mean()

import torch

import torch.nn as nn

class q_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=4):
        super(q_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def forward(self, x, grad=True):
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if not grad:
            x = x.detach()
        return x
    
    def backward(self, state, target):
        self.optimizer.zero_grad()
        loss = self.loss_fn(target, self.forward(state, grad=True))
        loss.backward()
        self.optimizer.step()
        return loss.item()

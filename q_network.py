import torch

import torch.nn as nn

class q_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(q_network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)


    def forward(self, x, grad=True):
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=0)
        if not grad:
            x = x.detach()
        return x
    

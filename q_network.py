import numpy as np

class q_network:
    # class representing a neural net layer
    def __init__(self, input_size, output_size, activation=None, lr       = 0.001):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        self.activation_function = activation
        self.lr = lr

    def forward(self):
        pass

    def backward(self):
        pass

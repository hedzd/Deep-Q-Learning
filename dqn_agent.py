from environment import Environment
from collections import deque 
from q_network import q_network
import numpy as np

class dqn_agent:
    env = None
    def __init__(self):
        self.env = Environment()
        self.replay_buffer_size = 1000
        self.buffer = deque(maxlen=self.replayBufferSize)
        self.network = q_network()
        self.target_network = q_network()
        self.T = 100
        self.batch_size = 10
        self.convergence = False

    def select_action(self, policy='egreedy', epsilon=None, observation=None):
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")

            values = self.network.forward(observation)
            random_num = np.random.random()
            if random_num < epsilon:
                a = np.random.randint(self.env.action_space.n)
            else:
                a = np.argmax(values)   

        return a

    def update_network(self, index):
        state, action, reward, observation, done = self.buffer[index]
        #TODO: complete this function


    def run(self):
        while not self.convergence:
            self.state, info = self.env.reset_env()
            for i in range(self.T):
                action = self.select_action(self.state)
                observation, reward, done, truncated, info = self.env.step(action)
                self.buffer.append((self.state, action, reward, observation, done))
                if len(self.buffer) >= self.batch_size:
                    batch_indices = np.random.choice(len(self.buffer), self.batch_size)
                    for index in batch_indices:
                        self.update_network(index)
                        

                        


    








        

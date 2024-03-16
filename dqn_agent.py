from environment import Environment
from collections import deque 
from q_network import q_network
import numpy as np

class dqn_agent:
    env = None
    def __init__(self):
        self.env = Environment()
        self.replay_buffer_size = 1000
        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.network = q_network(input_size=self.env.env.observation_space.shape[0], output_size=self.env.env.action_space.n, hidden_size=4)
        # self.target_network = q_network()
        self.T = 100
        self.batch_size = 100
        self.convergence = False

        self.epsilon = 0.1

    def select_action(self, observation=None, policy='egreedy'):
        if policy == 'egreedy':
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            values = self.network.forward(observation, grad=False)
            random_num = np.random.random()
            if random_num < self.epsilon:
                a = np.random.randint(self.env.env.action_space.n)
            else:
                a = np.argmax(values.numpy())

        return a

    def update_network(self, index):
        state, action, reward, observation, done = self.buffer[index]
        #TODO: complete this function


    def run(self):
        while not self.convergence:
            self.state, info = self.env.reset_env()
            for i in range(self.T):
                action = self.select_action(self.state)
                self.state, reward, done, truncated, info = self.env.env.step(action)
                self.buffer.append((self.state, action, reward, done))
                if done:
                    break
                if len(self.buffer) >= self.batch_size:
                    batch_indices = np.random.choice(len(self.buffer), self.batch_size)
                    for index in batch_indices:
                        self.update_network(index)


if __name__ == '__main__':
    agent = dqn_agent()
    agent.run()
                        

                        


    








        

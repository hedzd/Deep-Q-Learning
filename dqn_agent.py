from environment import Environment
from collections import deque 
from q_network import q_network
import numpy as np
import matplotlib.pyplot as plt

class dqn_agent:
    env = None
    def __init__(self):
        self.env = Environment()
        self.replay_buffer_size = 1000
        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.network = q_network(input_size=self.env.env.observation_space.shape[0], output_size=self.env.env.action_space.n, hidden_size=4)
        # self.target_network = q_network()
        
        self.training_iterations = 10000
        self.batch_size = 100        

        self.epsilon = 0.3

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

        elif policy == 'boltzmann':
            raise NotImplementedError("Boltzmann policy not implemented yet") # TODO
        
        elif policy == 'greedy':
            values = self.network.forward(observation, grad=False)
            a = np.argmax(values.numpy())

        return a

    def update_network(self, index):
        state, action, reward, done = self.buffer[index]
        
        target = self.network.forward(state, grad=False)
        if done:
            target[action] = 0
        else:
            next_state, _, _, _ = self.buffer[index+1]
            next_q_values = self.network.forward(next_state, grad=False)
            target[action] = reward + np.max(next_q_values.numpy()) # gamma = 1
        
        return self.network.backward(state, target)
        

    def eval(self):
        rewards = []
        for _ in range(10):
            env = Environment()
            state, info = env.reset_env()
            done = False
            total_reward = 0
            while not done and total_reward < 200:
                action = self.select_action(state, policy='greedy')
                next_state, reward, done, truncated, info = env.env.step(action)
                total_reward += reward
                state = next_state
            
            rewards.append(total_reward)

        return np.mean(rewards)    
    
    def run(self):
        avg_rewards = []
        self.state, info = self.env.reset_env()
        for i in range(self.training_iterations):
            action = self.select_action(self.state)
            next_state, reward, done, truncated, info = self.env.env.step(action)
            self.buffer.append((self.state, action, reward, done))
            self.state = next_state
            if done:
                self.state, info = self.env.reset_env()
            if i % self.batch_size == 0 and i > 0:
                batch_indices = np.random.choice(len(self.buffer)-1, self.batch_size)
                losses = []
                for index in batch_indices:
                    losses.append(self.update_network(index))
                
                avg_rewards.append(self.eval())
                print(f"Iteration {i}: average loss: {np.mean(losses)}, average reward: {avg_rewards[-1]}")

        # test
        self.env = Environment(render=True)
        self.state, info = self.env.reset_env()
        done = False
        while not done:
            action = self.select_action(self.state, policy='greedy')
            next_state, reward, done, truncated, info = self.env.env.step(action)
            self.state = next_state

        
        plt.plot(avg_rewards)
        plt.show()

if __name__ == '__main__':
    agent = dqn_agent()
    agent.run()
                        

                        


    








        

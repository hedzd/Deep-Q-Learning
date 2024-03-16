from environment import Environment
from collections import deque 
from q_network import q_network
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class dqn_agent:
    env = None
    def __init__(self, experience_replay=True, use_target_network=True):
        self.env = Environment()
        self.replay_buffer_size = 1_000_000
        self.training_iterations = 5000
        self.batch_size = 16
        self.epsilon = 0.1

        self.experience_replay = experience_replay
        self.use_target_network = use_target_network

        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.network = q_network(input_size=self.env.env.observation_space.shape[0], output_size=self.env.env.action_space.n, hidden_size=4)
        
        if self.use_target_network:
            self.target_network = q_network(input_size=self.env.env.observation_space.shape[0], output_size=self.env.env.action_space.n, hidden_size=4)
            self.target_update = 50
        
        

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

    def update_network(self, state, action, reward, done, next_state):
        
        target = self.network.forward(state, grad=False)
        if done:
            target[action] = reward
        else:
            if self.use_target_network:
                next_q_values = self.target_network.forward(next_state, grad=False)
            else:
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
        pbar = tqdm(range(self.training_iterations))
        for i in pbar:
            # select action and take a step
            action = self.select_action(self.state)
            next_state, reward, done, truncated, info = self.env.env.step(action)
            
            # reset if done
            if done:
                self.state, info = self.env.reset_env()
            
            if self.experience_replay:
                # append to buffer
                self.buffer.append((self.state, action, reward, done))
                self.state = next_state
                
                if i > self.batch_size:
                    # train with random samples from the buffer
                    batch_indices = np.random.choice(len(self.buffer)-1, self.batch_size)
                    for index in batch_indices:
                        state, action, reward, done = self.buffer[index]
                        next_state, _, _, _ = self.buffer[index+1]
                        self.update_network(state, action, reward, done, next_state)
            else:
                # train with the last state
                self.update_network(self.state, action, reward, done, next_state)
                self.state = next_state

            # clone the network to the target network
            if i % self.target_update == 0 and self.use_target_network:
                self.target_network.load_state_dict(self.network.state_dict())
            
            # evaluate
            if i % 100 == 0:
                avg_rewards.append(self.eval())
                pbar.set_description(f"Average reward {avg_rewards[-1]}")
                if avg_rewards[-1] > 150:
                    print(f"Training completed in {i} iterations")
                    break

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
                        

                        


    








        

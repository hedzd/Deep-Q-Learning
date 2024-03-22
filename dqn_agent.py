from environment import Environment
from collections import deque
from q_network import q_network
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from hyperopt import fmin, tpe, hp
from functools import partial

class dqn_agent:
    env = None

    def __init__(
        self,
        training_iterations,
        learning_rate,
        gamma,
        num_hidden_layers,
        hidden_layer_size,
        exploration_policy,
        epsilon,        
        eval_episodes,
        eval_frequency,
        early_stopping,
        use_experience_replay,
        batch_size,
        replay_buffer_size,
        use_target_network,
        target_update,
        temperature,
    ):
        self.env = Environment()

        self.training_iterations = training_iterations
        self.gamma = gamma

        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        self.temperature = temperature
        
        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency
        self.early_stopping = early_stopping

        self.use_experience_replay = use_experience_replay
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        self.use_target_network = use_target_network

        self.buffer = deque(maxlen=self.replay_buffer_size)
        self.network = q_network(
            lr=learning_rate,
            input_size=self.env.env.observation_space.shape[0],
            output_size=self.env.env.action_space.n,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_layer_size,
        )

        if self.use_target_network:
            self.target_network = q_network(
                lr=learning_rate,
                input_size=self.env.env.observation_space.shape[0],
                output_size=self.env.env.action_space.n,
                num_hidden_layers=num_hidden_layers,
                hidden_size=hidden_layer_size,
            )
            self.target_update = target_update

    def select_action(self, observation=None, policy="egreedy"):
        if policy == "egreedy":
            if self.epsilon is None:
                raise KeyError("Provide an epsilon")

            values = self.network.forward(observation, grad=False)
            random_num = np.random.random()
            if random_num < self.epsilon:
                a = np.random.randint(self.env.env.action_space.n)
            else:
                a = np.argmax(values.numpy())

        elif policy == "boltzmann":
            if self.temperature is None:
                raise KeyError("Provide a temperature")
            
            values = self.network.forward(observation, grad=False)
            probabilities = np.exp(values.numpy() / self.temperature)
            probabilities = probabilities / np.sum(probabilities)  # Normalize probabilities
            if np.any(np.isnan(probabilities)):  # Handle NaN values by selecting a random action
                a = np.random.randint(self.env.env.action_space.n)
            else:
                a = np.random.choice(range(len(probabilities)), p=probabilities)
        
        elif policy == "greedy":
            values = self.network.forward(observation, grad=False)
            a = np.argmax(values.numpy())

        return a

    def update_network(self, states, actions, rewards, done, next_states):

        target = self.network.forward(states, grad=False).numpy()
        target[done][:, actions[done]] = rewards[done]        
        
        if self.use_target_network:
            next_q_values = self.target_network.forward(next_states, grad=False).numpy()
        else:
            next_q_values = self.network.forward(next_states, grad=False).numpy()
        target[~done, actions[~done]] = rewards[~done] + self.gamma * np.max(next_q_values, axis=1)[~done]

        target = torch.tensor(target)
        return self.network.backward(states, target)

    def eval(self):
        rewards = []
        for _ in range(self.eval_episodes):
            env = Environment()
            state, info = env.reset_env()
            done = False
            truncated = False
            total_reward = 0
            while not done and not truncated:
                action = self.select_action(state, policy="greedy")
                next_state, reward, done, truncated, info = env.env.step(action)
                total_reward += reward
                state = next_state

            rewards.append(total_reward)

        return np.mean(rewards)

    def run(self, test=False):
        avg_rewards = []
        self.state, info = self.env.reset_env()
        pbar = tqdm(range(self.training_iterations)) if test else range(self.training_iterations)
        for i in pbar:
            # select action and take a step
            action = self.select_action(self.state, self.exploration_policy)
            next_state, reward, done, truncated, info = self.env.env.step(action)

            # reset if done
            if done or truncated:
                self.state, info = self.env.reset_env()

            if self.use_experience_replay:
                # append to buffer
                self.buffer.append((self.state, action, reward, done or truncated))
                self.state = next_state

                if len(self.buffer) > self.batch_size:
                    # train with random samples from the buffer
                    batch_indices = np.random.choice(len(self.buffer) - 1, self.batch_size).astype(int)
                    batch = [self.buffer[index] for index in batch_indices]
                    states, actions, rewards, done = zip(*batch)
                    next_states = [self.buffer[index+1][0] for index in batch_indices]
                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    done = np.array(done)
                    next_states = np.array(next_states)
                    self.update_network(states, actions, rewards, done, next_states)
                    
            else:
                # train with the last state
                self.update_network(self.state, action, reward, done or truncated, next_state)
                self.state = next_state

            # clone the network to the target network
            if i % self.target_update == 0 and self.use_target_network:
                self.target_network.load_state_dict(self.network.state_dict())

            # evaluate
            if (i % self.eval_frequency == 0 and test) or i == self.training_iterations - 1:
                avg_rewards.append(self.eval())
                if test:
                    pbar.set_description(f"Average reward {avg_rewards[-1]}")
                if avg_rewards[-1] > self.early_stopping:
                    print(f"Training completed in {i} iterations")
                    break

        if test:
            # test
            plt.plot(avg_rewards)
            plt.show()
            
            self.env = Environment(render=True)
            self.state, info = self.env.reset_env()
            done = False
            truncated = False
            while not done and not truncated:
                action = self.select_action(self.state, policy="greedy")
                next_state, reward, done, truncated, info = self.env.env.step(action)
                self.state = next_state

        return avg_rewards[-1]

def objective(params):
    agent = dqn_agent(
        training_iterations=10_000,
        learning_rate=params['learning_rate'],
        gamma=0.99,
        num_hidden_layers=params['num_hidden_layers'],
        hidden_layer_size=params['hidden_layer_size'],
        
        exploration_policy="egreedy",
        epsilon=params['epsilon'],
        
        eval_episodes=10,
        eval_frequency=50, # not used for optimization
        early_stopping=400, # not used for optimization

        use_experience_replay=True, # cmd line option
        batch_size=params['batch_size'],
        replay_buffer_size=1_000_000, # just infinite?
        
        use_target_network=True, # cmd line option
        target_update=params['target_update']
    )
    avg_reward = agent.run()
    return -avg_reward

if __name__ == "__main__":
    
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
        'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]),
        'hidden_layer_size': hp.choice('hidden_layer_size', [16, 32, 64]),
        'epsilon': hp.uniform('epsilon', 0.1, 0.5),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'target_update': hp.choice('target_update', [5, 10, 20])
    }

    best = fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=2), max_evals=10)
    print(best)
    agent = dqn_agent(
        training_iterations=10_000,
        learning_rate=best['learning_rate'],
        gamma=0.99,
        num_hidden_layers=best['num_hidden_layers'],
        hidden_layer_size=best['hidden_layer_size'],
        
        exploration_policy="egreedy",
        epsilon=best['epsilon'],
        temperature=0.5,
        
        eval_episodes=10,
        eval_frequency=50,
        early_stopping=400,

        use_experience_replay=True, # cmd line option
        batch_size=best['batch_size'],
        replay_buffer_size=1_000_000, # just infinite?
        
        use_target_network=True, # cmd line option
        target_update=best['target_update']
    )
    agent.run(test=True)
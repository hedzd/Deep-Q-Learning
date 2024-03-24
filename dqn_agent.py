from environment import Environment
from collections import deque
from q_network import q_network
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from hyperopt import hp, fmin, tpe, space_eval
from functools import partial
from scipy.signal import savgol_filter

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
        exploration_param,        
        eval_episodes,
        eval_frequency,
        early_stopping,
        use_experience_replay,
        batch_size,
        replay_buffer_size,
        use_target_network,
        target_update
    ):
        self.env = Environment()

        self.training_iterations = training_iterations
        self.gamma = gamma

        self.exploration_policy = exploration_policy
        self.exploration_param = exploration_param
        
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
            if self.exploration_param is None:
                raise KeyError("Provide an epsilon")

            values = self.network.forward(observation, grad=False)
            random_num = np.random.random()
            if random_num < self.exploration_param:
                a = np.random.randint(self.env.env.action_space.n)
            else:
                a = np.argmax(values.numpy())

        elif policy == "boltzmann":
            if self.exploration_param is None:
                raise KeyError("Provide a temperature")
            
            values = self.network.forward(observation, grad=False)
            probabilities = np.exp(values.numpy() / self.exploration_param)
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
        
        if self.use_target_network:
            next_q_values = self.target_network.forward(next_states, grad=False).numpy()
        else:
            next_q_values = self.network.forward(next_states, grad=False).numpy()

        target[np.arange(len(target)), actions] = rewards + (1 - done) * self.gamma * np.max(next_q_values, axis=1)

        target = torch.tensor(target)
        return self.network.backward(states, target)

    def eval(self):
        rewards = []
        for _ in range(self.eval_episodes):
            env = Environment()
            state, _ = env.reset_env()
            done = False
            truncated = False
            total_reward = 0
            while not done and not truncated:
                action = self.select_action(state, policy="greedy")
                next_state, reward, done, truncated, _ = env.env.step(action)
                total_reward += reward
                state = next_state

            rewards.append(total_reward)

        return np.mean(rewards)

    def run(self, test=False):
        avg_rewards = []
        self.state, _ = self.env.reset_env()
        pbar = tqdm(range(self.training_iterations)) if test else range(self.training_iterations)
        for i in pbar:
            # select action and take a step
            action = self.select_action(self.state, self.exploration_policy)
            next_state, reward, done, truncated, _ = self.env.env.step(action)

            # reset if done
            if done or truncated:
                self.state, _ = self.env.reset_env()

            if self.use_experience_replay:
                # append to buffer
                self.buffer.append((self.state, action, reward, done))
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
                self.update_network(self.state, action, reward, done, next_state)
                self.state = next_state

            # clone the network to the target network
            if self.use_target_network and i % self.target_update == 0:
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
            plt.plot(savgol_filter(avg_rewards, window_length=9, polyorder=2)) # TODO make smoothing a parameter
            plt.show()
            
            self.env = Environment(render=True)
            self.state, _ = self.env.reset_env()
            done = False
            truncated = False
            while not done and not truncated:
                action = self.select_action(self.state, policy="greedy")
                next_state, reward, done, truncated, _ = self.env.env.step(action)
                self.state = next_state

        return avg_rewards[-1]

def objective(params):
    avg_reward = 0
    runs = 5
    for _ in range(runs):
        avg_reward += dqn_agent(
            training_iterations=20_000,
            learning_rate=params['learning_rate'],
            gamma=0.99,
            num_hidden_layers=int(params['num_hidden_layers']),
            hidden_layer_size=int(params['hidden_layer_size']),
            
            exploration_policy="egreedy",
            exploration_param=params['epsilon'],
            
            eval_episodes=10,
            eval_frequency=100, # not used for optimization
            early_stopping=500, # not used for optimization

            use_experience_replay=True, # cmd line option
            batch_size=int(params['batch_size']),
            replay_buffer_size=int(params['replay_buffer_size']),
            
            use_target_network=True, # cmd line option
            target_update=params['target_update']
        ).run() / runs
    
    print("\t".join([f"{k}: {round(v,5)}" for k, v in params.items()])+f"\tavg_reward: {round(avg_reward, 5)}")
    
    return -avg_reward

# cmd line arg
HYPEROPT = False

if __name__ == "__main__":
    
    if HYPEROPT:
        space = {
            'learning_rate': hp.qloguniform('learning_rate', np.log(1e-5), np.log(1e-1), 1e-5),
            'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]),
            'hidden_layer_size': hp.qloguniform('hidden_layer_size', np.log(4), np.log(256), 4),
            'epsilon': hp.quniform('epsilon', 0.05, 0.5, 0.05),
            'batch_size': hp.qloguniform('batch_size', np.log(16), np.log(128), 16),
            'target_update': hp.qloguniform('target_update', np.log(5), np.log(1000), 5),
            'replay_buffer_size': hp.qloguniform('replay_buffer_size', np.log(1e3), np.log(1e9), 1e3)
        }

        best = space_eval(space, fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=5), max_evals=50))
        print("\t".join([f"{k}: {round(v,5)}" for k, v in best.items()]))
    
    else:
        # NOTE these are manually/rounded values with the optimized hyperparameters as starting point
        # the result from tuning was: 
        # batch_size:           80.0
        # epsilon:              0.35 
        # hidden_layer_size:    88.0
        # learning_rate:        0.04646
        # num_hidden_layers:    1
        # replay_buffer_size:   875000.0
        # target_update:        15.0
        # avg_reward:           207.66

        best = { 
            'learning_rate': 0.05,
            'num_hidden_layers': 1,
            'hidden_layer_size': 64,
            'epsilon': 0.35,
            'batch_size': 128,
            'target_update': 15,
            'replay_buffer_size': 1_000_000
        }
    
    agent = dqn_agent(
        training_iterations=20_000,
        learning_rate=best['learning_rate'],
        gamma=0.99,
        num_hidden_layers=int(best['num_hidden_layers']),
        hidden_layer_size=int(best['hidden_layer_size']),
        
        exploration_policy="egreedy",
        exploration_param=best['epsilon'],
        
        eval_episodes=10,
        eval_frequency=100,
        early_stopping=500,

        use_experience_replay=True, # cmd line option
        batch_size=int(best['batch_size']),
        replay_buffer_size=int(best['replay_buffer_size']),
        
        use_target_network=True, # cmd line option
        target_update=best['target_update']
    )
    agent.run(test=True)
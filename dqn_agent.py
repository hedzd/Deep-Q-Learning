from environment import Environment
from collections import deque
from q_network import q_network
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    ):
        self.env = Environment()

        self.training_iterations = training_iterations
        self.gamma = gamma

        self.exploration_policy = exploration_policy
        self.epsilon = epsilon
        
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
            raise NotImplementedError("Boltzmann policy not implemented yet")  # TODO

        elif policy == "greedy":
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
            target[action] = reward + self.gamma * np.max(next_q_values.numpy())

        return self.network.backward(state, target)

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

    def run(self):
        avg_rewards = []
        self.state, info = self.env.reset_env()
        pbar = tqdm(range(self.training_iterations))
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

                if i > self.batch_size:
                    # train with random samples from the buffer
                    batch_indices = np.random.choice(
                        len(self.buffer) - 1, self.batch_size
                    )
                    for index in batch_indices:
                        state, action, reward, done = self.buffer[index]
                        next_state, _, _, _ = self.buffer[index + 1]
                        self.update_network(state, action, reward, done, next_state)
            else:
                # train with the last state
                self.update_network(self.state, action, reward, done or truncated, next_state)
                self.state = next_state

            # clone the network to the target network
            if i % self.target_update == 0 and self.use_target_network:
                self.target_network.load_state_dict(self.network.state_dict())

            # evaluate
            if i % self.eval_frequency == 0:
                avg_rewards.append(self.eval())
                pbar.set_description(f"Average reward {avg_rewards[-1]}")
                if avg_rewards[-1] > self.early_stopping:
                    print(f"Training completed in {i} iterations")
                    break

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

if __name__ == "__main__":
    agent = dqn_agent(
        training_iterations=5000,
        learning_rate=1e-2,
        gamma=0.99,
        num_hidden_layers=1,
        hidden_layer_size=32,
        
        exploration_policy="egreedy",
        epsilon=0.3,
        
        eval_episodes=10,
        eval_frequency=50,
        early_stopping=400,

        use_experience_replay=True,
        batch_size=64,
        replay_buffer_size=1_000_000,
        
        use_target_network=True,
        target_update=10,
    )
    agent.run()

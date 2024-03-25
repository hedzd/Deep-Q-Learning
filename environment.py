import gymnasium as gym


class Environment:
    def __init__(self, environment="CartPole-v1", render=False):
        if render:
            self.env = gym.make(environment, render_mode="human")
        else:
            self.env = gym.make(environment)

    def reset_env(self):
        observation, info = self.env.reset()
        return observation, info

    def simulate_env(self):
        observation, info = self.env.reset()

        for _ in range(1000):
            action = (
                self.env.action_space.sample()
            )  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()


if __name__ == "__main__":
    env = Environment(render=True)
    env.simulate_env()

import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, space_eval
from functools import partial
from scipy.signal import savgol_filter
import argparse
from tqdm import tqdm

from dqn_agent import dqn_agent

def plot_rewards(avg_rewards, x_multipl=1, labels=None, title="", window_length=9, polyorder=2, smooth=False, filename=None):
    
    fig, ax = plt.subplots()
    
    episodes = np.arange(0, avg_rewards.shape[-1] * x_multipl, x_multipl)

    if smooth:
        avg_rewards = savgol_filter(avg_rewards, window_length=window_length, polyorder=polyorder)
        
    if avg_rewards.ndim == 2:
        episodes = np.tile(episodes, (avg_rewards.shape[0], 1))
    
        for i in range(avg_rewards.shape[0]):
            if labels:
                ax.plot(episodes[i], avg_rewards[i], label=labels[i])
            else:
                ax.plot(episodes[i], avg_rewards[i])
    
    else:
        if labels:
            ax.plot(episodes, avg_rewards, label=labels)
        else:
            ax.plot(episodes, avg_rewards)
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    
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
            
            exploration_policy=param['exploration_policy'],
            exploration_param=params['exploration_parameter'],
            exploration_decay=params['exploration_decay'],
            
            eval_episodes=10,
            eval_frequency=100, # not used for optimization

            use_experience_replay=True, # cmd line option
            batch_size=int(params['batch_size']),
            replay_buffer_size=int(params['replay_buffer_size']),
            
            use_target_network=True, # cmd line option
            target_update=params['target_update']
        ).run()[-1] / runs
    
    print("\t".join([f"{k}: {round(v,5)}" for k, v in params.items()])+f"\tavg_reward: {round(avg_reward, 5)}")
    
    return -avg_reward

def hyperparameter_tuning():
    space = {
        'learning_rate': hp.qloguniform('learning_rate', np.log(1e-5), np.log(1e-1), 1e-5),
        'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]),
        'hidden_layer_size': hp.qloguniform('hidden_layer_size', np.log(4), np.log(256), 4),
        'batch_size': hp.qloguniform('batch_size', np.log(16), np.log(128), 16),
        'target_update': hp.qloguniform('target_update', np.log(5), np.log(1000), 5),
        'replay_buffer_size': hp.qloguniform('replay_buffer_size', np.log(1e3), np.log(1e9), 1e3),
        'exploration_policy': 'egreedy',
        'exploration_parameter': hp.quniform('exploration_parameter', 0.05, 0.5, 0.05),
        'exploration_decay': 1.0,
    }

    best = space_eval(space, fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=5), max_evals=50))
    print("\t".join([f"{k}: {round(v,5)}" for k, v in best.items()]))

    return best

def experiment(param, runs=5):
    avg_rewards = []
    for _ in tqdm(range(runs)):
        avg_rewards.append(dqn_agent(
            training_iterations=20_000,
            learning_rate=param['learning_rate'],
            gamma=0.99,
            num_hidden_layers=int(param['num_hidden_layers']),
            hidden_layer_size=int(param['hidden_layer_size']),
            
            exploration_policy=param['exploration_policy'],
            exploration_param=param['exploration_parameter'],
            exploration_decay=param['exploration_decay'],
            
            eval_episodes=10,
            eval_frequency=50,

            use_experience_replay=True,
            batch_size=int(param['batch_size']),
            replay_buffer_size=int(param['replay_buffer_size']),
            
            use_target_network=True,
            target_update=param['target_update']
        ).run())

    return np.mean(np.array(avg_rewards), axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-er', '--experience_replay', action='store_true')
    parser.add_argument('-tn', '--target_network', action='store_true')
    parser.add_argument('-hpo', '--hyperopt', action='store_true')
    parser.add_argument('-e', '--experiment', action='store_true')
    args = parser.parse_args()

    print("Experience Replay: ", args.experience_replay)
    print("Target Network: ", args.target_network)
    print("Hyperopt: ", args.hyperopt)
    print("Experiment: ", args.experiment)
    
    if args.hyperopt:
        param = hyperparameter_tuning()
    
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

        param = { 
            'learning_rate': 0.05,
            'num_hidden_layers': 1,
            'hidden_layer_size': 64,
            'batch_size': 128,
            'target_update': 15,
            'replay_buffer_size': 1_000_000,
            
            'exploration_policy': 'egreedy',
            'exploration_parameter': 0.1,
            'exploration_decay': 1.0,
        }
    
    if args.experiment:
        
        # hidden layers experiment
        avg_rewards = []
        labels = []
        for layers in [1, 2, 3]:
            param['num_hidden_layers'] = layers
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/hidden_layers_{layers}.txt", avg_rewards[-1])
            labels.append(f"{layers} hidden layers")
        param['num_hidden_layers'] = 1
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Comparison Hidden Layer Amount", window_length=9, polyorder=2, smooth=True, filename="experiments/num_hidden_layers.png")

        # hidden layer size experiment
        avg_rewards = []
        labels = []
        for nodes in [16, 32, 64, 128]:
            param['hidden_layer_size'] = nodes
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/hidden_layer_size_{nodes}.txt", avg_rewards[-1])
            labels.append(f"{nodes} nodes")
        param['hidden_layer_size'] = 64
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Comparison Hidden Layer Amount", window_length=9, polyorder=2, smooth=True, filename="experiments/hidden_layer_size.png")

    else:
        agent = dqn_agent(
            training_iterations=20_000,
            learning_rate=param['learning_rate'],
            gamma=0.99,
            num_hidden_layers=int(param['num_hidden_layers']),
            hidden_layer_size=int(param['hidden_layer_size']),
            
            exploration_policy=param['exploration_policy'],
            exploration_param=param['exploration_parameter'],
            exploration_decay=param['exploration_decay'],
            
            eval_episodes=10,
            eval_frequency=100,

            use_experience_replay=args.experience_replay,
            batch_size=int(param['batch_size']),
            replay_buffer_size=int(param['replay_buffer_size']),
            
            use_target_network=args.target_network,
            target_update=param['target_update'],

            environment="CartPole-v1"
        )
        avg_rewards = agent.run(test=True)
        plot_rewards(np.array(avg_rewards), x_multipl=100, window_length=9, polyorder=2, smooth=True)
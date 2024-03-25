import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, space_eval
from functools import partial
from scipy.signal import savgol_filter
import argparse

from dqn_agent import dqn_agent

def plot_rewards(avg_rewards, x_multipl=1, window_length=9, polyorder=2, smooth=False):
    episodes = np.arange(0, len(avg_rewards) * x_multipl, x_multipl)
    if smooth:
        plt.plot(episodes, savgol_filter(avg_rewards, window_length=window_length, polyorder=polyorder))
    else:
        plt.plot(episodes, avg_rewards)
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
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
            
            exploration_policy="egreedy",
            exploration_param=params['epsilon'],
            
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
        'epsilon': hp.quniform('epsilon', 0.05, 0.5, 0.05),
        'batch_size': hp.qloguniform('batch_size', np.log(16), np.log(128), 16),
        'target_update': hp.qloguniform('target_update', np.log(5), np.log(1000), 5),
        'replay_buffer_size': hp.qloguniform('replay_buffer_size', np.log(1e3), np.log(1e9), 1e3)
    }

    best = space_eval(space, fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=5), max_evals=50))
    print("\t".join([f"{k}: {round(v,5)}" for k, v in best.items()]))

    return best

def experiment(param, runs=20):
    avg_rewards = []
    for _ in range(runs):
        avg_rewards.append(dqn_agent(
            training_iterations=20_000,
            learning_rate=param['learning_rate'],
            gamma=0.99,
            num_hidden_layers=int(param['num_hidden_layers']),
            hidden_layer_size=int(param['hidden_layer_size']),
            
            exploration_policy="egreedy",
            exploration_param=param['epsilon'],
            
            eval_episodes=10,
            eval_frequency=50,

            use_experience_replay=True,
            batch_size=int(param['batch_size']),
            replay_buffer_size=int(param['replay_buffer_size']),
            
            use_target_network=True,
            target_update=param['target_update']
        ).run())

    avg_rewards = np.mean(np.array(avg_rewards), axis=0)
    plot_rewards(avg_rewards, x_multipl=50, window_length=9, polyorder=2, smooth=True)    

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
            'epsilon': 0.35,
            'batch_size': 128,
            'target_update': 15,
            'replay_buffer_size': 1_000_000
        }
    
    if args.experiment:
        experiment(param)

    else:
        agent = dqn_agent(
            training_iterations=20_000,
            learning_rate=param['learning_rate'],
            gamma=0.99,
            num_hidden_layers=int(param['num_hidden_layers']),
            hidden_layer_size=int(param['hidden_layer_size']),
            
            exploration_policy="egreedy",
            exploration_param=param['epsilon'],
            
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
        plot_rewards(avg_rewards, x_multipl=100, window_length=9, polyorder=2, smooth=True)
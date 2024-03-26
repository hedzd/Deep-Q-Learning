import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp, fmin, tpe, space_eval
from functools import partial
from scipy.signal import savgol_filter
import argparse
from tqdm import tqdm

from dqn_agent import dqn_agent

def plot_rewards(avg_rewards, x_multipl=1, labels=None, title="", window_length=18, polyorder=2, smooth=True, filename=None):
    
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
    
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title(title)
    if labels:
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
            
            exploration_policy=params['exploration_policy'],
            exploration_param=params['exploration_parameter'],
            exploration_decay=params['exploration_decay'],
            
            eval_episodes=10,
            eval_frequency=100, # not used for optimization

            use_experience_replay=True, # cmd line option
            batch_size=int(params['batch_size']),
            replay_buffer_size=int(params['replay_buffer_size']),
            
            use_target_network=True, # cmd line option
            target_update=params['target_update'],

            environment=params['environment']
        ).run()[-1] / runs
    
    print("\t".join([f"{k}: {round(v,5)}" for k, v in params.items() if type(v) != str])+f"\tavg_reward: {round(avg_reward, 5)}")
    
    return -avg_reward

def hyperparameter_tuning(environment="CartPole-v1"):
    space = {
        'learning_rate': hp.qloguniform('learning_rate', np.log(1e-5), np.log(1e-1), 1e-5),
        'num_hidden_layers': hp.choice('num_hidden_layers', [1, 2, 3]),
        'hidden_layer_size': hp.qloguniform('hidden_layer_size', np.log(4), np.log(256), 4),
        'batch_size': hp.qloguniform('batch_size', np.log(16), np.log(128), 16),
        'target_update': hp.qloguniform('target_update', np.log(5), np.log(1000), 5),
        'replay_buffer_size': hp.qloguniform('replay_buffer_size', np.log(1e3), np.log(1e9), 1e3),
        'exploration_policy': 'boltzmann',
        'exploration_parameter': hp.qloguniform('exploration_parameter', np.log(0.01), np.log(10.0), 0.01),
        'exploration_decay': 1.0,
        'environment': environment
    }

    best = space_eval(space, fmin(objective, space, algo=partial(tpe.suggest, n_startup_jobs=5), max_evals=50))
    print("\t".join([f"{k}: {round(v,5)}" for k, v in best.items() if type(v) != str]))

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
    parser.add_argument('-env', '--environment', type=str, default="CartPole-v1")
    args = parser.parse_args()

    print("Experience Replay: ", args.experience_replay)
    print("Target Network: ", args.target_network)
    print("Hyperopt: ", args.hyperopt)
    print("Experiment: ", args.experiment)
    print("Environment: ", args.environment)
    
    if args.hyperopt:
        param = hyperparameter_tuning(args.environment)

        # Hyperparameter tuning results for cartpole-v1 using egreedy:
        # batch_size:           80.0
        # epsilon:              0.35 
        # hidden_layer_size:    88.0
        # learning_rate:        0.04646
        # num_hidden_layers:    1
        # replay_buffer_size:   875000.0
        # target_update:        15.0
        # avg_reward:           207.66

        # Hyperparameter tuning results for cartpole-v1 using boltzmann:
        # batch_size:           80.0
        # temperature:          3.43    
        # hidden_layer_size:    64
        # learning_rate:        0.09881 
        # num_hidden_layers:    1
        # replay_buffer_size:   122000
        # target_update:        70
        # avg_reward:           182.14
    
    else: # use manually tuned parameters

        if args.environment == "CartPole-v1":

            # further manually tuned parameters for cartpole-v1
            param = { 
                'learning_rate': 0.1,
                'num_hidden_layers': 1,
                'hidden_layer_size': 128,
                'batch_size': 128,
                'target_update': 10,
                'replay_buffer_size': 1_000_000,
                
                'exploration_policy': 'egreedy', # egreedy or boltzmann
                'exploration_parameter': 0.1, # 0.1 for egreedy, 1.0 for boltzmann
                'exploration_decay': 1.0,
            }

        elif args.environment == "Acrobot-v1":
            param = {
                'learning_rate': 1e-4,
                'num_hidden_layers': 1,
                'hidden_layer_size': 16,
                'batch_size': 80,
                'target_update': 600,
                'replay_buffer_size': 15_000,
                
                'exploration_policy': 'boltzmann',
                'exploration_parameter': 1.0,
                'exploration_decay': 1.0,
            }

        else:
            raise ValueError("Invalid environment. Choose from ['CartPole-v1', 'Acrobot-v1']")
    
    if args.experiment:        
        
        # hidden layers experiment
        avg_rewards = []
        labels = []
        for layers in [1, 2, 3]:
            param['num_hidden_layers'] = layers
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/hidden_layers_{layers}.txt", avg_rewards[-1])
            if layers == 1:
                labels.append(f"{layers} hidden layer")
            else:
                labels.append(f"{layers} hidden layers")
        param['num_hidden_layers'] = 1
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Comparison Hidden Layer Amount", filename="experiments/num_hidden_layers.png")
        
        # hidden layer size experiment
        avg_rewards = []
        labels = []
        for nodes in [16, 32, 64, 128]:
            param['hidden_layer_size'] = nodes
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/hidden_layer_size_{nodes}.txt", avg_rewards[-1])
            labels.append(f"{nodes} nodes")
        param['hidden_layer_size'] = 128

        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Comparison Hidden Layer Size", filename="experiments/hidden_layer_size.png")
        
        # learning rate experiment
        avg_rewards = []
        labels = []
        for lr in [0.5, 0.1, 0.05, 0.01]:
            param['learning_rate'] = lr
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/lr_{lr}.txt", avg_rewards[-1])
            labels.append(r"$\alpha$={}".format(lr))
        param['learning_rate'] = 0.1
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Learning Rate Comparison", filename="experiments/lr.png")
        
        # batch size experiment
        avg_rewards = []
        labels = []
        for batch_size in [16, 32, 64, 128]:
            param['batch_size'] = batch_size
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/batch_size_{batch_size}.txt", avg_rewards[-1])
            labels.append(f"batch size={batch_size}")
        param['batch_size'] = 128
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Batch Size Comparison", filename="experiments/batch_size.png")

        # buffer size experiment
        avg_rewards = []
        labels = []
        for buffer_size in [1e3, 1e4, 1e5, 1e6]:
            param['replay_buffer_size'] = buffer_size
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/buffer_size_{buffer_size}.txt", avg_rewards[-1])
            labels.append(f"buffer size={buffer_size}")
        param['replay_buffer_size'] = 1e5
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Buffer Size Comparison", filename="experiments/buffer_size.png")

        # target update experiment
        avg_rewards = []
        labels = []
        for target_update in [10, 20, 50, 100, 1000]:
            param['target_update'] = target_update
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/target_update_{target_update}.txt", avg_rewards[-1])
            labels.append(f"target update freq={target_update}")
        param['target_update'] = 20
            
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Target Update Frequency Comparison", filename="experiments/target_update.png")
       
        # # e-greedy exploration parameter experiment
        # avg_rewards = []
        # labels = []
        # param['exploration_policy'] = 'egreedy'
        # for exploration_param in [0.01, 0.03, 0.05, 0.1, 0.3]:
        #     param['exploration_parameter'] = exploration_param
        #     avg_rewards.append(experiment(param))
        #     np.savetxt(f"experiments/results/exploration_param_{exploration_param}.txt", avg_rewards[-1])
        #     labels.append(rf"$\epsilon$ = {exploration_param}")

        # plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title=r"$\epsilon$-greedy Exploration Parameter Comparison", filename="experiments/e-greedy.png")
        
        # # boltzmann exploration parameter experiment
        # avg_rewards = []
        # labels = []
        # param['exploration_policy'] = 'boltzmann'
        # for exploration_param in [0.01, 0.1, 0.5, 0.7, 1.0]:
        #     param['exploration_parameter'] = exploration_param
        #     avg_rewards.append(experiment(param))
        #     np.savetxt(f"experiments/results/exploration_param_{exploration_param}.txt", avg_rewards[-1])
        #     labels.append(rf'$\tau$ = {exploration_param}')

        # plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Boltzmann Exploration Parameter Comparison", filename="experiments/boltzmann.png")

        # exploration method experiment
        avg_rewards = []
        labels = []
        param['exploration_policy'] = 'egreedy'
        for exploration_param in [0.03,0.1,0.3]:
            param['exploration_parameter'] = exploration_param
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/exploration_method_egreedy_exploration_param_{exploration_param}.txt", avg_rewards[-1])
            labels.append(rf'$\epsilon$-greedy, $\epsilon $ = {exploration_param}')
            
            print(labels[-1])

        param['exploration_policy'] = 'boltzmann'
        for exploration_param in [0.1,0.5,1.0]:
            param['exploration_parameter'] = exploration_param
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/exploration_method_boltzmann_exploration_param_{exploration_param}.txt", avg_rewards[-1])
            labels.append(rf'Softmax, $\tau$ = {exploration_param}')
        
        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Exploration Method Comparison", filename="experiments/exploration_method.png")        

        # exploration decay experiment
        avg_rewards = []
        labels = []
        param['exploration_policy'] = 'egreedy'
        param['exploration_parameter'] = 0.1
        for exploration_decay in [0.95, 0.99, 1.0]:
            param['exploration_decay'] = exploration_decay
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/exploration_decay_egreedy_{exploration_decay}.txt", avg_rewards[-1])
            labels.append(rf'$\epsilon$-greedy, decay = {exploration_decay}')

        param['exploration_policy'] = 'boltzmann'
        param['exploration_parameter'] = 1.0
        for exploration_decay in [0.95, 0.99, 1.0]:
            param['exploration_decay'] = exploration_decay
            avg_rewards.append(experiment(param))
            np.savetxt(f"experiments/results/exploration_decay_boltzman_{exploration_decay}.txt", avg_rewards[-1])
            labels.append(rf'Softmax, decay = {exploration_decay}')

        plot_rewards(np.array(avg_rewards), x_multipl=50, labels=labels, title="Exploration Decay Comparison", filename="experiments/exploration_decay.png")
    
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
            eval_frequency=50,

            use_experience_replay=args.experience_replay,
            batch_size=int(param['batch_size']),
            replay_buffer_size=int(param['replay_buffer_size']),
            
            use_target_network=args.target_network,
            target_update=param['target_update'],

            environment=args.environment
        )
        avg_rewards = agent.run(test=True)
        plot_rewards(np.array(avg_rewards), x_multipl=50, smooth=True)
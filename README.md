# Deep Q Learning

This project explores the implementation of Deep Q-Networks (DQN) for the Cartpole environment, including experiments for finetuning different hyperparameters, and an ablation study
investigating the impact of experience replay and target network mechanisms on performance of this algorithm.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing

A step by step series of examples that tell you how to get a development environment running:

```sh
pip install -r requirements.txt
```

## How to run our code 
Reproduce our experimental results by utilizing the following command-line options. After the full execution of the code, plots and results will be saved automatically in the project directory.
### Running the Experiments for Hyperparameters

To execute experiments with various hyperparameters, use the `-e` option followed by the specific experiment flag:

- `-hl`, `--hidden_layers`: Run a hidden layers experiment
- `-hs`, `--hidden_size`: Run a hidden size experiment
- `-lr`, `--learning_rate`: Run a learning rate experiment
- `-bs`, `--batch_size`: Run a batch size experiment
- `-bu`, `--buffer_size`: Run a buffer size experiment
- `-tu`, `--target_update`: Run a target update experiment

For instance, to run an experiment that explores the effect of varying the number of hidden layers, you would use the following command:
```sh
python main.py -e -hl
```
### Running Experiments for Exploration Strategies

To replicate our exploration strategy experiments, including those with and without annealing, you can run the following command:

```sh
python main.py -exp
```
### Running Ablation Study

You can rerun our ablation study using the following command:
```sh
python main.py -a
```
### Running the TPE Algorithm for Automatic Hyperparameter Optimization
To perform automatic hyperparameter optimization using the TPE algorithm, use:
```sh
python main.py -hpo
```
The script will explore the hyperparameter space to find the most effective values that lead to optimal performance.

### Training the agent with optimized hyperparameters values with simulation of the environment
To run the agent with optimized hyperparameters values, use the following command:
```sh
python main.py
```
By default, this uses the optimized value for hyperparameters that we mentioned in our report for Cartpole environment.

### Changing the Environment
To apply the previous commands to different environments such as 'Acrobot-v1', 'MountainCar-v0', or 'LunarLander-v2', specify the environment using the -env flag like so:
```sh
python main.py -env 'Acrobot-v1'
```


# Repo for Udacity Deep Reinforcement Learning Nanodegree

## I. Intro
This repository is destined to contain a summary of my work towards graduating from the Udacity's Deep RL Nanodegree. It will be structured based on the degree's syllabus and will address different methods used to solve Reinforcement Learning problems.

Right now, there are two algorithms available:
- DQN (Deep Q-Learning) 
  - I implemented and calibrated my algo while solving the OpenAI Gym environement "Lunar-Lander" (note that it's heavily inspired by the implementation presented in the course)
  - I used it to solve the degree's first project, labelled "Navigation"
- DDPG (Deep Deterministic Policy Gradient):
  - I implemented and calibrated my algo while solving the OpenAI Gym environment "Pendulum-v0" (note that it's heavily inspired by the implementation presented in the course)
  - I used it to solved the degree's second project, labelled "Continuous Control"

## II. Dependencies
The content of this repository will be heavily dependent on the below git repos:
- Udacity Deep RL Nanodegree: https://github.com/udacity/deep-reinforcement-learning
- OpenAI Gym project: https://github.com/openai/gym
- Unity: https://github.com/Unity-Technologies/ml-agents

Also, please note that my repo is installed and used wihtin an Anaconda environment.

## III. Installation
To be able to run my repo, you need to follow the below steps:
```
conda create --name myuda python=3.6  
activate myuda  
pip install unityagents  
conda install -c pytorch pytorch
conda install pandas (for results analysis)
```

## IV. Content
### IV.1 Use the training environment
If not already done, follow the [installation steps](#installation)
Then clone the repo and `cd UdacityRL`

My goal is to have:
- a set of different agents such as the dqn_agent present here
- a set of different agent_trainer classes such as the one present here
so that one is able to train/test any agent within several Gym or Unity environment

For now, are available:
- the DQN algo in interaction with the Unity Banana Environment
- the DDPG algo in interaction with the Unity Reacher Environment

### IV.2 DQN algo
#### IV.2.a Implementation and calibration
The implementation of the DQN algo was heavily inspired from the Udacity course.
It implements the below features:
- Simple DQN algo
- Double DQN algo
- Simple DQN algo with priority sampling

The whole DQN algo can be found in the file dqn_agent.py
The constructor to use if you want to instantiate an agent takes the below params:
``` python
state_size (int): dimension of each state
action_size (int): dimension of each action
layers_params ([int]): sizes of hidden fc layers in NN
double_dqn (bool): enable double-dqn logic
step_reward (int): extra reward to be added to environment reward
soft_tau (float): rate at which target NN converges towards local NN
learning_rate (float): optimizer learning rate
memory_prio_params (bool, float, float, float): (memory_prio_enabled, memory_prio_a, memory_prio_b0, memory_prio_b_step)
debug_mode (bool): enable debug logging
time_analysis (TimeAnalysis): contains timers stats
```

An example would be:
```python
agent = dqn_agent.Agent(
    state_size=37, 
    action_size=4, 
    layers_params=[64,64,64], 
    double_dqn=False, 
    step_reward=0,
    soft_tau=1e-3,
    learning_rate=5e-4,
    memory_prio_params=(False, 0, 0, 0),
    debug_mode=False,
    time_analysis=None
)
```

#### IV.2.b Udacity Navigation project
##### Environment description
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

##### Solution implementation
The main file to use is navigation.py:
```
usage: navigation.py [-h] [--test_model TEST_MODEL]
                     [--test_params TEST_PARAMS] [--train_params TRAIN_PARAMS]
                     [--train_start_id TRAIN_START_ID]
                     [--train_results_path TRAIN_RESULTS_PATH]
                     [--train_debug TRAIN_DEBUG]
                     mode

Parameters for navigation project

positional arguments:
  mode

optional arguments:
  -h, --help            show this help message and exit
  --test_model TEST_MODEL
  --test_params TEST_PARAMS
  --train_params TRAIN_PARAMS
  --train_start_id TRAIN_START_ID
  --train_results_path TRAIN_RESULTS_PATH
  --train_debug TRAIN_DEBUG
```

##### Example of training
You can create a list of training configurations in json format like I did in ./Navigation/Params/training_params.json:
```
python ./Navigation/navigation.py 'train' --train_params='training_params.json'
```

All results will be stored in the file ./Navigation/Results/results.csv and all models that solve the environment will be saved in the directory ./Navigation/ModelWeights

##### Example of testing
You can test saved model weights (together with a corresponding agent params) against the environment:
```
python ./Navigation/navigation.py 'test' --test_params='default_params.json' --test_model='simple_[64, 64, 64]_0.994' --test_results_path=''
```

##### Results
My training results can be analyzed using the simple library ./Navigation/Results/results_analysis.py

You can find a report of my results in the Jupyter notebook ./Navigation/Results/Results_report.ipynb
- [Jupyter pdf view](./Navigation/Results/Results_report.pdf) 
- [Video of my model tested in the environment](./Navigation/Results/best_model.mp4))

### IV.3 DDPG algo
#### IV.3.a Implementation and calibration
The implementation of the DDPG algo was heavily inspired from the Udacity course.
It implements the below features:
- Ornstein-Uhlenbeck process to improve exploration

The whole DDPG algo can be found in the file ddpg_agent.py
The constructor to use if you want to instantiate an agent takes the below params:
``` python
state_size (int): dimension of each state
action_size (int): dimension of each action
actor_layers_params (int, int): number of nodes for each hidden layer of actor NN
critic_layers_params (int, int): number of nodes for each hidden layer of critic NN
soft_tau (float): soft-update parameter for target networks
batch_size (int): number of steps sampled at each learning step
learning_rate_actor (float): learning rate for actor NN
learning_rate_critic (float): learning rate for critic NN
weight_decay_critic (float): weight decay for the critic NN
noise_params (float, float): (theta, sigma) values for the Ornstein-Uhlenbeck process
learn_step_nb (int): number of steps between each learning phase
debug_mode (bool)
time_analysis (TimeAnalysis): used to profile agent
```

An example would be:
```python
agent = ddpg_agent.Agent(
    state_size=33, 
    action_size=4, 
    actor_layers_params=(128, 128), 
    critic_layers_params=(128, 128),
    soft_tau=1e-3,
    batch_size=100,
    learning_rate_actor=1e-3,
    learning_rate_critic=1e-3,
    weight_decay_critic=0,
    noise_params=(0.15, 0.2),
    learn_step_nb=4,
    debug_mode=False,
    time_analysis=None
)
```

#### IV.3.b Udacity Continuous Control project
##### Environment description
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

##### Solution implementation
The main file to use is continuous.py:
```
usage: continuous_control.py [-h] [--test_model TEST_MODEL]
                             [--test_params TEST_PARAMS] [--test_no_display]
                             [--test_results_path TEST_RESULTS_PATH]
                             [--test_episodes TEST_EPISODES]
                             [--train_params TRAIN_PARAMS]
                             [--train_start_id TRAIN_START_ID]
                             [--train_results_path TRAIN_RESULTS_PATH]
                             [--train_debug TRAIN_DEBUG]
                             [--train_worker_id TRAIN_WORKER_ID]
                             mode

Parameters for continuous control project

positional arguments:
  mode

optional arguments:
  -h, --help            show this help message and exit
  --test_model TEST_MODEL
  --test_params TEST_PARAMS
  --test_no_display
  --test_results_path TEST_RESULTS_PATH
  --test_episodes TEST_EPISODES
  --train_params TRAIN_PARAMS
  --train_start_id TRAIN_START_ID
  --train_results_path TRAIN_RESULTS_PATH
  --train_debug TRAIN_DEBUG
  --train_worker_id TRAIN_WORKER_ID
```

##### Example of training
You can create a list of training configurations in json format like I did in ./ContinuousControl/Params/training_params.json:
```
python ./ContinuousControl/continuous.py 'train' --train_params='training_params.json'
```

All results will be stored in the file ./ContinuousControl/Results/results.csv and all models that solve the environment will be saved in the directory ./ContinuousControl/ModelWeights

##### Example of testing
You can test saved model weights (together with a corresponding agent params) against the environment:
```
python ./ContinuousControl/continuous.py 'test' --test_params='best_params.json' --test_model="auto" --test_results_path=""
```

##### Results
My training results can be analyzed using the simple library ./ContinuousControl/Results/results_analysis.py

You can find a report of my results in the Jupyter notebook ./ContinuousControl/Results/Results_report.ipynb
- [Jupyter pdf view](./ContinuousControl/Results/Results_report.pdf) 
- [Video of my model tested in the environment](./ContinuousControl/Results/best_model.mp4))

## V. License
That repo has no license at the moment.

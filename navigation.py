#external imports
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import json
import sys

#internal imports
from dqn_agent import Agent

default_params = {
    'model_tag':'simple_dqn',
    'agent':{
        'nn_layers':[64,64,64],
        'memory_prio_params':(
            False, #memory_prio_enabled
            0., #memory_prio_a
            0., #memory_prio_b0
            0. #memory_prio_b_step
        ),
        'double_dqn':False, 
        'step_reward':0.,
        'debug_mode':False,
        'soft_tau':1e-3,
        'learning_rate':5e-4,
    },
    'training':{
        'n_episodes':1000,
        'max_t':1000,
        'max_score':14,
        'eps_start':1.,
        'eps_end':0.01,
        'eps_decay':0.995,
        'max_consec_worse':3,
        'max_total_worse':5,
    },
}

#     Params
#     ======
#         n_episodes (int): maximum number of training episodes
#         max_t (int): maximum number of timesteps per episode
#         eps_start (float): starting value of epsilon, for epsilon-greedy action selection
#         eps_end (float): minimum value of epsilon
#         eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
#     """

def train_agent(env, params, results_file):
    #Environments contain brains which are responsible for deciding the actions of their 
    # associated agents. Here we check for the first brain available, and set it as the default 
    # brain we will be controlling from Python.
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment and print details
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)
    
    print('Number of agents:', len(env_info.agents))
    print('Number of actions:', action_size)
    print('States look like:', state)
    print('obs', env_info.vector_observations)
    print('States have length:', state_size)

    #create agent
    agent = Agent(
        state_size=state_size, 
        action_size=action_size,
        layers_params=params['agent']['nn_layers'],
        double_dqn=params['agent']['double_dqn'],
        step_reward=params['agent']['step_reward'],
        soft_tau=params['agent']['soft_tau'],
        learning_rate=params['agent']['learning_rate'],
        memory_prio_params=(
            params['agent']['memory_prio_params']['memory_prio_enabled'],
            params['agent']['memory_prio_params']['memory_prio_a'],
            params['agent']['memory_prio_params']['memory_prio_b0'],
            params['agent']['memory_prio_params']['memory_prio_b_step']
        ),
        debug_mode=params['agent']['debug_mode'])

    #train agent while interacting with environment
    scores = []
    scores_window = deque(maxlen=100)
    eps = params['training']['eps_start']
    consecutive_worse_count = 0
    total_worse_count = 0
    for i_episode in range(1, params['training']['n_episodes']+1):
        
        # reset the environment and get current state
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        
        #train agent with another episode
        score = 0
        for t in range(params['training']['max_t']):
            # total_step_t = datetime.datetime.now()
            action = agent.act(state, eps).astype(int)
            
            # env_step_t = datetime.datetime.now()
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # agent.time_analysis.env_step.append(datetime.datetime.now()-env_step_t)
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            # agent.time_analysis.total_step.append(datetime.datetime.now()-total_step_t)
            if done:
                break 

        #post-processing of latest episode
        scores_window.append(score)
        scores.append(score)
        eps = max(params['training']['eps_end'], params['training']['eps_decay']*eps)
        
        print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.5f}'.format(i_episode, np.mean(scores_window), eps), end="")

        #post-processing of latest 100 episodes        
        if i_episode % scores_window.maxlen == 0:
            # agent.time_analysis.to_str()
            # agent.time_analysis.reset()
            
            if i_episode > scores_window.maxlen:
                if np.mean(scores_window) < prev_scores_window_avg:
                    consecutive_worse_count += 1
                    total_worse_count += 1
                else:
                    consecutive_worse_count = 0

            prev_scores_window_avg = np.mean(scores_window)
            print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.5f}'.format(i_episode, np.mean(scores_window), eps))
            results_file.write(','.join([params['model_tag'], str(i_episode), str(np.mean(scores_window))]))

        #test if agent solved the environment or failed
        #if the environment was solved, save the agent model weights
        if np.mean(scores_window)>=params['training']['max_score']:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), params['model_tag'] + '.pth')
            break
        elif total_worse_count == params['training']['max_total_worse'] or consecutive_worse_count == params['training']['max_consec_worse']:
            print('\nEnvironment failed in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            break

    #close environment
    env.close()

def test_agent(env, model='random'):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment and get current state
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    action_size = brain.vector_action_space_size
    state_size = len(state)
    score = 0
    
    agent = None
    if model != 'random':
        agent = Agent(
            state_size=state_size, 
            action_size=action_size,
            layers_params=default_params['agent']['nn_layers'],
            double_dqn=default_params['agent']['double_dqn'],
            step_reward=default_params['agent']['step_reward'],
            soft_tau=default_params['agent']['soft_tau'],
            learning_rate=default_params['agent']['learning_rate'],
            memory_prio_params=default_params['agent']['memory_prio_params'],
            debug_mode=default_params['agent']['debug_mode'])
        agent.qnetwork_local.load_state_dict(torch.load(model + '.pth'))

    while True:
        if agent:
            action = agent.act(state).astype(int)
        else:
            action = np.random.randint(action_size)

        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break
            
        print("\rScore: {}".format(score), end='')
    
def plot_scores(scores):
    # import torch
    # import numpy as np
    # import matplotlib.pyplot as plt
    # %matplotlib inline
    # is_ipython = 'inline' in plt.get_backend()
    # if is_ipython:
    #     from IPython import display
    # plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters for navigation project')
    parser.add_argument('mode')
    parser.add_argument('--test_model', dest='test_model', default='random')
    parser.add_argument('--train_params', dest='train_params', default='./training_params.json')
    parser.add_argument('--train_start_id', dest='train_start_id', default=0)
    args = parser.parse_args()
    
    #Load the Banana Unity environment
    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

    if args.mode == 'test':
        test_agent(env, model=args.test_model)
    elif args.mode == 'train':
        params_file = open(args.train_params, 'r')
        training_params = json.loads(params_file.read())
        params_file.close()

        results_file = open('results.csv', 'w')
        try:
            for training_params in training_params['training_params']:
                if(training_params['id'] < args.train_start_id):
                    continue
                print('Train model_id', training_params['id'])
                train_agent(env, training_params, results_file)
        except :
            print('Unexpected error:', sys.exc_info()[0])
        finally:
            results_file.close()
    else:
        print('Unknown mode', args.mode)
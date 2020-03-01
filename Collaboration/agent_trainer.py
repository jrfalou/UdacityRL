from collections import deque
import numpy as np
import torch

#internal imports
from mddpg_agent import Agent
from time_analysis import RunType

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentTrainer:
    '''
    AgentTrainer:
    ======
    env (UnityEnvironment): unity environment to interact with
    params (dictionary): format can be found in the ./default_params.json file
    results_path (string): file path to output training results
    debug_mode (bool): enables debug mode logging
    time_analysis (TimeAnalysis): contains timers stats
    '''
    def __init__(self, env, params, results_path='', debug_mode=False, time_analysis=None):
        #Environments contain brains which are responsible for deciding the actions of their
        # associated agents. Here we check for the first brain available, and set it as the default
        # brain we will be controlling from Python.
        self.env = env
        self.trainer_id = params['id']
        self.trainer_params = params['training'] if 'training' in params else None
        self.agent_params = params['agent']
        self.brain_name = self.env.brain_names[0]
        self.results_path = results_path
        self.debug_mode = debug_mode
        self.time_analysis = time_analysis

        # reset the environment and print details
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        state_size = len(states[0])
        action_size = self.env.brains[self.brain_name].vector_action_space_size

        assert(len(self.agent_params['agents']) == num_agents)
        assert(len(states) == num_agents)

        if self.debug_mode:
            print('Number of agents:', num_agents)
            print('Number of actions:', action_size)
            print('States look like:', states[0])
            print('obs', env_info.vector_observations)
            print('States have length:', state_size)

        # create agent
        self.agent = Agent(state_size, action_size, self.agent_params, self.time_analysis)

    def train(self):
        print('Train model_id', self.trainer_id)
        scores = []
        scores_window = deque(maxlen=100)
        # eps = self.trainer_params['eps_start']
        consecutive_worse_count = 0
        total_worse_count = 0
        for i_episode in range(1, self.trainer_params['n_episodes']+1):
            # reset the environment and get current state
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            states = env_info.vector_observations
            self.agent.reset()

            #train agent with another episode
            score = np.zeros(len(states))
            for _ in range(self.trainer_params['max_t']):
                self.time_analysis.start_timer(RunType.training_step)

                self.time_analysis.start_timer(RunType.agent_act)
                actions = self.agent.act(states)
                self.time_analysis.end_timer(RunType.agent_act)

                self.time_analysis.start_timer(RunType.env_step)
                env_info = self.env.step(actions)[self.brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                self.time_analysis.end_timer(RunType.env_step)

                self.agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                score += rewards
                done = np.any(dones)
                self.time_analysis.end_timer(RunType.training_step)
                if done:
                    break

            #post-processing of latest episode
            score = np.max(score)
            scores_window.append(score)
            scores.append(score)
            
            results_file_tmp = open(self.results_path.replace('.csv', '_step.csv'), 'a')
            results_file_tmp.write(','.join([str(self.trainer_id), self.agent_params['model_tag'].replace(',', ';'), str(i_episode), str(score)]) + '\n')
            results_file_tmp.close()

            # print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.5f}'.format(i_episode, np.mean(scores_window), eps), end="")
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

            #post-processing of latest 100 episodes        
            if i_episode % scores_window.maxlen == 0:
                if self.debug_mode:
                    print('\n' + self.time_analysis.to_str())

                if i_episode > scores_window.maxlen:
                    if np.mean(scores_window) < prev_scores_window_avg:
                        consecutive_worse_count += 1
                        total_worse_count += 1
                    else:
                        consecutive_worse_count = 0

                prev_scores_window_avg = np.mean(scores_window)
                # print('\rEpisode {}\tAverage Score: {:.2f}\teps: {:.5f}'.format(i_episode, np.mean(scores_window), eps))
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                self.write_score_to_file(i_episode=i_episode, mean_score=np.mean(scores_window))

            #test if agent solved the environment or failed
            #if the environment was solved, save the agent model weights
            if np.mean(scores_window) >= self.trainer_params['max_score']:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                self.agent.save_model()
                self.write_score_to_file(i_episode=i_episode, mean_score=np.mean(scores_window))
                break
            elif total_worse_count == self.trainer_params['max_total_worse'] or consecutive_worse_count == self.trainer_params['max_consec_worse']:
                print('\nEnvironment failed in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                break

    def write_score_to_file(self, i_episode, mean_score):
        results_file = open(self.results_path, 'a')
        results_file.write(','.join([str(self.trainer_id), self.agent_params['model_tag'].replace(',', ';'), str(i_episode), str(mean_score)]) + '\n')
        results_file.close()
                
    def test(self, model_weights='', test_id='test'):
        brain = self.env.brains[self.brain_name]

        # reset the environment and get current state
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        env_info = self.env.reset(train_mode=False)[self.brain_name]
        states = env_info.vector_observations
        action_size = brain.vector_action_space_size
        score = np.zeros(len(states))
        
        if model_weights != '' and model_weights != 'random':
            print('Load model weights', './ModelWeights/' + model_weights + '.pth')
            self.agent.load_model(model_weights)
        
        while True:
            if model_weights == 'random':
                actions = np.random.randn(len(states), action_size)
                actions = np.clip(actions, -1, 1)
            else:
                actions = self.agent.act(states)
            
            env_info = self.env.step(actions)[self.brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            states = next_states
            done = np.any(dones)
            if done:
                break
                
            print("\rScore: {}".format(round(np.max(score), 2)), end='')
        
        score = np.max(score)
        if self.results_path != '':    
            self.write_score_to_file(test_id, score)
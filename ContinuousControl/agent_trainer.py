from collections import deque
import numpy as np
import torch

#internal imports
from ddpg_agent import Agent
from time_analysis import RunType
from utils import get_layers_from_string

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
        state = env_info.vector_observations[0]
        state_size = len(state)
        action_size = self.env.brains[self.brain_name].vector_action_space_size

        if self.debug_mode:
            print('Number of agents:', len(env_info.agents))
            print('Number of actions:', action_size)
            print('States look like:', state)
            print('obs', env_info.vector_observations)
            print('States have length:', state_size)

        #create agent
        self.agent = Agent(state_size=state_size,
                            action_size=action_size,
                            actor_layers_params=
                                get_layers_from_string(self.agent_params['actor_hidden_layers']),
                            critic_layers_params=
                                get_layers_from_string(self.agent_params['critic_hidden_layers']),
                            soft_tau=self.agent_params['soft_tau'],
                            batch_size=self.agent_params['batch_size'],
                            learning_rate_actor=self.agent_params['learning_rate_actor'],
                            learning_rate_critic=self.agent_params['learning_rate_critic'],
                            weight_decay_critic=0,
                            noise_params=(
                                self.agent_params['noise_theta'],
                                self.agent_params['noise_sigma']
                            ),
                            learn_step_nb=self.agent_params['learn_step_nb'],
                            # memory_prio_params=(
                            #    self.agent_params['memory_prio_params']['memory_prio_enabled'],
                            #    self.agent_params['memory_prio_params']['memory_prio_a'],
                            #    self.agent_params['memory_prio_params']['memory_prio_b0'],
                            #    self.agent_params['memory_prio_params']['memory_prio_b_step']
                            # ),
                            debug_mode=self.agent_params['debug_mode'],
                            time_analysis=self.time_analysis)

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
            state = env_info.vector_observations[0]
            self.agent.reset()

            #train agent with another episode
            score = 0
            for _ in range(self.trainer_params['max_t']):
                self.time_analysis.start_timer(RunType.training_step)

                self.time_analysis.start_timer(RunType.agent_act)
                # action = self.agent.act(state, eps).astype(int)
                action = self.agent.act(state)
                self.time_analysis.end_timer(RunType.agent_act)

                self.time_analysis.start_timer(RunType.env_step)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                self.time_analysis.end_timer(RunType.env_step)

                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                self.time_analysis.end_timer(RunType.training_step)
                if done:
                    break

            #post-processing of latest episode
            scores_window.append(score)
            scores.append(score)
            # eps = max(self.trainer_params['eps_end'], self.trainer_params['eps_decay']*eps)
            results_file_tmp = open(self.results_path.replace('.csv', '_tmp.csv'), 'a')
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
                torch.save(self.agent.actor_local.state_dict(), './ModelWeights/' + self.agent_params['model_tag'] + '_actor.pth')
                torch.save(self.agent.critic_local.state_dict(), './ModelWeights/' + self.agent_params['model_tag'] + '_critic.pth')
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
        state = env_info.vector_observations[0]
        action_size = brain.vector_action_space_size
        score = 0
        
        if model_weights != '' and model_weights != 'random':
            print('Load model weights', './ModelWeights/' + model_weights + '.pth')
            self.agent.actor_local.load_state_dict(torch.load('./ModelWeights/' + model_weights + '_actor.pth'))
            self.agent.critic_local.load_state_dict(torch.load('./ModelWeights/' + model_weights + '_critic.pth'))
        
        while True:
            if model_weights == 'random':
                action = np.random.randint(action_size)
            else:
                action = self.agent.act(state).astype(int)
            
            env_info = self.env.step(action)[self.brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break
                
            print("\rScore: {}".format(score), end='')
        
        if self.results_path != '':    
            self.write_score_to_file(test_id, score)
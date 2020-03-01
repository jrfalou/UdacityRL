import torch

from ddpg_agent import Agent as ddpgAgent
from time_analysis import RunType
from utils import (
    get_layers_from_string,
    ReplayBuffer)

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self,
                 state_size,
                 action_size,
                 agent_params,
                 time_analysis=None):

        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = agent_params['batch_size']
        self.learn_step_nb = agent_params['learn_step_nb']

        self.agent_params = agent_params

        self.time_analysis = time_analysis
        
        # create ddpg agents
        self.agents = []
        for agent_params in agent_params['agents']:
            self.agents.append(
                ddpgAgent(state_size=state_size,
                    action_size=action_size,
                    actor_layers_params=
                        get_layers_from_string(agent_params['actor_hidden_layers']),
                    critic_layers_params=
                        get_layers_from_string(agent_params['critic_hidden_layers']),
                    soft_tau=agent_params['soft_tau'],
                    learning_rate_actor=agent_params['learning_rate_actor'],
                    learning_rate_critic=agent_params['learning_rate_critic'],
                    weight_decay_critic=0,
                    noise_params=(
                        agent_params['noise_theta'],
                        agent_params['noise_sigma']
                    ),
                    debug_mode=agent_params['debug_mode'],
                    time_analysis=self.time_analysis))

        # Replay memory
        self.memory = ReplayBuffer(self.action_size,
                                   BUFFER_SIZE,
                                   self.batch_size,
                                   self.time_analysis,
                                   device=DEVICE)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def start_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.start_timer(runType)

    def end_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.end_timer(runType)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        return [agent.act(obs, add_noise) for agent, obs in zip(self.agents, states)]

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.start_analysis_timer(RunType.memory_add)
        self.memory.add(states, actions, rewards, next_states, dones)
        self.end_analysis_timer(RunType.memory_add)
        
        # Learn every learn_step_nb time steps.
        self.t_step = (self.t_step + 1) % self.learn_step_nb
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                self.start_analysis_timer(RunType.memory_sample)
                experiences = self.memory.sample()
                self.end_analysis_timer(RunType.memory_sample)

                self.start_analysis_timer(RunType.agent_learn)
                self.learn(experiences, GAMMA)
                self.end_analysis_timer(RunType.agent_learn)

    def learn(self, experiences, gamma):
        for agent, exp in zip(self.agents, experiences):
            agent.learn(exp, gamma)

    def save_model(self):
        models_list = []
        for agent in self.agents:
            models_list.append({'actor': agent.actor_local.state_dict(), 'critic': agent.critic_local.state_dict()})

        torch.save(models_list, './ModelWeights/' + self.agent_params['model_tag'] + '.pth')
                
    def load_model(self, model_weights):
        models_list = torch.load('./ModelWeights/' + model_weights + '.pth',map_location=DEVICE)
        for agent, model in zip(self.agents, models_list):
            agent.actor_local.load_state_dict(model['actor'])
            agent.critic_local.load_state_dict(model['critic'])
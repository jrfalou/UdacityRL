import random
import copy

from collections import namedtuple, deque

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
from time_analysis import RunType

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self,
                state_size,
                action_size,
                actor_layers_params,
                critic_layers_params,
                soft_tau=1e-3,
                batch_size=128,
                learning_rate_actor=1e-4,
                learning_rate_critic=1e-3,
                weight_decay_critic=0,
                noise_params=(0.15, 0.2),
                learn_step_nb=1,
                # memory_prio_params=(False, 0, 0, 0), #(memory_prio_enabled, memory_prio_a, memory_prio_b0, memory_prio_b_step)
                debug_mode=False,
                time_analysis=None
                ):
        """Initialize an Agent object.
        
        Params
        ======
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
        """
        self.debug_mode = debug_mode
        
        self.state_size = state_size
        self.action_size = action_size
        self.soft_tau = soft_tau
        self.batch_size = batch_size
        self.lr_actor = learning_rate_actor
        self.lr_critic = learning_rate_critic
        self.weight_decay_critic = weight_decay_critic
        self.learn_step_nb = learn_step_nb
        
        #perf analysis
        self.time_analysis = time_analysis
    
        #set seeds
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size=self.state_size,
                                action_size=self.action_size,
                                fc1_units=actor_layers_params[0],
                                fc2_units=actor_layers_params[1]).to(DEVICE)
        self.actor_target = Actor(state_size=self.state_size,
                                action_size=self.action_size,
                                fc1_units=actor_layers_params[0],
                                fc2_units=actor_layers_params[1]).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size=self.state_size,
                                action_size=self.action_size,
                                fcs1_units=critic_layers_params[0],
                                fc2_units=critic_layers_params[1]).to(DEVICE)
        self.critic_target = Critic(state_size=self.state_size,
                                action_size=self.action_size,
                                fcs1_units=critic_layers_params[0],
                                fc2_units=critic_layers_params[1]).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                          lr=self.lr_critic,
                                          weight_decay=self.weight_decay_critic)

        # Noise process
        self.noise = OUNoise(self.action_size, mu=0, theta=noise_params[0], sigma=noise_params[1])

        # Replay memory
        self.memory = ReplayBuffer(self.action_size,
                                   BUFFER_SIZE,
                                   self.batch_size,
                                   self.time_analysis)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def start_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.start_timer(runType)

    def end_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.end_timer(runType)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.start_analysis_timer(RunType.memory_add)
        self.memory.add(state, action, reward, next_state, done)
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

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        self.start_analysis_timer(RunType.agent_critic_step)
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.end_analysis_timer(RunType.agent_critic_step)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.start_analysis_timer(RunType.agent_actor_step)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.end_analysis_timer(RunType.agent_actor_step)

        # ----------------------- update target networks ----------------------- #
        self.start_analysis_timer(RunType.agent_soft_update)
        self.soft_update(self.critic_local, self.critic_target, self.soft_tau)
        self.soft_update(self.actor_local, self.actor_target, self.soft_tau)
        self.end_analysis_timer(RunType.agent_soft_update)                  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, 
                action_size,
                buffer_size,
                batch_size,
                debug_mode=False,
                time_analysis=None):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.debug_mode = debug_mode
        self.time_analysis = time_analysis
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def start_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.start_timer(runType)

    def end_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.end_timer(runType)
                
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        self.start_analysis_timer(RunType.memory_sample_random)
        experiences = random.sample(self.memory, k=self.batch_size)
        self.end_analysis_timer(RunType.memory_sample_random)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
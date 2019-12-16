import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import itertools
from collections import OrderedDict

from time_analysis import TimeAnalysis
from time_analysis import RunType

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 4        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_agent(state_size, action_size, layers_params):
    #set torch seed so that every agent is instantiated with same weights
    torch.manual_seed(0)

    #complete layers_params with state_size and action_size
    agent_layers_params = layers_params.copy()
    agent_layers_params.insert(0, state_size)
    agent_layers_params.append(action_size)
    
    #create layers as list of tuples containing (in_features,out_features) for all NN layers (including first and last)
    layers = [(agent_layers_params[i], agent_layers_params[i+1]) for i in range(len(agent_layers_params)-1)]
    print('Create NN with layers:',layers)

    nn_structure = []
    for i in range(len(layers)):
        nn_structure.append(('fc' + str(i+1), nn.Linear(layers[i][0], layers[i][1])))
        if (i < len(layers)-1):
            nn_structure.append(('relu' + str(i+1), nn.ReLU()))

    nn_dict = OrderedDict(nn_structure)

    return nn.Sequential(nn_dict).to(device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, 
                state_size, 
                action_size, 
                layers_params, 
                double_dqn=False, 
                step_reward=0,
                soft_tau=1e-3,
                learning_rate=5e-4,
                memory_prio_params=(False, 0, 0, 0), #(memory_prio_enabled, memory_prio_a, memory_prio_b0, memory_prio_b_step)
                debug_mode=False,
                time_analysis=None
                ):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            layers_params ([int]): sizes of hidden layers in NN
            double_dqn (bool): enable double-dqn logic
            step_reward (int): extra reward to be added to environment reward
            soft_tau (float): rate at which target NN converges towards local NN
            learning_rate (float): optimizer learning rate
            memory_prio_params (bool, float, float, float): (memory_prio_enabled, memory_prio_a, memory_prio_b0, memory_prio_b_step)
            debug_mode (bool): enable debug logging
            time_analysis (TimeAnalysis): contains timers stats
        """
        self.debug_mode = debug_mode
        
        self.state_size = state_size
        self.action_size = action_size    
        self.double_dqn = double_dqn
        self.step_reward = step_reward
        self.soft_tau = soft_tau
        
        #perf analysis
        self.time_analysis = time_analysis
    
        #set seeds
        random.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        
        #qnetwork_local is the NN that chooses next action
        self.qnetwork_local = create_agent(state_size, action_size, layers_params)
        
        #qnetwork_target is the NN we want to optimize
        self.qnetwork_target = create_agent(state_size, action_size, layers_params)
        
        #optimizer that will update local NN weights after each backward propag
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        
        # Replay memory
        self.memory_prio_enabled = memory_prio_params[0] and memory_prio_params[1] > 0
        memory_prio_a = memory_prio_params[1]
        memory_prio_b0 = memory_prio_params[2]
        memory_prio_b_step = memory_prio_params[3]
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 
                                   memory_prio_a, memory_prio_b0, memory_prio_b_step,
                                   self.debug_mode, self.time_analysis)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def start_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.start_timer(runType)

    def end_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.end_timer(runType)

    def get_memory_prio_b(self):
        return self.memory.prio_b
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.start_analysis_timer(RunType.memory_add)
        td_error = 0.
        if self.memory_prio_enabled:
            with torch.no_grad():
                td_error = self.compute_td_error(GAMMA, 
                                                 torch.Tensor([state]), 
                                                 torch.Tensor([[int(action)]]).to(torch.int64), 
                                                 torch.Tensor([[reward]]), 
                                                 torch.Tensor([next_state]), 
                                                 torch.Tensor([[float(done)]]))[0][0]
#             print('td_error', td_error)
        self.memory.add(state, action, reward+self.step_reward, next_state, done, td_error)
        self.end_analysis_timer(RunType.memory_add)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.start_analysis_timer(RunType.memory_sample)
                experiences, prio_weights = self.memory.sample(self.time_analysis)
                self.end_analysis_timer(RunType.memory_sample)
                
                self.start_analysis_timer(RunType.agent_learn)
                self.learn(experiences, GAMMA, prio_weights)
                self.end_analysis_timer(RunType.agent_learn)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def compute_td_error(self, gamma, states, actions, rewards, next_states, dones):
        q_target = self.compute_q_targets(gamma, rewards, next_states, dones)
        q_expected = self.compute_q_expected(states, actions)
        return (q_target - q_expected).detach().numpy()
        
    def compute_q_targets(self, gamma, rewards, next_states, dones):
        if self.double_dqn:
            Q_targets_next = self.qnetwork_target(next_states).gather(1, self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1))
        else:
            Q_targets_next = self.qnetwork_target(next_states).max(1)[0].unsqueeze(1)
                
        #Q_targets is of dimensions (len(next_states) x 1)
        return rewards + (gamma * Q_targets_next * (1 - dones))
        
    def compute_q_expected(self, states, actions):
        #Q_expected_all_actions is a tensor of dimensions (len(states) x nb_actions)
        #contains the action_values based on states going through the local NN
        Q_expected_all_actions = self.qnetwork_local(states)
        
        #Q_expected_actual_actions is a tensor of dimensions (len(states) x 1)
        #contains the action_values related to the actual actions taken by the local NN in the experiences
        Q_expected_actual_actions = Q_expected_all_actions.gather(1, actions)
        return Q_expected_actual_actions

    def learn(self, experiences, gamma, prio_weights):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        #1. Compute Q_targets
        self.start_analysis_timer(RunType.agent_q_targets)
        Q_targets = self.compute_q_targets(gamma, rewards, next_states, dones)
        self.end_analysis_timer(RunType.agent_q_targets)
        
        #2. Compute Q_expected
        self.start_analysis_timer(RunType.agent_q_expected)
        Q_expected = self.compute_q_expected(states, actions)
        self.end_analysis_timer(RunType.agent_q_expected)
        
        #3. Compute loss
        self.start_analysis_timer(RunType.agent_loss)
        if self.memory_prio_enabled:
            loss_deprec = F.mse_loss(Q_expected, Q_targets)
            weights = torch.from_numpy(prio_weights[1]).unsqueeze(1).float()
#             print(weights.type(), Q_expected.type(), Q_targets.type())
            loss = torch.sum(weights * (Q_expected - Q_targets) ** 2)/len(weights)
            if self.debug_mode:
                print('learn_loss_deprec', loss_deprec)
                print('learn_loss_weights', loss)
        else:
            loss = F.mse_loss(Q_expected, Q_targets)
        self.end_analysis_timer(RunType.agent_loss)
                
        #4. Minimize the loss
        self.optimizer.zero_grad()
        
        self.start_analysis_timer(RunType.agent_backward)
        loss.backward()
        self.end_analysis_timer(RunType.agent_backward)
        
        self.start_analysis_timer(RunType.agent_optimize)
        self.optimizer.step()
        self.end_analysis_timer(RunType.agent_optimize)
            
        #5. update target network
        self.start_analysis_timer(RunType.agent_soft_update)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.soft_tau) 
        self.end_analysis_timer(RunType.agent_soft_update)       

        #6. update td_errors in replay memory
        if self.memory_prio_enabled:
            new_td_errors = self.compute_td_error(gamma, states, actions, rewards, next_states, dones)
            if self.debug_mode:
                print('learn_new_tds', new_td_errors.squeeze(1))
            self.memory.update_td_errors(prio_weights[0], new_td_errors)
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class MyRingBuffer:
    def __init__(self, maxlen, debug_mode=False):
        self.debug_mode = debug_mode
        self.array_ = np.zeros(maxlen)
        self.maxlen = maxlen
        self.index = -1
        self.sum = 0.
        self.length = 0
        
    def array(self):
        if self.length == self.maxlen:
            return self.array_
        else:
            return self.array_[:self.length]
        
    def add(self, add_value):
        if self.length == self.maxlen:
            self.sum -= self.get_oldest()
        self.sum += add_value
        
        self.index = self.index + 1 if self.index < self.maxlen - 1 else 0
        self.array_[self.index] = add_value
        self.length = min(self.length + 1, self.maxlen)
        
    def get(self, idx):
        return self.array_[idx]
    
    def get_oldest(self):
        idx = self.index - 1 if self.index > 0 else self.maxlen - 1
        return self.get(idx)
    
    def put(self, indices, values):
        to_be_replaced = self.array_[indices]
        if self.debug_mode:
            print('update_td_errors_before',to_be_replaced)
        self.sum += np.sum(values) - np.sum(to_be_replaced)
        np.put(self.array_, indices, values)
        if self.debug_mode:
            print('update_td_errors_after',self.array_[indices])
        
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, 
        prio_a=0, prio_b0=0, prio_b_step=0, 
        debug_mode=False,
        time_analysis=None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.debug_mode = debug_mode
        self.time_analysis = time_analysis
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
        self.prio_a = prio_a
        self.prio_b0 = prio_b0
        self.prio_b_step = prio_b_step
        self.prio_b = prio_b0
        self.prio_epsilon = 0.01
        self.prio_errors = MyRingBuffer(buffer_size, self.debug_mode)
        
        self.np_memory_arange = None
    
    def start_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.start_timer(runType)

    def end_analysis_timer(self, runType=RunType.unknown):
        if self.time_analysis is not None:
            self.time_analysis.end_timer(runType)
                
    def add(self, state, action, reward, next_state, done, td_error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
        if self.prio_a > 0:
            prio_td_error = (abs(td_error) + self.prio_epsilon)**self.prio_a
            if self.debug_mode:
                print('add_td_error_before', td_error, prio_td_error)
            self.prio_errors.add(prio_td_error)
            if self.debug_mode:
                print('add_td_error_after', self.prio_errors.index, self.prio_errors.get(self.prio_errors.index))
    
    def sample(self, time_analysis):
        """Randomly sample a batch of experiences from memory."""
        experiences_indices = np.empty(1)
        prio_weights = np.empty(1)
        if self.prio_a == 0:
            self.start_analysis_timer(RunType.memory_sample_random)
            experiences = random.sample(self.memory, k=self.batch_size)
            self.end_analysis_timer(RunType.memory_sample_random)
        
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        else:
            if len(self.memory) == self.memory.maxlen:
                if len(self.np_memory_arange) < self.memory.maxlen:
                    self.np_memory_arange = np.arange(self.memory.maxlen)
            else:
                self.np_memory_arange = np.arange(len(self.memory))
        
            self.start_analysis_timer(RunType.memory_sample_random)
            prio_probas = self.prio_errors.array()/self.prio_errors.sum
            try:
                experiences_indices = np.random.choice(len(self.memory), size=self.batch_size, p=prio_probas)
            except: #this is to handle the case when self.prio_errors.sum != np.sum(self.prio_errors_array_ because of rounding issues)
#                 print('except: sum is not ok, reset to true')
                self.prio_errors.sum = np.sum(self.prio_errors.array())
                prio_probas = self.prio_errors.array()/self.prio_errors.sum
                experiences_indices = np.random.choice(len(self.memory), size=self.batch_size, p=prio_probas)
                
            self.end_analysis_timer(RunType.memory_sample_random)
            
            states = torch.from_numpy(np.vstack([self.memory[i].state for i in experiences_indices])).float().to(device)
            actions = torch.from_numpy(np.vstack([self.memory[i].action for i in experiences_indices])).long().to(device)
            rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in experiences_indices])).float().to(device)
            next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in experiences_indices])).float().to(device)
            dones = torch.from_numpy(np.vstack([self.memory[i].done for i in experiences_indices]).astype(np.uint8)).float().to(device)
            
            self.prio_b = min(self.prio_b + self.prio_b_step, 1.)
            prio_weights = (len(self.memory) * prio_probas[experiences_indices]) ** (-self.prio_b)
            prio_weights /= np.max(prio_weights)
            if self.debug_mode:
                print('sample_probas',prio_probas)
                print('sample_indices', experiences_indices)
                print('sample_weights', prio_weights)
            
        return ((states, actions, rewards, next_states, dones), (experiences_indices, prio_weights))
    
    def update_td_errors(self, prio_indices, new_td_errors):
        self.prio_errors.put(prio_indices, (abs(new_td_errors) + self.prio_epsilon)**self.prio_a)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
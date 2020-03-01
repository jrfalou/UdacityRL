import random
import torch
import numpy as np
from collections import namedtuple, deque
from time_analysis import RunType

def get_list_from_string(str_list):
    value_list = str_list.replace(' ', '').strip('][').split(',')
    return [float(v) for v in value_list]

def get_layers_list_from_string(str_layers):
    layers_list = str_layers.replace(' ', '').strip(']()[').split('),(')
    return [(int(a.split(',')[0]), int(a.split(',')[1])) for a in layers_list]

def get_layers_from_string(str_layers):
    layers_list = str_layers.replace(' ', '').strip('()').split(',')
    return [int(a) for a in layers_list]

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, 
                action_size,
                buffer_size,
                batch_size,
                debug_mode=False,
                time_analysis=None,
                device=torch.device('cpu')):
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

        self.device = device

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

        output_size = len(experiences[0].state)
        output = []
        for s in range(output_size):
            states = torch.from_numpy(np.vstack([e.state[s] for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action[s] for e in experiences if e is not None])).float().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward[s] for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state[s] for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done[s] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
            
            output.append((states, actions, rewards, next_states, dones))

        return output[0] if output_size == 1 else output

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
from enum import Enum
import datetime

class RunType(Enum):
    unknown = 0
    full_training = 1
    memory_add = 2
    memory_sample = 3
    memory_sample_random = 4
    agent_act = 5
    agent_learn = 6
    agent_q_targets = 7
    agent_q_expected = 8
    agent_loss = 9
    agent_backward = 10
    agent_optimize = 11
    agent_soft_update = 12
    env_step = 13
    training_step = 14

class TimeAnalysis:
    def __init__(self):
        self.timers = {}

    def start_timer(self, runType=RunType.unknown):
        if runType not in self.timers:
            new_timer = RunTimer(runType=runType)
            self.timers[runType] = new_timer
        self.timers[runType].start()

    def end_timer(self, runType=RunType.unknown):
        if runType not in self.timers:
            print('ERROR')
        else:
            self.timers[runType].end()

    def to_str(self):
        out_str = ''
        for t in self.timers.values():
            out_str += t.to_str() + '\n'
        return out_str

class RunTimer:
    def __init__(self, runType=RunType.unknown):
        self.type = runType
        self.start_t = 0
        self.count = 0
        self.sum = datetime.timedelta()

    def start(self):
        self.start_t = datetime.datetime.now()

    def end(self):
        end_t = datetime.datetime.now()
        self.sum += end_t - self.start_t
        self.count += 1
        self.start_t = 0

    def mean(self):
        return self.sum / self.count if self.count > 0 else 0

    def to_str(self):
        return ','.join([str(self.type), str(self.count), str(self.mean()), str(self.sum)]) 
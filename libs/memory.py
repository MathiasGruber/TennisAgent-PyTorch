import random
import numpy as np
import torch
from torch.autograd import Variable

from collections import namedtuple, deque
from libs.sumtree import SumTree

# Determine if CPU or GPU computation should be used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    Code expanded and adapted from code examples provided by Udacity DRL Team, 2018.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Arguments:
            buffer_size {int} -- maximum size of buffer
            batch_size {int} -- size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Arguments:
            state {int} -- maximum size of buffer
            action {int} -- size of each training batch
            reward {int} -- size of each training batch
            next_state {int} -- size of each training batch
            done {int} -- size of each training batch
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory.
        
        Returns:
            [int] -- samples
        """
        return len(self.memory)

class PrioritizedReplayMemory:  
    '''
    Implementation of prioritized experience replay. Adapted from:
    https://github.com/rlcode/per/blob/master/prioritized_memory.py
    '''

    def __init__(self, capacity, batch_size):
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.batch_size = batch_size

        self.tree = SumTree(capacity)
        self.capacity = capacity

    def __len__(self):
        """Number of samples in memory
        
        Returns:
            [int] -- samples
        """

        return self.tree.n_entries

    def _get_priority(self, error):
        """Get priority based on error
        
        Arguments:
            error {float} -- TD error
        
        Returns:
            [float] -- priority
        """

        return (error + self.e) ** self.a

    def add(self, error, sample):
        """Add sample to memory
        
        Arguments:
            error {float} -- TD error
            sample {tuple} -- tuple of (state, action, reward, next_state, done)
        """

        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self):
        """Sample from prioritized replay memory
        
        Arguments:
            n {int} -- sample size
        
        Returns:
            [tuple] -- tuple of ((state, action, reward, next_state, done), idxs, is_weight)
        """

        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if p > 0:
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)

        # Calculate importance scaling for weight updates
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)

        # Paper states that for stability always scale by 1/max w_i so that we only scale downwards
        is_weight /= is_weight.max()

        # Extract (s, a, r, s', done)
        batch = np.array(batch).transpose()        
        states = np.vstack(batch[0])        
        actions = list(batch[1])
        rewards = list(batch[2])
        next_states = np.vstack(batch[3])
        dones = batch[4].astype(int)

        # Move to device etc.
        states = Variable(torch.Tensor(states)).float().to(device)        
        actions = Variable(torch.Tensor(actions)).float().to(device)        
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = Variable(torch.Tensor(next_states)).float().to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        is_weight = torch.FloatTensor(is_weight).unsqueeze(1).to(device)

        return (states, actions, rewards, next_states, dones), idxs, is_weight

    def update(self, idx, error):
        """Update the priority of a sample
        
        Arguments:
            idx {int} -- index of sample in the sumtree
            error {float} -- updated TD error
        """

        p = self._get_priority(error)
        self.tree.update(idx, p)

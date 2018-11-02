import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # DEBUG experience sampling
        #print('sampling:')
        #for i, e in enumerate(experiences):
        #    print('------ experience {}:'.format(i))
        #    print(np.sum(e.state))
        #    print(e.action)
        #    print(e.reward)
        #    print(np.sum(e.next_state))
        #    print(e.done)
        #    print(e.priority)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # implicitly use LongTensor for discete actions and FloatTensor for continuous actions
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])

    def batch_update(self, indexes, experiences):
        """ Batch update existing elements in memory. """
        states, actions, rewards, next_states, dones, new_priorities = experiences
        for i in range(self.batch_size):
            e = self.experience(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]), float(new_priorities[i]))
            self.memory[indexes[i]] = e

    def sort(self):
        """ Sort memory based on priority (TD error) """
        # sort memory based on priority (sixth item in experience tuple)
        items = [self.memory.pop() for i in range(len(self.memory))]
        items.sort(key=lambda x: x[5], reverse=True)
        self.memory.extend(items)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self, alpha):
        """ Sample a batch of experiences from memory using Prioritized Experience. """
        # get the number of items in the experience replay memory
        n_items = len(self.memory)
        # calculate the sum of all the probabilities of all the items
        sum_of_probs = sum((1/i) ** alpha for i in range(1, n_items + 1))
        # build a probability list for all the items
        probs = [(1/i) ** alpha / sum_of_probs for i in range(1, n_items + 1)]
        # sample from the replay memory using the probability list
        indexes = np.random.choice(n_items, self.batch_size, p=probs)
        # use the indexes to generate a list of experience tuples
        experiences = [self.memory[i] for i in indexes]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # implicitly use LongTensor for discete actions and FloatTensor for continuous actions
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return indexes, (states, actions, rewards, next_states, dones, priorities)

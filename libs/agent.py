"""
Classes to model an RL agent.
"""

import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, model, action_size,
                 use_double_dqn=True,
                 use_prioritized_experience_replay=False,
                 alpha_start=0.5,
                 alpha_decay=0.9992):
        """Initialize an Agent object.

        Params
        ======
            action_size (int): dimension of each action
            use_double_dqn (bool): wheter to use double DQN algorithm
            use_prioritized_experience_replay (bool): wheter to use PER algorithm
            alpha_start (float): initial value for alpha, used in PER
            alpha_decay (float): decay rate for alpha, used in PER
        """
        self.action_size = action_size

        self.use_double_dqn = use_double_dqn
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        self.loss_list = []       # track loss across steps
        self.entropy_list = []    # track entropy across steps

        # Q-Network
        self.qnetwork_local = model.local
        self.qnetwork_target = model.target
        # debug weight initialization
        #print(self.qnetwork_local.fc_s.weight.data[0])
        #print(self.qnetwork_target.fc_s.weight.data[0])
        #self.qnetwork_local.fc_s.weight.data[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        #print(self.qnetwork_local.fc_s.weight.data[0])
        #print(self.qnetwork_target.fc_s.weight.data[0])
        #input('->')

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        #self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=.00025, momentum=0.95)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # initalize alpha (used in prioritized experience sampling probability)
        self.alpha_start = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha = self.alpha_start

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        priority = 100.0   # set initial priority to max value
        self.memory.add(state, action, reward, next_state, done, priority)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # if prioritized experience replay is enabled
                if self.use_prioritized_experience_replay:
                    self.memory.sort()
                    indexes, experiences = self.memory.sample_with_priority(self.alpha)
                    self.learn(indexes, experiences, GAMMA)
                    self.alpha = self.alpha_decay*self.alpha
                else:
                    experiences = self.memory.sample()
                    self.learn(None, experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # reshape 1-D states into 2-D
        # not strictly necessary but more asthetically pleasing
        if len(state.shape) == 1:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, indexes, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, priorities = experiences

        # Select double DQN or regular DQN
        if self.use_double_dqn:
            # get greedy actions (for next states) from local model
            q_local_argmax = self.qnetwork_local(next_states).detach().argmax(dim=1).unsqueeze(1)
            # get predicted q values (for next states) from target model indexed by q_local_argmax
            q_targets_next = self.qnetwork_target(next_states).gather(1, q_local_argmax).detach()
        else:
            # get max predicted q values (for next states) from target model
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # get q values from local model
        q_local = self.qnetwork_local(states)
        # get q values for chosen action
        predictions = q_local.gather(1, actions)
        # calculate td targets
        targets = rewards + (gamma * q_targets_next * (1 - dones))

        # calculate new priorities
        if self.use_prioritized_experience_replay:
            with torch.no_grad():
                new_priorities = torch.abs(targets - predictions).to(device)
                self.memory.batch_update(indexes, (states, actions, rewards, next_states, dones, new_priorities))

        # calculate loss using mean squared error: (targets - predictions).pow(2).mean()
        loss = F.mse_loss(predictions, targets)
        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients
        #for param in self.qnetwork_local.parameters():
        #    param.grad.data.clamp_(-10, 10)
        self.optimizer.step()

        # update stats
        with torch.no_grad():
            self.loss_list.append(loss.item())
            # calculate sparse softmax cross entropy
            self.entropy_list.append(F.cross_entropy(q_local, actions.squeeze(1)))

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)


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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""

        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return (states, actions, rewards, next_states, dones, priorities)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    # ---------- prioritized exeperience replay methods ---------- #
    def batch_update(self, indexes, experiences):
        """ Batch update existing elements in memory. """

        states, actions, rewards, next_states, dones, new_priorities = experiences
        for i in range(self.batch_size):
            e = self.experience(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]), float(new_priorities[i]))
            self.memory[indexes[i]] = e

    def sort(self):
        """ Sort memory based on priority (TD error) """

        # sort memory based on priority (sixth item in experience tuple)
        self.memory = sorted(self.memory, key=lambda x: x[5], reverse=True)

    def sample_with_priority(self, alpha):
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return indexes, (states, actions, rewards, next_states, dones, priorities)
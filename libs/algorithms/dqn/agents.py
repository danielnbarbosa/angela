"""
Deep Q Network agent.
"""

import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
#from visualize import show_frames_3d
import dill
from libs.agent_util import ReplayBuffer, PrioritizedReplayBuffer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, model, action_size, seed=0, load_file=None,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 tau=1e-3,
                 lr=5e-4,
                 update_every=4,
                 use_double_dqn=True,
                 use_prioritized_experience_replay=False,
                 alpha_start=0.5,
                 alpha_decay=0.9992,
                 action_map=None):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr (float): learning rate
            update_every (int): how often to update the network
            use_double_dqn (bool): wheter to use double DQN algorithm
            use_prioritized_experience_replay (bool): wheter to use PER algorithm
            alpha_start (float): initial value for alpha, used in PER
            alpha_decay (float): decay rate for alpha, used in PER
            action_map (dict): how to map action indexes from model output to gym environment
        """
        random.seed(seed)

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        self.loss_list = []       # track loss across steps
        self.entropy_list = []    # track entropy across steps

        # Q-Network
        self.qnetwork_local = model.local
        self.qnetwork_target = model.target

        # DEBUG weight initialization
        #print(self.qnetwork_local.fc_s.weight.data[0])
        #print(self.qnetwork_target.fc_s.weight.data[0])
        #self.qnetwork_local.fc_s.weight.data[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
        #print(self.qnetwork_local.fc_s.weight.data[0])
        #print(self.qnetwork_target.fc_s.weight.data[0])
        #input('->')

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        #self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=.00025, momentum=0.95)

        # Replay memory
        if use_prioritized_experience_replay:
            self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        else:
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        # initalize alpha (used in prioritized experience sampling probability)
        self.alpha_start = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha = self.alpha_start

        if load_file:
            self.qnetwork_local.load_state_dict(torch.load(load_file + '.pth'))
            self.qnetwork_target.load_state_dict(torch.load(load_file + '.pth'))
            #self.memory = dill.load(open(load_file + '.buffer.pck','rb'))
            print('Loaded: {}'.format(load_file))

        self.action_map = action_map

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.use_prioritized_experience_replay:
            priority = 100.0   # set initial priority to max value
            self.memory.add(state, action, reward, next_state, done, priority)
        else:
            self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # if prioritized experience replay is enabled
                if self.use_prioritized_experience_replay:
                    self.memory.sort()
                    indexes, experiences = self.memory.sample(self.alpha)
                    self.learn(experiences, self.gamma, indexes)
                    self.alpha = self.alpha_decay*self.alpha
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if len(state.shape) == 1:   # reshape 1-D states into 2-D (as expected by the model)
            state = np.expand_dims(state, axis=0)
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

    def learn(self, experiences, gamma, indexes=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.use_prioritized_experience_replay:
            states, actions, rewards, next_states, dones, priorities = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # DEBUG replay memory
        #print('learning:')
        #show_frames(states)
        #show_frames(next_states)

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
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)


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

"""
Multi-Agent DDPG with single shared actor/critic across all agents.
"""

import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import dill
#from visualize import show_frames_3d
from libs.agent_util import OUNoise, ReplayBuffer, PrioritizedReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, model, action_size, seed=0, load_file=None,
                 n_agents=1,
                 buffer_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 weight_decay=0.0001,
                 clip_critic_gradients=False,
                 update_every=1,
                 use_prioritized_experience_replay=False,
                 alpha_start=0.5,
                 alpha_decay=0.9992,
                 evaluation_only=False):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            n_agents (int): number of agents to train simultaneously
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
            clip_critic_gradients (bool): whether to clip critic gradients
            update_every (int): how often to update the network
            use_prioritized_experience_replay (bool): wheter to use PER algorithm
            alpha_start (float): initial value for alpha, used in PER
            alpha_decay (float): decay rate for alpha, used in PER
            evaluation_only (bool): set to True to disable updating gradients and adding noise
        """
        random.seed(seed)

        self.action_size = action_size
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_every = update_every
        self.use_prioritized_experience_replay = use_prioritized_experience_replay
        self.clip_critic_gradients = clip_critic_gradients
        self.evaluation_only = evaluation_only

        self.loss_list = []       # track loss across steps

        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # DEBUG weight initialization
        #print(self.actor_local.fcs1.weight.data[0])
        #print(self.actor_target.fcs1.weight.data[0])
        #print(self.critic_local.fcs1.weight.data[0])
        #print(self.critic_target.fcs1.weight.data[0])
        #input('->')

        # Noise process
        self.noise = OUNoise((n_agents, action_size), seed)

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
            if device.type == 'cpu':
                self.actor_local.load_state_dict(torch.load(load_file + '.actor.pth', map_location='cpu'))
                self.actor_target.load_state_dict(torch.load(load_file + '.actor.pth', map_location='cpu'))
                self.critic_local.load_state_dict(torch.load(load_file + '.critic.pth', map_location='cpu'))
                self.critic_target.load_state_dict(torch.load(load_file + '.critic.pth', map_location='cpu'))
                #self.memory = dill.load(open(load_file + '.buffer.pck','rb'))
            elif device.type == 'cuda:0':
                self.actor_local.load_state_dict(torch.load(load_file + '.actor.pth'))
                self.actor_target.load_state_dict(torch.load(load_file + '.actor.pth'))
                self.critic_local.load_state_dict(torch.load(load_file + '.critic.pth'))
                self.critic_target.load_state_dict(torch.load(load_file + '.critic.pth'))
                #self.memory = dill.load(open(load_file + '.buffer.pck','rb'))
            print('Loaded: {}'.format(load_file))


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.use_prioritized_experience_replay:
            priority = 100.0   # set initial priority to max value
            if self.n_agents == 2:  # DIFF from DDPG
                self.memory.add(state, action, reward, next_state, done, priority)
            else:
                for i in range(self.n_agents):
                    self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i], priority[i,:])
        else:
            if self.n_agents == 2:  # DIFF from DDPG
                self.memory.add(state, action, reward, next_state, done)
            else:
                for i in range(self.n_agents):
                    self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.evaluation_only == False:
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


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        if len(state.shape) == 1:   # reshape 1-D states into 2-D (as expected by the model)
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma, indexes=None):
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
        if self.use_prioritized_experience_replay:
            states, actions, rewards, next_states, dones, priorities = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # DEBUG replay memory
        #print('learning:')
        #show_frames(states)
        #show_frames(next_states)

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # compute critic loss
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        critic_loss = F.mse_loss(q_expected, q_targets)
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # DEBUG gradients
        #for m in self.critic_local.parameters():
        #    print(m.grad)
        # clip gradients
        if self.clip_critic_gradients:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
            #for param in self.qnetwork_local.parameters():
            #    param.grad.data.clamp_(-10, 10)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # DEBUG gradients
        #for m in self.actor_local.parameters():
        #    print(m.grad)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # ---------------- update prioritized experience replay ---------------- #
        if self.use_prioritized_experience_replay:
            with torch.no_grad():
                new_priorities = torch.abs(q_targets - q_expected).to(device)
                self.memory.batch_update(indexes, (states, actions, rewards, next_states, dones, new_priorities))

        # ---------------------------- update stats ---------------------------- #
        with torch.no_grad():
            self.loss_list.append(critic_loss.item())


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

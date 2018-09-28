"""
Proximal Policy Optimization agent.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ProximalPolicyOptimization():
    def __init__(self, model, seed=0, load_file=None, lr=1e-4, action_map=None):
        """
        Params
        ======
            model: model object
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            lr (float): learning rate
            action_map (dict): how to map action indexes from model output to gym environment
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model = model.to(device)
        if load_file:
            # self.model.load_state_dict(torch.load(load_file))
            self.model.load_state_dict(torch.load(load_file, map_location='cpu'))  # load from GPU to CPU
            print('Loaded: {}'.format(load_file))
        self.action_map = action_map
        self.optimizer = optim.Adam(model.parameters(), lr=lr)


    def _discount(self, rewards, gamma, normal):
        """
        Calulate discounted future (and optionally normalized) rewards.
        From https://github.com/wagonhelm/Deep-Policy-Gradient
        """

        discounted_rewards = np.zeros_like(rewards)
        G = 0.0
        for i in reversed(range(0, len(rewards))):
            G = G * gamma + rewards[i]
            discounted_rewards[i] = G
        # normalize rewards
        if normal:
            mean = np.mean(discounted_rewards)
            std = np.std(discounted_rewards)
            std = max(1e-8, std) # avoid divide by zero if rewards = 0.0
            discounted_rewards = (discounted_rewards - mean) / (std)
        return discounted_rewards


    def act(self, state):
        """
        Given a state, determine the next action.
        Returns (environment action, action index, action probability)
        """

        if len(state.shape) == 1:   # reshape 1-D states into 2-D (as expected by the model)
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(device)
        probs = self.model.forward(state).cpu().detach()
        m = Categorical(probs)
        action = m.sample()
        # DEBUG
        #print(probs)
        #print(probs.shape)
        #print(action)
        #print(action.shape)
        #print(probs[0][action.item()])
        action_index = action.item()
        action_prob = probs[0][action.item()]
        # use action_map if it exists
        if self.action_map:
            return self.action_map[action_index], action_index, action_prob
        else:
            return action_index, action_index, action_prob

    def calc_probs(self, states, actions):
        """
        Given a list of states and actions run each state through the model
        and get the new probabilities for the actions.
        """

        states = np.asarray(states)
        states = states.squeeze()  # this is needed for Conv2D envs, doesn't hurt other envs
        states = torch.from_numpy(states).float().to(device)
        probs = self.model.forward(states)
        actions = actions.unsqueeze(1)
        # DEBUG
        #print(probs)
        #print(probs.shape)
        #print(actions)
        #print(actions.shape)
        #print(probs.gather(1, actions))
        return probs.gather(1, actions).squeeze(1)

    def learn(self, old_probs, states, actions, rewards_feed, gamma, epsilon, beta):
        """Update model weights."""
        # DEBUG
        #print('PRE old_probs: {}'.format(old_probs))
        #print('PRE states: {}'.format(states))
        #print('PRE actions: {}'.format(actions))
        #print('PRE rewards: {}'.format(rewards_feed))

        # calculate discounted rewards for each step and normalize them
        discounted_rewards_feed = []
        for rewards in rewards_feed:
            discounted_rewards = self._discount(rewards, gamma, True)
            discounted_rewards_feed.append(discounted_rewards)
        # DEBUG
        #print('rewards: {}'.format(rewards))
        #print('discounted_rewards: {}'.format(discounted_rewards))

        # flatten feeds
        flatten = lambda l: [item for sublist in l for item in sublist]
        old_probs = flatten(old_probs)
        states = flatten(states)
        actions = flatten(actions)
        rewards = flatten(discounted_rewards_feed)
        # DEBUG
        #print('PST old_probs: {}'.format(old_probs))
        #print('PST states: {}'.format(states))
        #print('PST actions: {}'.format(actions))
        #print('PST rewards: {}'.format(rewards))

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        new_probs = self.calc_probs(states, actions)

        # ratio for clipping
        ratio = new_probs/old_probs
        # DEBUG
        #print('old_probs: {}'.format(old_probs))
        #print('new_probs: {}'.format(new_probs))
        #print('ratio: {}'.format(ratio))

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        # TODO: fix this to work with softmax
        entropy = -(new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
        # DEBUG
        #print('old_probs: {}'.format(old_probs))
        #print('new_probs: {}'.format(new_probs))
        #print('clipped_surrogate: {}'.format(clipped_surrogate))
        #print('entropy: {}'.format(entropy))

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        #policy_loss = -torch.mean(clipped_surrogate + beta*entropy)
        policy_loss = -torch.sum(clipped_surrogate + beta*entropy)
        # DEBUG
        #print('policy_loss: {}'.format(policy_loss))

        self.optimizer.zero_grad()
        policy_loss.backward()
        # DEBUG
        #for m in self.model.parameters():
        #    print(m.grad)
        self.optimizer.step()

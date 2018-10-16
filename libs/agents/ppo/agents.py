"""
Proximal Policy Optimization agent.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ProximalPolicyOptimization():
    def __init__(self, model, seed=0, load_file=None, lr=1e-4,
                 action_map=None,
                 n_agents=1):
        """
        Params
        ======
            model: model object
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            lr (float): learning rate
            action_map (dict): how to map action indexes from model output to gym environment
            n_agents (int): number of agents (used for UnityML multi-agent environments)

        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.model = model.to(device)
        if load_file:
            # self.model.load_state_dict(torch.load(load_file))
            self.model.load_state_dict(torch.load(load_file, map_location='cpu'))  # load from GPU to CPU
            print('Loaded: {}'.format(load_file))
        self.action_map = action_map
        self.n_agents = n_agents
        self.optimizer = optim.Adam(model.parameters(), lr=lr)


    def _discount(self, rewards, gamma, normal):
        """
        Calculate discounted future (and optionally normalized) rewards.
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


    @staticmethod
    def _convert_lists(input_lists):
        """
        Convert list of list of steps to list of list of trajectories and vice-versa.
        """

        output_lists = []
        for i in range(len(input_lists[0])):
            output_list = []
            for input_list in input_lists:
                output_list.append(input_list[i])
            output_lists.append(output_list)
        return(output_lists)


    def _discount_multi_agent(self, rewards, gamma, normal):
        """
        Calulate discounted future (and optionally normalized) rewards for multi-agent environments.
        """

        # convert rewards by steps to rewards by trajectories
        rewards_by_trajectories = self._convert_lists(rewards)

        # apply discounting
        discounted_rewards_list = []
        for rewards in rewards_by_trajectories:
            discounted_rewards = np.zeros_like(rewards)
            G = 0.0
            for i in reversed(range(0, len(rewards))):
                G = G * gamma + rewards[i]
                #print(i, G)
                discounted_rewards[i] = G
            # normalize rewards
            if normal:
                mean = np.mean(discounted_rewards)
                std = np.std(discounted_rewards)
                std = max(1e-8, std) # avoid divide by zero if rewards = 0.0
                discounted_rewards = (discounted_rewards - mean) / (std)
            discounted_rewards_list.append(discounted_rewards)

        # convert rewards by trajectories back to rewards by steps
        rewards_by_steps = self._convert_lists(discounted_rewards_list)
        return(rewards_by_steps)


    def act(self, state):
        """
        Given a state, run state through the model.
        Returns the action expected by the environment (after passing through action_map),
        index of sampled action (for replaying saved trajectories),
        probability of sampled action
        """
        if len(state.shape) == 1:   # reshape 1-D states into 2-D (as expected by the model)
            state = np.expand_dims(state, axis=0)
        state = torch.from_numpy(state).float().to(device)
        probs = self.model.forward(state).cpu().detach()
        m = Categorical(probs)
        action = m.sample()
        if self.n_agents == 1:
            action_index = action.item()
            action_prob = probs[0][action.item()]
        else:
            action_index = action.numpy()
            action_prob = probs.gather(1, action.unsqueeze(1))
        # DEBUG
        #print(self.action_map[action_index], action_index, action_prob, probs)
        #print(action_index, action_prob, probs)
        # use action_map if it exists
        if self.action_map:
            return self.action_map[action_index], action_index, action_prob
        else:
            return action_index, action_index, action_prob


    def calc_probs(self, states, actions):
        """
        Given states and actions, run states through the model.
        Returns new probabilities for the action
        and the full probability distribution (for calculating entropy).
        """
        states = np.asarray(states)
        states = states.squeeze()  # this is needed for Conv2D envs, doesn't hurt other envs
        states = torch.from_numpy(states).float().to(device)
        probs = self.model.forward(states)
        if self.n_agents == 1:
            actions = actions.unsqueeze(1)
            return probs.gather(1, actions).squeeze(1), probs
        else:
            actions = actions.unsqueeze(2)
            return probs.gather(2, actions).squeeze(2), probs


    def learn(self, old_probs, states, actions, rewards_lists, gamma, epsilon, beta):
        """Update model weights."""

        # calculate discounted rewards for each step and normalize them across each trajectory
        discounted_rewards_lists = []
        for rewards in rewards_lists:
            if self.n_agents == 1:
                discounted_rewards = self._discount(rewards, gamma, True)
            else:
                discounted_rewards = self._discount_multi_agent(rewards, gamma, True)
            discounted_rewards_lists.append(discounted_rewards)
        # flatten discounted_rewards_lists
        flatten = lambda l: [item for sublist in l for item in sublist]
        rewards = flatten(discounted_rewards_lists)
        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int64, device=device)
        if self.n_agents == 1:
            old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        else:
            old_probs = [t.numpy() for t in old_probs]
            old_probs = torch.tensor(old_probs, dtype=torch.float, device=device).squeeze(2)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # get new probabilities for actions using current policy
        new_probs, new_probs_all = self.calc_probs(states, actions)
        # ratio for clipping
        ratio = new_probs/old_probs
        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
        # entropy bonus (for regularization)
        # used to encourage less extreme probabilities and hence exploration, early on
        # add in 1.e-10 to avoid log(0) which gives nan
        #entropy = -(new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
        entropy = -torch.sum((new_probs_all * torch.log(new_probs_all+1.e-10)), dim=1)
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        #policy_loss = -torch.mean(clipped_surrogate + beta*entropy)
        if self.n_agents == 1:
            policy_loss = -torch.sum(clipped_surrogate + beta*entropy)
        else:
            policy_loss = -torch.sum(clipped_surrogate)  # TODO fix entropy calc in multi-agent
        self.optimizer.zero_grad()
        policy_loss.backward()
        # DEBUG
        #for m in self.model.parameters():
        #    print(m.grad)
        self.optimizer.step()

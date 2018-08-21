import random
import numpy as np
from unityagents import UnityEnvironment
import gym

class Environment():
    """ Define an environment, currently either OpenAI Gym or UnityML. """

    def __init__(self, name, type, one_hot=None, max_steps=None):
        self.type = type
        self.one_hot = one_hot

        if self.type == 'gym':
            self.env = gym.make(name)
            # override environment default for max steps in an episode
            if max_steps:
                self.env._max_episode_steps = max_steps
        elif self.type == 'unity':
            # need to manually set seed to ensure a random environment is initialized
            self.env = UnityEnvironment(file_name=name, seed=random.randint(0, 2 ** 30))
            self.brain_name = self.env.brain_names[0]


    def reset(self):
        """Reset the environment."""

        if self.type == 'gym':
            state = self.env.reset()
            # one-hot encode state space (needed in environments where each state is numbered)
            if self.one_hot:
                state = np.eye(self.one_hot)[state]
        elif self.type == 'unity':
            info = self.env.reset(train_mode=True)[self.brain_name]
            state = info.vector_observations[0]
        return state

    def step(self, action):
        """Take a step in the environment.  Given an action, return the next state."""

        if self.type == 'gym':
            state, reward, done, _ = self.env.step(action)
            #state, reward, done, _ = env.step([(action/2) - 2]) # action discretization for Pendulum
            # one-hot encode state space (needed in environments where each state is numbered)
            if self.one_hot:
                state = np.eye(self.one_hot)[state]
        elif self.type == 'unity':
            info = self.env.step(action)[self.brain_name]   # send the action to the environment
            state = info.vector_observations[0]             # get the next state
            reward = info.rewards[0]                        # get the reward
            done = info.local_done[0]
        return state, reward, done

    def render(self):
        """ Render the environment to visualize the agent interacting."""
        
        if self.type == 'gym':
            self.env.render()
        elif self.type == 'unity':
            pass

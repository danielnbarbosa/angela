import gym
import random
from unityagents import UnityEnvironment

class Environment():
    """ Defines an environment. """

    def __init__(self, name, type, max_steps=None):
        self.type = type
        if self.type == 'gym':
            self.env = gym.make(name)
            if bool(max_steps):
                self.env._max_episode_steps = max_steps
        elif self.type == 'unity':
            # need to manually set seed to ensure a random environment is initialized
            SEED = random.randint(0, 2 ** 30)
            self.env = UnityEnvironment(file_name=name, seed=SEED)
            self.brain_name = self.env.brain_names[0]



    def reset(self):
        if self.type == 'gym':
            state = self.env.reset()
            #state = np.eye(64)[state] # one-hot encode for FrozenLake
        elif self.type == 'unity':
            info = self.env.reset(train_mode=True)[self.brain_name]
            state = info.vector_observations[0]
        return state

    def step(self, action):
        if self.type == 'gym':
            state, reward, done, _ = self.env.step(action)
            #state, reward, done, _ = env.step([(action/2) - 2]) # action discretization for Pendulum
            #state = np.eye(64)[state] # one-hot encode for FrozenLake
        elif self.type == 'unity':
            info = self.env.step(action)[self.brain_name]   # send the action to the environment
            state = info.vector_observations[0]             # get the next state
            reward = info.rewards[0]                        # get the reward
            done = info.local_done[0]
        return state, reward, done

    def render(self):
        if self.type == 'gym':
            self.env.render()
        elif self.type == 'unity':
            pass

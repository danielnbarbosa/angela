import unittest
import torch
import numpy as np
from agent_util import OUNoise, ReplayBuffer
np.set_printoptions(threshold=np.inf)

class AgentTestCase(unittest.TestCase):

    def test_ou_noise_sample(self):
        noise = OUNoise(size=(2,), seed=0, mu=0., theta=0.15, sigma=0.2)
        result = np.array([0.35281047, 0.08003144])
        self.assertTrue(np.allclose(noise.sample(), result))

    def test_ou_noise_reset_and_sample(self):
        noise = OUNoise(size=(2,), seed=0, mu=0., theta=0.15, sigma=0.2)
        noise.reset()
        result = np.array([0.35281047, 0.08003144])
        self.assertTrue(np.allclose(noise.sample(), result))

    def test_replay_buffer_sample(self):
        memory = ReplayBuffer(action_size=1, buffer_size=10, batch_size=2, seed=0)
        memory.add(state=1, action=0, reward=0, next_state=2, done=0)
        memory.add(state=2, action=1, reward=1, next_state=3, done=0)
        memory.add(state=3, action=0, reward=0, next_state=4, done=0)
        memory.add(state=4, action=1, reward=-1, next_state=5, done=0)
        memory.add(state=5, action=0, reward=0, next_state=6, done=1)
        states, actions, rewards, next_states, dones = memory.sample()
        self.assertTrue(torch.allclose(states, torch.tensor([[4.], [5.]])))
        self.assertTrue(all(actions == torch.tensor([[1], [0]])))
        self.assertTrue(torch.allclose(rewards, torch.tensor([[-1.], [0.]])))
        self.assertTrue(torch.allclose(next_states, torch.tensor([[5.], [6.]])))
        self.assertTrue(torch.allclose(dones, torch.tensor([[0.], [1.]])))

if __name__ == '__main__':
    unittest.main()

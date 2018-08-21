import sys
sys.path.insert(0, '../libs')
from monitor import train, watch
from dqn_agent import Agent
from environment import Environment


"""
NOTE:
uncomment action discretization in environment.py step()
comment out line just above it
"""

environment = Environment('Pendulum-v0', 'gym', max_steps=1000)
agent = Agent(state_size=3, action_size=9, fc1_units=32, fc2_units=32, seed=0)
train(env, agent, n_episodes=4000, max_t=N_MAX_STEPS)

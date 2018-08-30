import random
import sys
sys.path.insert(0, '../../libs')
from monitor import train_dqn, train_hc, watch, load
from agent_dqn import DQNAgent
from agent_hc import HillClimbingAgent
from environments import GymEnvironment
from models import TwoHiddenLayerQNet, DuelingQNet

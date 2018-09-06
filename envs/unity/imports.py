import random
import sys
sys.path.insert(0, '../../libs')
from monitor import train_dqn, train_hc, watch, load
from agent_dqn import DQNAgent
from agent_hc import HillClimbingAgent
from environments import UnityMLVectorEnvironment, UnityMLVisualEnvironmentSimple
#from environments_experimental import UnityMLVisualEnvironment
from models import TwoHiddenLayerQNet, DuelingQNet, Simple3DConvQNet
#from models_experimental import ConvQNet, DuelingConvQNet, ThreeDConvQNet, OneHiddenLayerWithFlattenQNet

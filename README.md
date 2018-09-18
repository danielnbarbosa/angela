```
█████╗ ███╗   ██╗ ██████╗ ███████╗██╗      █████╗
██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║     ██╔══██╗
███████║██╔██╗ ██║██║  ███╗█████╗  ██║     ███████║
██╔══██║██║╚██╗██║██║   ██║██╔══╝  ██║     ██╔══██║
██║  ██║██║ ╚████║╚██████╔╝███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝
```

ANGELA: Artificial Neural Game Environment Learning Agent


## Introduction

Angela uses reinforcement learning to solve a variety of [Open AI Gym](https://gym.openai.com/) and [Unity ML](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environments.

The code is originally based on [code](https://github.com/udacity/deep-reinforcement-learning) from the Udacity Deep Reinforcement Learning Nanodegree course that I am taking.  It is written in python3 and pytorch.

Consider this a sandbox to test out different RL algorithms in a variety of environments.


## Features

#### DQN Agent
- Standard DQN with Experience Replay and Fixed Q-Targets
- Double DQN
- Dueling Networks
- Prioritized Experience Replay (without importance sampling)
- 3D Convolutional Neural Network with frame stacking

#### Hill Climbing Agent
- Standard Hill Climbing
- Adaptive Noise
- Steepest Ascent
- Simulated Annealing

#### Policy Gradient Agent
- REINFORCE
- 2D Convolutional Neural Network with frame stacking

#### General
- supports discrete state spaces using one-hot encoding
- supports continuous action spaces using discretization
- graph training metrics
- visualize neural network layout
- realtime output of training stats
- save and load model weights
- visualize agent in action


## Installation

#### Pre-requisites
- [Anaconda](https://www.anaconda.com/download/).

#### Step 1: Clone this repo
Clone this repo using `git clone https://github.com/danielnbarbosa/angela.git`.

#### Step 2: Install dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

```
brew install swig
conda create -n angela python=3.6 anaconda
source activate angela
conda install pytorch torchvision -c pytorch
conda install -n angela opencv scikit-image
pip install gym Box2D box2d-py torchsummary unityagents
```

#### Step 3: Install OpenAI Gym
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
```

#### Step 4: Install Unity ML Agents
```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/python
pip install .
```

## Usage
Each environment has its own file with a separate function for each agent type (e.g. dqn, hc, pg).  The file acts as a config file for setting all the various hyperparameters that you may care to tweak.

To start training, use the wrapper script and pass in the desired environment and agent type.  For example to train on the CartPole-v1 environment with the DQN agent:
```
./train.py --env cartpole --agent dqn
```

To load a saved model:
```
./train.py --env cartpole --agent dqn --load=checkpoints/best/cartpole.pth
```

To render an agent:
```
./train.py --env cartpole --agent dqn --render=True
```

## Results
Current best results for each environment are stored in the results directory.  I haven't done an exhaustive hyperparameter search so there is probably lots of room for improvement!

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

Angela is a sandbox for experimenting with reinforcement learning.  It provides a modular way to mix and match various environments, agents and models.  It's great for testing out ideas and building intuition about how an agent learns.

It comes with a variety of built in environments, agents and models but should be fairly straightforward to expand.

Everything is written in python3 and pytorch.


## Features

#### Environments
 - [Open AI Gym](https://gym.openai.com/): Acrobot | Breakout_ram | Breakout | Cartpole | FrozenLake | FrozenLake8x8 | LunarLander | MountainCar | Pendulum | Pong
 - [Unity ML](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector): Basic | Banana | VisualBanana
 - [PyGame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html): FlappyBird

#### Agents
 - DQN: experience replay, fixed Q-targets, double DQN, prioritized experience replay
 - Hill Climbing: adaptive noise, steepest ascent, simulated annealing
 - Policy Gradient: REINFORCE

#### Models
 - DQN: multi-layer perceptron, dueling networks, CNNs
 - Hill Climbing: single-layer perceptron
 - Policy Gradient: multi-layer perceptron, CNNs

#### General
- supports discrete state spaces using one-hot encoding (e.g. FrozenLake)
- supports continuous action spaces using discretization (e.g. Pendulum)
- summarizes model structure
- provides realtime output of training stats
- plots training metrics with matplotlib
- saves and loads model weights
- renders an agent in action


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
conda install -n angela pytorch torchvision -c pytorch
conda install -n angela opencv scikit-image
pip install torchsummary gym Box2D box2d-py unityagents pygame
```

#### Step 3: Install OpenAI Gym
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
cd ..
```

#### Step 4: Install Unity ML Agents
```
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents
pip install .
cd ../..
```

#### Step 5: Install PLE
```
git clone https://github.com/ntasfi/PyGame-Learning-Environment
cd PyGame-Learning-Environment
pip install -e .
cd ..
```

## Usage
To start training, use the `train.py` wrapper script and pass in the desired environment and agent type.  For example to train on the Pong environment with the policy gradient agent:
```
./train.py --env pong --agent pg
```

To load a saved model use `--load`:
```
./train.py --env pong --agent pg --load=checkpoints/best/cartpole.pth
```

To render an agent use `--render`:
```
./train.py --env pong --agent pg --render=True
```


## Project layout
The directory tree structure is as follows:
 - `checkpoints`: saved model weights
 - `envs`: one file per environment.  configuration of hyperparameters pertaining to agent, model and training loop
 - `libs`: shared libraries.  code for models, agents, training loops and various utility functions
 - `results`: current best results for each environment


## Results
Current best results for each environment are stored in the results directory.  I haven't done an exhaustive hyperparameter search so there is probably lots of room for improvement!


## Acknowledgements
Code from the following repos has been used to build this project:
 - [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning) a nanodegree course that I am taking.
 - [Learning Pong from Pixels](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) by Andrej Karpathy.
 - [Deep Policy Gradient Reinforcement Learning](https://github.com/wagonhelm/Deep-Policy-Gradient) by Justin Francis.

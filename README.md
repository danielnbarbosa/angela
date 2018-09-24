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

Angela is a sandbox for experimenting with reinforcement learning.  It provides a modular way to mix and match various environments, agents and models.  It's great for prototyping and getting an agent training quickly without having to re-write a lot of boilerplate.

It comes with a variety of built in environments, agents and models but should be fairly straightforward to expand.

Everything is written in python3 and pytorch.


## Features

#### Environments
 - [Open AI Gym](https://gym.openai.com/): Acrobot | Cartpole | FrozenLake | FrozenLake8x8 | LunarLander | MountainCar | Pendulum | Pong
 - [Unity ML](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector): Basic | Banana | VisualBanana
 - [PyGame Learning Environment](https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html): FlappyBird

#### Agents
 - DQN: experience replay, fixed Q-targets, double DQN, prioritized experience replay
 - Hill Climbing: adaptive noise, steepest ascent, simulated annealing
 - Policy Gradient: REINFORCE

#### Models
 - DQN: multi-layer perceptron, dueling networks, CNN
 - Hill Climbing: single-layer perceptron
 - Policy Gradient: multi-layer perceptron, CNN

#### Misc
- supports discrete state spaces using one-hot encoding (e.g. FrozenLake)
- supports continuous action spaces using discretization (e.g. Pendulum)
- summarizes model structure
- provides realtime output of training stats
- plots training metrics with matplotlib
- saves and loads model weights
- renders an agent in action


## Installation
The below process works for MacOS, but should be easily adopted for Windows.  For AWS see separate [instructions](run_in_aws.md).


#### Pre-requisites
- [Anaconda](https://www.anaconda.com/download/).


#### Step 1: Install dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

```
git clone https://github.com/danielnbarbosa/angela.git
brew install swig
conda create -y -n angela python=3.6 anaconda
source activate angela
conda install -y pytorch torchvision -c pytorch
conda install -y opencv scikit-image
conda uninstall -y ffmpeg # needed for gym monitor
conda install -y -c conda-forge opencv ffmpeg  # needed for gym monitor
pip install torchsummary gym Box2D box2d-py unityagents pygame
cd ..
```

#### Step 2: Install environment toolkits
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
cd ..

git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents
pip install .
cd ../..

git clone https://github.com/ntasfi/PyGame-Learning-Environment
cd PyGame-Learning-Environment
pip install -e .
cd ..
```

## Usage
To start training, use the `learn.py` wrapper script and pass in the desired configuration file.  For example, to train on the Pong environment with the policy gradient agent:
```
./learn.py --cfg pong_pg
```

To load a saved model:
```
./learn.py --cfg pong_pg --load=checkpoints/best/pong.pth
```

To render an agent:
```
./learn.py --cfg pong_pg --render
```

## Project layout
The directory tree structure is as follows:
 - `cfg`: Configuration files with saved hyperparameters.
 - `checkpoints`: Saved model weights.
 - `libs`: Shared libraries.  Code for models, agents, training loops and various utility functions.
 - `results`: Current best training results for each environment.


## Acknowledgements
Code from the following repos has been used to build this project:
 - [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning), a nanodegree course that I am taking.
 - [Learning Pong from Pixels](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) by Andrej Karpathy.
 - [Deep Policy Gradient Reinforcement Learning](https://github.com/wagonhelm/Deep-Policy-Gradient) by Justin Francis.

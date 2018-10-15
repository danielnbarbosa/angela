```
█████╗ ███╗   ██╗ ██████╗ ███████╗██╗      █████╗
██╔══██╗████╗  ██║██╔════╝ ██╔════╝██║     ██╔══██╗
███████║██╔██╗ ██║██║  ███╗█████╗  ██║     ███████║
██╔══██║██║╚██╗██║██║   ██║██╔══╝  ██║     ██╔══██║
██║  ██║██║ ╚████║╚██████╔╝███████╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝
```

ANGELA: Artificial Neural Generated Environment Learning Agent


## Introduction

Angela is a reinforcement learning agent capable of solving a variety environments.  She implements several different RL algorithms (agents) and neural network models.

She can work with both discrete and continuous action spaces allowing her to tackle anything from Atari games to robotic control problems.

She is coded in python3 and pytorch and is getting smarter every day :).

Basically I use Angela as a modular way to test out different RL algorithms in a variety of environments.  It's great for prototyping and getting an agent training quickly without having to re-write a lot of boilerplate.


## Features

#### Environments
 - [**Open AI Gym**](https://gym.openai.com/): Acrobot | [BipedalWalker](results/videos/bipedalwalker.mp4) | [Cartpole](results/videos/cartpole.gif) | FrozenLake | FrozenLake8x8 | [LunarLander](results/videos/lunarlander.gif) | [MountainCar](results/videos/mountaincar.gif) | MountainCarContinuous | Pendulum | [Pong](results/videos/pong.mp4)
 - [**Unity ML**](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector): [Banana](results/videos/banana.gif) | Basic | Crawler | [Reacher](results/videos/reacher.gif) | VisualBanana
 - [**PyGame Learning Environment**](https://pygame-learning-environment.readthedocs.io/en/latest/user/home.html): [FlappyBird](results/videos/flappybird.gif)

#### Agents
 - **dqn**: Deep Q Networks with experience replay, fixed Q-targets, double DQN and prioritized experience replay
 - **hc**: Hill Climbing with adaptive noise, steepest ascent and simulated annealing
 - **pg**: Vanilla Policy Gradient (REINFORCE)
 - **ppo**: Proximal Policy Optimization
 - **ddpg**: Deep Deterministic Policy Gradient

#### Models
 - **dqn**: multi-layer perceptron, dueling networks, CNN
 - **hc**: single-layer perceptron
 - **pg**: multi-layer perceptron, CNN
 - **ppo**: multi-layer perceptron, CNN
 - **ddpg**: low dimensional state spaces

#### Misc
- supports discrete state spaces using one-hot encoding (e.g. FrozenLake)
- supports continuous action spaces using discretization (e.g. Pendulum)
- summarizes model structure
- outputs training stats via console, tensorboard and matplotlib
- saves and loads model weights
- renders an agent in action


## Installation
The below process works for MacOS, but should be easily adopted for Windows.  For AWS see separate [instructions](docs/run_in_aws.md).


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
pip install torchsummary tensorboardX dill gym Box2D box2d-py unityagents pygame
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
 - `compiled_unity_environments`: Pre-compiled unity environments for use with ML Agents.
 - `docs`: Auxiliary documentation.
 - `libs`: Shared libraries.  Code for agents, environments and various utility functions.
 - `logs`: Copies of configs, weights and logging for various training runs.
 - `results`: Current best training results for each environment.
 - `runs`: Output of tensorboard logging.
 - `scripts`: Helper scripts.


## Acknowledgements
Code from the following repos has been used to build this project:
 - [Udacity Deep Reinforcement Learning](https://github.com/udacity/deep-reinforcement-learning), a nanodegree course that I am taking.
 - [Learning Pong from Pixels](https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5) by Andrej Karpathy.
 - [Deep Policy Gradient Reinforcement Learning](https://github.com/wagonhelm/Deep-Policy-Gradient) by Justin Francis.

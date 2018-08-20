## Introduction

ANGELA: Artificial Neural Game Environment Learning Agent

Angela uses reinforcement learning to solve a variety of [Open AI Gym](https://gym.openai.com/) environments.

The code is based on [this code](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn) from the Udacity Deep Reinforcement Learning Nanodegree, a course that I am enrolled in.  I have since made several of my own enhancements to improve it's performance and further my understanding.  I use this code to quickly experiment and build intuition around reinforcement learning.


## Features
- implements the DQN algorithm and DQN enhancements: Double DQN and Dueling Networks
- can work with environments with both discrete and continuous state spaces
- graph metrics related to reward, loss and entropy
- visualize neural network layout
- realtime output of average score, best score, epsilon, experience buffer length, steps taken, wall time
- save and visualize agent training progress over time
- play a sound to indicate when training is finished


## Installation

#### Pre-requisites
- [Anaconda](https://www.anaconda.com/download/).


#### Dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

Mac:
```
conda create -y -n angela python=3.6 anaconda
source activate full
conda install -y pytorch torchvision -c pytorch
conda install -y -n full pympler opencv openmpi
conda install -y -n full -c conda-forge keras tensorflow pydot jupyter_contrib_nbextensions
pip install msgpack gym tensorforce Box2D box2d-py simpleaudio torchsummary unityagents
cd ~/src/ml/openai/gym
pip install -e '.[atari]'
```

Windows:
```
conda create -y -n angela python=3.6 anaconda
activate full
conda install -y pytorch torchvision -c pytorch
conda install -y -n full pympler opencv openmpi
conda install -y -n full -c conda-forge keras tensorflow pydot jupyter_contrib_nbextensions
pip install msgpack gym tensorforce Box2D box2d-py simpleaudio torchsummary unityagents
cd ~/src/ml/openai/gym
pip install -e '.[atari]'
```


## Usage

Each environment has its own configuration file that will run the agent in that environment with the hyperparameters specified.  It will also output some graphs showing reward, loss and entropy over the the training run.

To train the agent just run the desired environment file, for example to train on CartPole-v1:

```
cd envs
python cartpole.py
```

`cartpole.py` contains any environment specific configuration that overrides the defaults.



## Coming soon
- Support for [Unity ML environments](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).
- tile coding for better discretization of continuous action spaces
- convolutional neural networks for learning from pixels

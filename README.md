## Introduction

ANGELA: Artificial Neural Game Environment Learning Agent

Angela uses reinforcement learning to solve a variety of [Open AI Gym](https://gym.openai.com/) environments.

The code is originally based on [this code](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn) from the Udacity Deep Reinforcement Learning Nanodegree, a course that I am enrolled in.


## Features
- implements the DQN algorithm and DQN enhancements: Double DQN and Dueling Networks
- can work with both discrete and continuous state spaces
- graph metrics related to reward, loss and entropy
- visualize neural network layout
- realtime output of average score, best score, epsilon, experience buffer length, steps taken, wall time
- save and visualize agent training progress over time
- play a sound to indicate when training is finished :)


## Installation

#### Pre-requisites
- [Anaconda](https://www.anaconda.com/download/).

#### Step 1: Clone this repo
Clone this repo using `git clone https://github.com/danielnbarbosa/angela.git`.

#### Step 2: Install dependencies
Create an anaconda environment that contains all the required dependencies to run the project.

```
conda create -n angela python=3.6 anaconda
source activate angela
conda install pytorch torchvision -c pytorch
conda install -n angela pympler opencv openmpi
conda install -n angela -c conda-forge keras tensorflow pydot jupyter_contrib_nbextensions
pip install msgpack gym tensorforce Box2D box2d-py simpleaudio torchsummary unityagents
```

#### Step 3: Install OpenAI Gym
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
```


## Usage
Each environment has its own configuration file that will run the agent in that environment with the hyperparameters specified.

To train the agent just run the desired environment file, for example to train on CartPole-v1:

```
cd envs
python cartpole.py
```



## Coming soon
- Support for [Unity ML environments](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).
- tile coding for better discretization of continuous action spaces
- convolutional neural networks for learning from pixels

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

Angela uses reinforcement learning to solve a variety of [Open AI Gym](https://gym.openai.com/) environments.

The code is originally based on [this code](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn) from the Udacity Deep Reinforcement Learning Nanodegree course that I am taking.  It it is written in python3 and pytorch.


#### Algorithms supported
- DQN
- Double DQN
- Dueling Networks
- Prioritized Experience Replay (without importance sampling)

#### Features
- supports discrete state spaces using one-hot encoding
- supports continuous action spaces using discretization
- graph metrics related to reward, loss and entropy
- visualize neural network layout
- realtime output of training stats
- save and load model weights
- visualize agent training
- aural indication when training is finished :)


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
conda install -n angela opencv
pip install gym Box2D box2d-py simpleaudio torchsummary unityagents
```

#### Step 3: Install OpenAI Gym
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e '.[atari]'
```


## Usage
Each environment has its own file that will run the agent in that environment.  The file also acts as a config file for setting all the various hyperparameters that you may care to tweak.

To train the agent just run the desired environment file, for example to train on CartPole-v1:

```
cd envs
python cartpole.py
```

You can also visualize a trained agent by uncommenting the `watch()` function.


## Coming soon
- Support for [Unity ML environments](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).
- convolutional neural networks for learning from pixels

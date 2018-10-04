Training can also be run in AWS to take advantage of cloud GPUs.  Launch Templates are used to automate a lot of the instance setup.


#### Step 1: Launch Spot Instance
- Use Launch Template to start instance (see below for setup)


#### Step 2: On Every SSH
```
source activate pytorch_p36
export DISPLAY=:0
```



## Launch Template
- Region: us-east-1
- AMI ID: ami-18642967
- Instance Type: p2.xlarge
- Security Group: ssh-inbound-only  (you'll need to create your own)
- Key Pair: daniel  (you'll need to create your own)
- Purchasing option: spot
- User Data (see below)


#### User Data Script

```
#!/bin/bash

/usr/bin/X :0 &
apt-get install -y swig
su - -c "conda install -y -n pytorch_p36 opencv scikit-image" ubuntu
su - -c "source activate pytorch_p36; pip install torchsummary gym Box2D box2d-py unityagents pygame" ubuntu
DIR="/home/ubuntu"

# install ml agents toolkit
su -c "cd $DIR/ml-agents; git pull" ubuntu
su - -c "source activate pytorch_p36; cd $DIR/ml-agents/ml-agents; pip install ." ubuntu

# install gym toolkit
su -c "cd $DIR; git clone https://github.com/openai/gym.git" ubuntu
su - -c "source activate pytorch_p36; cd $DIR/gym; pip install -e '.[atari]'" ubuntu

# install PLE toolkit
su -c "cd $DIR; git clone https://github.com/ntasfi/PyGame-Learning-Environment" ubuntu
su - -c "source activate pytorch_p36; cd $DIR/PyGame-Learning-Environment; pip install -e ." ubuntu

# install angela
cd $DIR
su - -c "git clone https://github.com/danielnbarbosa/angela.git" ubuntu

# install udacity unity ml environments
cd $DIR/angela/cfg/compiled_unity_environments
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip
wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip
unzip VisualBanana_Linux.zip Reacher_Linux.zip
rm VisualBanana_Linux.zip Reacher_Linux.zip
chown -R ubuntu:ubuntu VisualBanana_Linux Reacher_Linux
```

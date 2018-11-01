#### Step 1: Get mujoco license
Follow the instructions [here](https://github.com/openai/mujoco-py#obtaining-the-binaries-and-license-key) to get a license key, and install the mujoco binaries.

#### Step 2: Install dependencies
Mujoco requires gcc which doesn't play nice with some of the other packages.  Therefore mujoco gets its own anaconda environment.

```
conda create -y -n mujoco python=3.6 anaconda
source activate mujoco
conda install -y pytorch torchvision -c pytorch
conda install -y opencv scikit-image gcc # gcc needed for mujoco
pip install torchsummary tensorboardX dill gym Box2D box2d-py unityagents pygame
pip install mujoco-py
```

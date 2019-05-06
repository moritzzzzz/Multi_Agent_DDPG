# Multi_Agent_DDPG
Unity Environment: Tennis


# Project: Multi Agent Deep Reinforcement Learning

### Intro

[//]: # (Image References)

[image1]: https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af7955a_tennis/tennis.png "Playing Tennis"


This projects goal is to utilize Deep Reinforcement Learning (DRL) to train two agents to control one tennis bat each to keep the ball from touching the ground.

A trained agent will can be seen in below animation, in which the defined position is marked with a green sphere: 

### Environment
In RL the environment defines what the agent will learn. In this case the environment allows the agent to choose the magnitude of 2 dimensions of its action in each timesequence. The action space is continuous, which poses the essence, as well as the challenge in this project. Every action dimension must be in range -1 to 1.

Also the state space is not discrete, but continuous and is perceived by the agent in 24 dimensions. (24 continuous input features)

The rewards of the environment, which serve as reinforcement for the agent to learn, are assigned when its rules are fullfilled:

A reward of +0.1 is given when the agent hits the balla cross the net. A penalty of -0.01 is given when the ball hits the ground, or if the ball gets hit out-of-bounds.

The task is episodic. To solve the problem the average score must exceed +0.5 for at least 100 consecutive episodes.


### How to use this Github repository to train an agent to control the reacher

The following system prerequisites are required to get it running with my instructions:

- Windows 10 64-Bit.
- Anaconda for Windows.
- GPU with CUDA support(this will not run on CPU only, as its explicitly disabled).
    Cudatoolkit version 9.0.

#### Setting up the Anaconda environment

- Set up conda environment with Python >=3.6
	- Conda create –name <env_name> python=3.6
- Install Jupyter Kernel for this new environment.
    - python -m ipykernel install --user --name <Kernel_name> --display-name "<kernel_name>".
- Download ML-Agents Toolkit beta 0.4.0a.
   - https://github.com/Unity-Technologies/ml-agents/releases/tag/0.4.0a
- install it by running following command, with activated conda environment in the directory of ml-agents, that contains the setup.py.
   - pip install -e . .
- install PyTorch.
    - conda install pytorch torchvision cudatoolkit=9.0 -c pytorch.
- Get Unity Environment designed for this project.
   -  Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip).
    - Place the file in the DRLND GitHub repository, in the `p3_collab_comp/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in the Jupyter notebook file `Tennis.ipynb` to train the agent! For explanations, please see the attached Report.pdf and the comments in the code of model.py and ddpg_agent.py.

### Expected Result
After approximately 500 training episodes the agent will reach the average score of +0.5 for 100 consecutive episodes, which defines this environment as solved.

### Techniques utilized
#### DDPG
In this project an "actor-critic" DDPG, as defined in [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) by Timothy P. Lillicrap et al., with some adjustments to allow multi-agent learning, was implemented. 

#### Improved Exploration
To improve exploration of this agent stochastic Ornstein-Uhlenbeck was added to the selected action, which lead to an improved learning curve. 

#### Stable Deep Reinforcement Learning(DRL)
In order to make the DRL agent more stable in regards to auto-correlation of weights adjustments, the "Fixed Q-Targets" method was utilized combined with a "Replay Buffer". We update the target neural networks (NN) wiht a "soft-update" method after each training step. Thereby the target network (see ["Fixed Q-targets"](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847) ) is iteratively updated with the weights of the trained "regular" NN.

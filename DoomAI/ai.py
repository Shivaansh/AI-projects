import numpy as numpy #for arrays
import torch #PyTorch
import torch.nn as nn #Neural Network, convolution etc (creators)
import torch.nn.functional as F #contains NN functions like max pooling (methods)
import torch.optim as optim #for optimizers
from torch.autograd import Variable #dynamic references, fast computations

#For OpenAI gym
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#importing the other python files
import experience_replay, image_preprocessing

#Build AI
#REMEMBER TO DOCUMENT THE DIFFERENT COMPONENETS OF THE AI

#Apply Deep Convolutional Q-learning
#REMEMBER TO DOCUMENT THE DIFFERENT COMPONENETS OF THE Q LEARNING SYSTEM
import numpy as numpy #for arrays
import torch #PyTorch
import torch.nn as nn #Neural Network, convolution etc (creators)
import torch.nn.functional as F #contains NN functions like max pooling (methods)
import torch.optim as optim #for optimizers
from torch.autograd import Variable #dynamic references, fast computations

#For OpenAI gym
import gym
from gym.wrappers import SkipWrapper

#Contains Doom environemnt, movement and atttack actions
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#importing supporting python modules
import experience_replay, image_preprocessing

#Build AI - brain (CNN + fully connected layers, forward), body, AI
#TODO: REMEMBER TO DOCUMENT THE DIFFERENT COMPONENTS OF THE AI

## Class: Convolutional neural network with forward propagation method
class CNN(nn.Module):
    """
    method: constructor (uses inheritance)
    param self: reference to the object
    param num_of_actions: number of actions that can be taken, varies by environment
    """
    def __init__(self, num_of_actions, ):
        super(CNN, self).__init__()

        #Convolutional variables TODO play with kernel sizes
        self.convolution_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)#input layer
        self.convolution_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4)#layer 1 as input
        self.convolution_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)#layer 2 as input

        #Flattening and hidden layer variables (basically, full connections)
        self.fc1 = nn.Linear(in_features = neuron_count, out_features = 50) #create full connection using Linear method
        self.fc2 = nn.Linear(in_features = 40, out_features = num_of_actions)#output, 

#Apply Deep Convolutional Q-learning
#REMEMBER TO DOCUMENT THE DIFFERENT COMPONENTS OF THE Q LEARNING SYSTEM


"""
######################################## NOTES ############################################
in_channels is set to 1, as the AI only analyzes enemies in black and white. 3 channels needed for RGB images
out_channels is the desired number of features to detect in input image (we have one channel per feature we want to detect)
kernel_size is the size of feature detector matrix
no. of outputs = no. of Q values = no. of actions!
"""
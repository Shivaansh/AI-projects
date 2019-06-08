import numpy as np #for arrays
import random #for randomness in samples from different batches
import os #for loading and saving model
import torch #used for PyTorch, which can handle dynamic graphs
import torch.nn.functional as F #neural network library of PyTorch
import torch.optim as optim #used for optimizations
import torch.autograd as autograd
from torch.autograd import Variable #variables used to store tensor and gradient in one

#CREATE NN ARCHITECTURE:
##NN is an object, so we need to build a class
#FORWARD FUNCTION RETURNS Q VALUES FOR ACTIONS

## Class: The AI agent that drives the car
class DriveMind(nn.Module): #inherits from nn.Module

    """
    method: constructor
    param self: reference to the object
    param input_size: number of input neurons in the NN, which together describe input_state of environment, arg to init function
    param poss_actions: number of neurons in output layer
    """
    def __init__(self, input_size, poss_actions):

        #inherits from nn.Module, enables class to use methods in parent class
        super(DriveMind, self).__init__()

        #attached to object, take value from params
        self.input_size = input_size
        self.poss_actions = poss_actions

        #creating full connections between layers
        #full connection: all neurons of one layer connected to all neurons of another layer
        #params: no. on neurons in each layer (L and R)
        #30 derived by experimentation, is number of neurons in Hidden Layer. Experiment with this, see AI behavior
        self.input_hidden_conn = nn.Linear(input_size, 30)
        self.hidden_output_conn = nn.Linear(30, poss_actions)

    """
    method: performs forward propagation
    param self: the referenced object
    param input_state: input input_state of agent, output based on this
    """
    def forward(self, input_state):

        #get all hidden neurons from first full connection, apply rectifier activation function on them
        x = F.relu(self.input_hidden_conn(input_state))
        q_values = F.relu(self.hidden_output_conn(x))
        return q_values
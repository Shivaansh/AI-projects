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
        
        self.input_hidden_conn = nn.Linear(input_size, 30)
        self.hidden_output_conn = nn.Linear(30, poss_actions)
        #30 derived by experimentation, is number of neurons in Hidden Layer. Experiment with this, see AI behavior

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

## Class: Implementing Experience Replay
class RevisitMemory(object):

    """
    method: constructor
    param self: the referenced object
    param capacity: number of experiences/transitions
    """
    def __init__(self, capacity):

        self.capacity = capacity
        self.memory = [] #contains all previous transition, populated when a future state is visited

    """
    method: adds state (experience) to memory list, ensures memory has 100 transitions
    param self: the referenced object
    param event: experience to be stored in memory
    """
    def push(self, event):
        #transition format: (last state st, new state st+1, last action at, last reward rt)
        self.memory.append(event)
        if(len(self.memory) > self.capacity):
            del self.memory[0]

    """
    method: return random samples from memory
    param self: the referenced object
    param sample_size: size of sample to be returned
    """
    def sample(self, sample_size):
        #use random library method to randomly extract a sample of certain size from memory
        samples = zip(*random.sample(self.memory, sample_size))

        #FORMAT: STATE, ACTION, REWARD
        #take samples, concatenate wrt first dimension, map to PyTorch variables
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


## Class: Implement Deep Q learning model
class Dqn():

    """
    method: constructor
    param self: reference to the object
    param input_size: number of input neurons in the NN, which together describe input_state of environment, arg to init function
    param poss_actions: number of neurons in output layer
    param gamma: learning rate for the q-learning model
    """
    def __init__(self, input_size, poss_actions):
        self.gamma = gamma
        self.reward_window = [] #average of 100 rewards, should increase over time
        self.model = DriveMind(input_size, poss_actions) #instance of NN
        self.memory = RevisitMemory(100000) #100000 transitions instance of memory
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) #connect Adam optimizer to NN
        
        #STATE FORMAT: lSensor, RSensor, FSensor, Orientation, -Orientation
        self.last_state = torch.Tensor(input_size).unsqueeze(0) #unsqueeze creates a fake dimension for tensor

        self.last_action = 0
        self.last_reward = 0.0

    """
    method: decides which action to play
    param self: the referenced object
    param : 
    """
    def select_action(self, ):
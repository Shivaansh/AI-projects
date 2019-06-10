import numpy as np #for arrays
import random #for randomness in samples from different batches
import os #for loading and saving model
import torch #used for PyTorch, which can handle dynamic graphs
import torch.nn.functional as F #neural network library of PyTorch
import torch.optim as optim #used for optimizations
import torch.autograd as autograd
from torch.autograd import Variable #variables used to store tensor and gradient in one

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
        #params: no. of neurons in each layer (L and R)
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

        #randomly extract a sample of certain size from memory
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
    method: decides and returns which action to play
    param self: the referenced object
    param state: the state on the basis of which q-value is computer and action is chosen
    """
    def select_action(self, state):
        probabilities = F.softmax(self.model(Variable(state, volatile = True))*7) #temperature = 7, higher temp, higher P(winning q value)
        action = probabilities.multinomial()
        return action.data[0,0]

    """
    method: implements forward and back propagation
    param self: the referenced object
    param batch_state: current state batch
    param batch_next_state: next state batch
    param batch_reward: current state reward batch
    param batch_action: current state action batch
    """
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #batches from memory become the transitions
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(0)).squeeze(1) #unsqueeze: gather only chosen actions
        next_outputs = self.model(batch_next_state).detach().max(1)[0] #next_outputs needed for target computation, extract max value action q value
        targets = self.gamma*next_outputs + batch_reward #accd to formula
        temporal_difference_loss = F.smooth_l1_loss(outputs, targets)
        self.optimizer.zero_grad() #used for back propagation, reinitialize each loop iteration
        temporal_difference_loss.backward(retain_variables = True)
        self.optimizer.step[] #update weights / synapses

    """
    method: updates the neural network for every new state, returns a next action
    param self: the referenced object
    param reward: the referenced object
    param signal: 
    """
    def update(self, reward, signal):
        #signal is signals from 3 sensors, state is signal itself plus orientation (+-)
        new_state = torch.Tensor(signal).float().unsqueeze(0)
        #need to update memory with new state
        self.memory.push(self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.LongTensor([float(self.last_reward)]))
        action = self.select_action(new_state)
        
        #ensure 100 events have been reached and learning can begin
        if(len(self.memory.memory) > 100):
            #create batches for states, rewards and actions
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #last quantity updated
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        #update batch window
        self.reward_window.append(reward)

        if(len(self.reward_window) > 1000):
            del self.reward_window[0]
        return action  
############################## NOTES ####################################

"""
NN is an object, so we need to build a class
FORWARD FUNCTION RETURNS Q VALUES FOR ACTIONS

To create a fake dimension for a batch, use the unsqueeze method. The fake dimension for
a batch is zero.

numpy: for arrays
random: for randomness in samples from different batches
os: for loading and saving model
torch: PyTorch, which can handle dynamic graphs
torch.nn.functional: neural network library of PyTorch
optim: used for optimizations
Variable: variables used to store tensor and gradient in one

full connection: all neurons of one layer connected to all neurons of another layer

Softmax generates distributed probabilities to all Q-values, which depend on input state
Q values derived from NN, which takes state as input and computes Q values
probs of all q values
Tensors are wrapped into a variable which contain a gradient
volatile = True excludes gradient from graph, saves memory and increases performance
temperature is about the certainty with which we decide our action       
"""
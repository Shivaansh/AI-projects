import numpy as np #for arrays
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
    method: constructor (uses inheritance), think of as 'eyes' of neural network
    param self: reference to the object
    param num_of_actions: number of actions that can be taken, varies by environment
    """
    def __init__(self, num_of_actions):
        super(CNN, self).__init__()

        #Convolutional variables TODO play with kernel sizes
        self.convolution_1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)#input layer
        self.convolution_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4)#layer 1 as input
        self.convolution_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)#layer 2 as input

        #Flattening and hidden layer variables (basically, full connections)
        self.fc1 = nn.Linear(in_features = self.neuron_counter((1, 80, 80)), out_features = 50) #create full connection using Linear method
        self.fc2 = nn.Linear(in_features = 40, out_features = num_of_actions)#output, 

    """
    method: counts the number of neurons in an image. forward propagates in fully connected layers
    param self: reference to the object
    param input_dimension: dimension of the input image
    """
    def neuron_counter(self, input_dimension):
        #dimensions are 80x80 by default in DOOM

        x = Variable(torch.rand(1, *input_dimension)) #extract elements of tuple input_dimension as a list of args
        
        #first layer now activated, propagates to next layer
        x = self.convolution_1(x) #convolution applied to image
        x = F.relu(F.max_pool2d(self.convolution_1(x), 3, 2))#max pooling applied: img, kernel_size, stride
        #relu activates pooled neurons

        #layer 2
        x = F.relu(F.max_pool2d(self.convolution_2(x), 3, 2))

        #layer 3
        x = F.relu(F.max_pool2d(self.convolution_3(x), 3, 2)) #this layer needs to be flattened

        #Take all pixels of all channels and put in one vector
        return x.data.view(1, -1).size(1) 


    """
    method: performs forward propagation
    param self: reference to the object
    param x: the input image updated during propagation
    """
    def forward(self, x):

        # we keep using x repeatedly because x is constantly updated
        x = F.relu(F.max_pool2d(self.convolution_1(x), 3, 2)) #Conv layer
        x = F.relu(F.max_pool2d(self.convolution_2(x), 3, 2)) #Hidden layer
        x = F.relu(F.max_pool2d(self.convolution_3(x), 3, 2)) #Output layer

        #Flatten all pixels of all channels (which becomes input to FCL)
        x = x.view(x.size(0), -1)

        #break linearity using rectifier function
        x = F.relu(self.fc1(x)) #go from flattening layer to hidden layer, then activate hidden layer
        #x becomes output layer
        x = self.fc2(x)
        return x

## Class: The body of the AI, which implements the softmax function
class Softmax_Body(nn.Module):

    """
    method: constructor (uses inheritance)
    param self: reference to the object
    param temperature: the temperature (certainty) of the agent
    """
    def __init__(self, temperature):
        super(Softmax_Body, self).__init__()
        self.T = temperature #class variable for temperature

    """
    method: performs forward propagation from brain to the body
    param self: reference to the object
    param output: the output signals of the brain (in brain's output layer)
    """
    def forward(self, output):

        #get distribution of action probabilities and sample it
        q_val_probls = F.softmax(output * self.T) #create distribution of probabilites of all action q-values

        # retrieve and return possible actions as a multinomial of q value probabilities
        playable_actions = q_val_probls.multinomial()
        return playable_actions


## Class: The AI combines the brain and body
class Geralt:

    """
    method: constructor (uses inheritance)
    param self: reference to the object
    param brain: the CNN
    param body: the Softmax implementation
    """
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    """
    method: constructor (uses inheritance)
    param self: reference to the object
    param input_images: the input images entering the CNN
    param body: the Softmax implementation
    """
    def __call__(self, input_images):
        #create torch variable of input images
        input = Variable(torch.from_numpy(np.array(input_images, dtype = np.float32)))

        #propagate inputs into brain, return output signal of brain
        brain_output = self.brain(input)

        #actions generated by propagating brain output to body
        actions = self.body(brain_output)
        return actions.data.numpy()


############################################ Apply Deep Convolutional Q-learning

#Get Doom environment, images dimensions according to 1x80x80 BnW image format used above
doom_environment = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_environment = gym.wrappers.Monitor(doom_environment, "videos", force = True)
number_actions = doom_environment.action_space.n #get number of actions from doom environment, here equals 7
#REMEMBER TO DOCUMENT THE DIFFERENT COMPONENTS OF THE Q LEARNING SYSTEM

brain_var = CNN(number_actions)
body_var = Softmax_Body(T = 1.0)
ai_agent_var = Geralt(brain = brain_var, body = body_var)


## Experience replay of up to 10000 states combined with eligibility trace of 10 steps
n_step_count = experience_replay.NStepProgress(env = doom_environment, ai = ai_agent_var, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_step_count, capacity = 10000)

# Eligibility trace: implement algorithm found here:
# minimizes rms error between outputs and targets
def eligibility_trace(batch):
    gamma = 0.99 #learning rate
    inputs = []
    targets = []

    for step_series in batch:

        #input is a torch tensor created from a numpy array of first and last state of transition of series
        input = Variable(torch.from_numpy(np.array([step_series[0].state, step_series[-1].state], dtype = np.float32)))
        
        #output is the prediction made by the AI, ie, q-values of all states per transition
        output = brain_var(input) 

        #check if terminal or non terminal state and update reward accordingly
        cumul_reward = 0.0 if step_series[-1].done else output[1].data.max() #output max from a Torch variable

        for step in reversed(step_series[:-1]): # going from last but one element upto first element
            cumul_reward = (cumul_reward * gamma) + step.reward
        state = step_series[0].state #state of first transition
        target = output[0].data #q value of first step. which is the target q value

        #update target for action selected in first step of series
        target[step_series[0].action] = cumul_reward

        # update list of inputs and targets, store learning
        inputs.append(state)
        targets.append(target)

    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

## Class: moving average
class MA:
    """
    method: constructor (uses inheritance)
    param self: reference to the object
    param size: size of rewards list to compute average of
    """
    def __init__(self, size):
        self.reward_list = []
        self.size = size
        
    """
    method: adds rewards to list of rewards
    param self: reference to the object
    param rewards: cumulative reward to append to reward list (after every n steps, according to eligibility trace)
    """
    def add_cumulative_reward(self, rewards):
        
        #if rewards are a list
        if(isinstance(rewards), list):
            self.reward_list += rewards 
        else:
            self.reward_list.append(rewards)

        #ensure a max of 100 elements in list
        while(len(self.reward_list) > self.size):
            del self.reward_list[0]

    """
    method: compute average of rewards
    param self: reference to the object
    """
    def reward_average(self):
        return np.mean(self.reward_list) #from numpy

move_average = MA(100)


######################################### TRAINING ########################################

loss_fn = nn.MSELoss() #Regression generally uses Mean Square Error
optimizer = optim.Adam(brain_var.parameters(), lr = 0.001) #small learning rate for more exploration
num_of_epochs = 100



for epoch in range(1, num_of_epochs+1):
    memory.run_steps(200, 10) #200 runs of 10 steps per epoch
    for batch in memory.sample_batch(128): #return batches of series of 10 step transitions, size 128 (remember this is a BATCH), every 128 steps
        #10 step eliginility trace running in every batch
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        predictions = brain_var(inputs) #inputs sent to CNN, predictions are output
        loss_error = loss_fn(predictions, targets)
        optimizer.zero_grad() #initialize optimizer
        loss_error.backward() #back propagation
        optimizer.step() #update weights

    #Compute average reward per epoch
    reward_steps = n_step_count.rewards_steps()
    move_average.add_cumulative_reward(reward_steps)
    average_reward = move_average.reward_average()
    #

    print("*Epoch: %s, Average reward is: %s" % (str(epoch), str(average_reward)))

    if(average_reward >= 1500):
        print("WOW YOU WIN!!!")
        break

#Close DOOM environment
doom_environment.close()



"""
######################################## NOTES ############################################
in_channels is set to 1, as the AI only analyzes enemies in black and white. 3 channels needed for RGB images
out_channels is the desired number of features to detect in input image (we have one channel per feature we want to detect)
kernel_size is the size of feature detector matrix
no. of outputs = no. of Q values = no. of actions!

To format images:
1. Convert into numpy array
2. Convert numpy array into torch tensor
3. Convert tensor to torch variable containing tensor and gradient for dynamic graphs

tensors by definition are arrays of a single type

Will be using Asynchronous n-step Q-learning: cumulative rewards and experiences on n-steps instead of 1

Only need to update first step of series, as Ai trains on 10 steps, input is first of 10 steps and we get target in this state only
Learning happens after 10 steps are reached

you can add lists in python, but a single element is appended to a python list
"""
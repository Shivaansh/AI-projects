# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0 #last memory x point
last_y = 0 #last memory y point
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(5,3,0.9) #instance of the Dqn class, which is the AI agent
#(a, b, c): state, number of possible actions, learning rate (gamma)
action2rotation = [0,20,-20] #vector, action 0, 1, 2 corresponds to 0, 20, -20 (array indices)
last_reward = 0 #updated each state, can be positive or negative
scores = []

# Initializing the map
first_update = True
def init():
    global sand #array in which cells are pixels of map, 1 = sand, 0 = no sand
    global goal_x #target area of map (goal) - airport
    global goal_y #target area 2 of map (goal) - downtown
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20 #upper left
    goal_y = largeur - 20 #bottom right
    #goal not 0, to avoid touching the wall
    first_update = False

# Initializing the last distance
last_distance = 0 #current distance from car to goal

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0) #between x axis and car direction
    rotation = NumericProperty(0) #latest rotation (0, 20, -20)
    velocity_x = NumericProperty(0) #x coord
    velocity_y = NumericProperty(0) #Y coord
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    # 3 sensors
    
    #sensor 1 = front
    sensor1_x = NumericProperty(0) 
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    
    #sensor 2 = left
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    
    #sensor 3 = right
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    
    #signals received from each sensor
    signal1 = NumericProperty(0) #density around sensor
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    
    #density = number of 1 squares in the 200x200 block, divided by total number of squares
    # EG: 5 squares, so 5 / (20**2) = 5 / 400

    #handles car movement
    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos #updating position with velocity vector
        self.rotation = rotation 
        self.angle = self.angle + self.rotation
        #when car rotates, sensors rotate as well
        #30 is the distance between car and sensors
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
        #get x coord of sensor, then take all cells from -10 to 10, getting a square of 20x20, same for y
        #Sum of all 1's around a sensor, divide by 400 to get density of sand around sensor
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
       
        
        #if sensor 1 approaches RIGHT or LEFT edge of the map, or if approaching UPPER or LOWER edge of map
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1. #maximum sand density, signal to stop- NEGATIVE REWARD
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt): #action to take is decided by this function

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # - ensures explore in both directions
        #action is output of NN. last_reward obtained, last_signal of all 3 sensors + orientation wrt goal
        #brain is an instance of Dqn class
        action = brain.update(last_reward, last_signal) #action to play decided here
        scores.append(brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            #SLOW DOWN ON SAND
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -2
        else: # otherwise
            #USUAL SPEED
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.4
            if distance < last_distance: #slightly positive as approaching goal
                last_reward = 0.1

        #very close to edge (left, right, bottom, top)
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        #update goal position, when goal is reached 
        if distance < 100:
            goal_x = self.width-goal_x
            goal_y = self.height-goal_y
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj): #save for reuse in the future using 'load method'
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()

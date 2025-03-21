# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import torch



class Q_approximator(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Q_approximator, self).__init__()
        self.linear1 = torch.nn.Linear(n_observations, 64)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(64, n_actions)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # self.layer3(x) torch.Size([2, 2])
        print('x', x.shape)
        return x

# Load the pre-trained Q-table

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    # Q net to map (s, a) to expected value
    
    # pickup location: randomly chosen from R, G, Y, B
    # dropoff location: randomly selected from the remaining R, G, Y, B locations

    # can the agent know the grid size from obs? will the grid size be different during testing?
    
    return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


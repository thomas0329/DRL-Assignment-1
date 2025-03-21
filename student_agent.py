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
        
        return x
    
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Load the pre-trained Q-table
Q_net = Q_approximator(n_observations=16, n_actions=6).to(device)
Q_net.load_state_dict(torch.load('./training_best.pt', weights_only=True))
Q_net.eval()

def select_action(state):
    # input: tensor
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return Q_net(state).max(1).indices.view(1, 1)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    # taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs
    # Q net to map (s, a) to expected value
    
    # pickup location: randomly chosen from R, G, Y, B
    # dropoff location: randomly selected from the remaining R, G, Y, B locations

    # can the agent know the grid size from obs? will the grid size be different during testing?
    
    # convert input tuple to tensor
    obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return Q_net(obs).max(1).indices.view(1, 1)
    
    


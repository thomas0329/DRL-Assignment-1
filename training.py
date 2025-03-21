from student_agent import Q_approximator
import torch

import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random
from student_agent import Q_approximator
from simple_custom_taxi_env import SimpleTaxiEnv
import math
from collections import namedtuple, deque
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Get number of actions from gym action space
n_actions = 6
# Get the number of state observations
# state, info = env.reset()
# n_observations = len(state)
n_observations = 16

policy_net = Q_approximator(n_observations, n_actions).to(device)

# target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.choice([0, 1, 2, 3, 4, 5])]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        print('not called')
        return
    print('called')
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    print('batch', batch)
    # batch should be like Transition(state=(tensor([[ 0.0390,  0.9523, -0.0552, -1.4081]]), tensor([[-0.0328,  0.3890,  0.0090, -0.6204]])), action=(tensor([[1]]), tensor([[0]])), next_state=(tensor([[ 0.0580,  1.1481, -0.0834, -1.7176]]), tensor([[-0.0250,  0.1938, -0.0034, -0.3249]])), reward=(tensor([1.]), tensor([1.])))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    # non_final_mask torch.Size([2])
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action should be (tensor([[1]]), tensor([[0]]))
    # action=(5, 3)
    action_batch = torch.cat(batch.action)
    # reward should be (tensor([1.]), tensor([1.])))
    # reward=(-10.1, -0.1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    # state_batch should be of type Tensor
    print('state_batch', state_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)  # 2
    # next_state_values torch.Size([2])
    with torch.no_grad():
        # target network to compute V(st+1) = max,a ​Q(st+1​,a)
        print('non_final_next_states', non_final_next_states.shape)
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values
    # Compute the expected Q values (targets)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 5000
else:
    num_episodes = 50

def train_agent(agent_file, env_config, render=False, episodes=5000):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)

    for episode in range(episodes):

        # Initialize the environment and get its state
        obs, _ = env.reset()    # obs: return of reset
        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

        obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        total_reward = 0
        done = False
        step_count = 0
        stations = [(0, 0), (0, 4), (4, 0), (4,4)]
        
        

        if render:
            env.render_env((taxi_row, taxi_col),
                        action=None, step=step_count, fuel=env.current_fuel)
            time.sleep(0.5)
        while not done:
            
            # action = student_agent.get_action(obs)
            action = select_action(obs)

            obs, reward, terminated, truncated = env.step(action) # obs: return of step
            reward = torch.tensor([reward], device=device)
            taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

            

            done = terminated or truncated
            if terminated:
                next_obs = None
            else:
                next_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
            print('obs=',obs)
            
            obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(obs, action, next_obs, reward)

            
            total_reward += reward
            step_count += 1

            if render:
                env.render_env((taxi_row, taxi_col),
                            action=action, step=step_count, fuel=env.current_fuel)
                
            # Move to the next state
            obs = next_obs

            # Perform one step of the optimization (on the policy network)
            optimize_model()

        print('episode', episode)
        print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
        

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    train_agent("student_agent.py", env_config, render=True)
    
# reference
# for i_episode in range(num_episodes):
    
    
#     for t in count():
#         action = select_action(state)
#         observation, reward, terminated, truncated, _ = env.step(action.item())
#         reward = torch.tensor([reward], device=device)
#         done = terminated or truncated

#         if terminated:
#             next_state = None
#         else:
#             next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

#         # Store the transition in memory
#         memory.push(state, action, next_state, reward)

#         # Move to the next state
#         state = next_state

#         # Perform one step of the optimization (on the policy network, Q net)
#         optimize_model()

#         # # Soft update of the target network's weights
#         # # θ′ ← τ θ + (1 −τ )θ′
#         # target_net_state_dict = target_net.state_dict()
#         # policy_net_state_dict = policy_net.state_dict()
#         # for key in policy_net_state_dict:
#         #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
#         # target_net.load_state_dict(target_net_state_dict)

#         if done:
#             break



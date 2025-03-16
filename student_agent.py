import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

passenger_on_taxi = 0
current_station_idx = 0

def get_direction(taxi_r, taxi_c, station_r, station_c):
    dx = station_r - taxi_r
    dy = station_c - taxi_c
    if dx == 0 and dy == 0:
        return 8  # on station
    elif dx > 0 and dy > 0:
        return 0  # quadrant 1
    elif dx > 0 and dy < 0:
        return 1  # quadrant 2
    elif dx < 0 and dy > 0:
        return 2  # quadrant 3
    elif dx < 0 and dy < 0:
        return 3  # quadrant 4
    elif dx == 0 and dy > 0:
        return 4  # east
    elif dx == 0 and dy < 0:
        return 5  # west
    elif dx > 0 and dy == 0:
        return 6  # south
    elif dx < 0 and dy == 0:
        return 7  # north

def get_discrete_state(obs, passenger_on_taxi, target_station):
    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obs_n, obs_s, obs_e, obs_w,
     p_look, d_look) = obs

    station_positions = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]
    station_r, station_c = station_positions[target_station]
    direction = get_direction(taxi_r, taxi_c, station_r, station_c)
    
    new_pickup = int((not passenger_on_taxi) and (p_look == 1) and ((taxi_r, taxi_c) in station_positions))
    new_dropoff = int(passenger_on_taxi and (d_look == 1) and ((taxi_r, taxi_c) in station_positions))
    
    return (
        direction,
        int(obs_n), int(obs_s), int(obs_e), int(obs_w),
        new_pickup, new_dropoff
    )

def is_near_station(taxi_pos, station_pos):
    taxi_x, taxi_y = taxi_pos
    station_x, station_y = station_pos
    return abs(taxi_x - station_x) + abs(taxi_y - station_y) == 1  

def encode_state(state_tuple):
    (direction, n_, s_, e_, w_, new_pickup, new_dropoff) = state_tuple
    index = 0
    index = index * 9 + direction  # 9 directions
    index = index * 2 + n_
    index = index * 2 + s_
    index = index * 2 + e_
    index = index * 2 + w_
    index = index * 2 + new_pickup
    index = index * 2 + new_dropoff
    return index

class PolicyTable(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self.logits_table = nn.Parameter(torch.zeros(num_states, num_actions))

    def forward(self, state_idx):
        logits = self.logits_table[state_idx]
        probs = F.softmax(logits, dim=-1)
        return probs

NUM_STATES = 9 * (2**4) * (2**2)
NUM_ACTIONS = 6

policy = PolicyTable(NUM_STATES, NUM_ACTIONS)
policy.load_state_dict(torch.load("policy_table_long_checkpoint_2380000.pth", map_location=torch.device('cpu')))
policy.eval()

def get_action(obs):
    global passenger_on_taxi, current_station_idx

    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obs_n, obs_s, obs_e, obs_w,
     p_look, d_look) = obs

    station_positions = [
        (s0_r, s0_c),
        (s1_r, s1_c),
        (s2_r, s2_c),
        (s3_r, s3_c)
    ]
    taxi_pos = (taxi_r, taxi_c)
    target_station = current_station_idx

    state_tuple = get_discrete_state(obs, passenger_on_taxi, target_station)
    state_idx = encode_state(state_tuple)

    with torch.no_grad():
        logits = policy.logits_table[state_idx]
        probs = F.softmax(logits, dim=-1)
        action = torch.distributions.Categorical(probs).sample().item()

    if action == 4:  # pickup
        if not passenger_on_taxi and p_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 1
            current_station_idx = 0 
    elif action == 5:  # dropoff
        if passenger_on_taxi and d_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 0

    if ((not passenger_on_taxi and p_look != 1) or (passenger_on_taxi and d_look != 1)) \
       and is_near_station(taxi_pos, station_positions[current_station_idx]):
        current_station_idx = (current_station_idx + 1) % 4

    return action

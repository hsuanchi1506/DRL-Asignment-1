import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

passenger_on_taxi = 0
next_station = 0

# ------------------------------------------------
# get_direction、get_discrete_state、encode_state
# ------------------------------------------------
def get_direction(taxi_r, taxi_c, station_r, station_c):
    dx = station_r - taxi_r
    dy = station_c - taxi_c
    if dx == 0 and dy == 0:
        return 8  # 0
    elif dx > 0 and dy > 0:
        return 0  # 1
    elif dx > 0 and dy < 0:
        return 1  # 2
    elif dx < 0 and dy > 0:
        return 2  # 3
    elif dx < 0 and dy < 0:
        return 3  # 4
    elif dx == 0 and dy > 0:
        return 4  # e
    elif dx == 0 and dy < 0:
        return 5  # w
    elif dx > 0 and dy == 0:
        return 6  # s
    elif dx < 0 and dy == 0:
        return 7  # n

def get_discrete_state(obs, passenger_on_taxi, next_station):
    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obs_n, obs_s, obs_e, obs_w,
     p_look, d_look) = obs

    station_positions = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]
    station_r, station_c = station_positions[next_station]
    direction = get_direction(taxi_r, taxi_c, station_r, station_c)

    return (
        direction,
        int(obs_n), int(obs_s), int(obs_e), int(obs_w),
        int(p_look), int(d_look),
        int(passenger_on_taxi),
        next_station
    )

def is_near_station(taxi_pos, station_pos):
    taxi_x, taxi_y = taxi_pos
    station_x, station_y = station_pos
    return abs(taxi_x - station_x) + abs(taxi_y - station_y) == 1 

def encode_state(state_tuple):
    (direction,
     n_, s_, e_, w_,
     p_, d_,
     on_taxi,
     next_station) = state_tuple

    index = 0
    # dir (9 )
    index = index * 9 + direction
    # obs_n / obs_s / obs_e / obs_w (each 2 )
    index = index * 2 + n_
    index = index * 2 + s_
    index = index * 2 + e_
    index = index * 2 + w_
    # p_look / d_look / passenger_on_taxi (2)
    index = index * 2 + p_
    index = index * 2 + d_
    index = index * 2 + on_taxi
    # next_station (0~3)
    index = index * 4 + next_station
    return index


class PolicyTable(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        # logits_table: shape (num_states, num_actions)
        self.logits_table = nn.Parameter(torch.zeros(num_states, num_actions))

    def forward(self, state_idx):
        logits = self.logits_table[state_idx]
        probs = F.softmax(logits, dim=-1)
        return probs

# 9 * 2^4 * 2^3 * 4
NUM_STATES = 9 * (2**4) * (2**3) * 4
NUM_ACTIONS = 6


policy = PolicyTable(NUM_STATES, NUM_ACTIONS)
policy.load_state_dict(torch.load("policy_table6_checkpoint_190000.pth", map_location=torch.device('cpu')))
policy.eval() 


def get_action(obs):
    global passenger_on_taxi, next_station
    
    state_tuple = get_discrete_state(obs, passenger_on_taxi, next_station)
    state_idx = encode_state(state_tuple)
    
    with torch.no_grad():
        probs = policy(state_idx)  # shape = (6,)
    
    # # optimal
    # action = torch.argmax(probs).item()

    # sample
    action = torch.distributions.Categorical(probs).sample().item()
    

    (taxi_r, taxi_c,
     s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
     obs_n, obs_s, obs_e, obs_w,
     p_look, d_look) = obs
    

    station_positions = [
        (s0_r, s0_c),  # station0
        (s1_r, s1_c),  # station1
        (s2_r, s2_c),  # station2
        (s3_r, s3_c)   # station3
    ]
    taxi_pos = (taxi_r, taxi_c)
    

    if action == 4:  # pickup
        if not passenger_on_taxi and p_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 1
            next_station = 0
    
    elif action == 5:  # dropoff
        if passenger_on_taxi and d_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 0
    

    if next_station < 3:
        if is_near_station(taxi_pos, station_positions[next_station]):
            if not p_look and not passenger_on_taxi:
                next_station += 1
            if not d_look and passenger_on_taxi:
                next_station += 1
    
    # print(f"passenger_on_taxi: {passenger_on_taxi}, next_station: {next_station}")
    return action
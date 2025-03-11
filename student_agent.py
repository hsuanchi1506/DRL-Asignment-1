import numpy as np
import pickle
import random


with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)


passenger_on_taxi = 0

def get_discrete_state(obs, passenger_on_taxi):
    return tuple(obs) + (passenger_on_taxi,)

def get_action(obs):
    global passenger_on_taxi


    taxi_row, taxi_col, station_0_row, station_0_col, station_1_row, station_1_col, \
    station_2_row, station_2_col, station_3_row, station_3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs

    stations = [(station_0_row, station_0_col), (station_1_row, station_1_col),
                (station_2_row, station_2_col), (station_3_row, station_3_col)]

    if passenger_look == 0:
        passenger_on_taxi = 0

    if passenger_look == 1 and (taxi_row, taxi_col) in stations: 
        passenger_on_taxi = 1

    state = get_discrete_state(obs, passenger_on_taxi)


    if random.uniform(0, 1) < 0.1:
        action = random.randint(0, 5)
    else:
        action = np.argmax(q_table.get(state, np.zeros(6))) 


    if passenger_look == 0:
        passenger_on_taxi = 0


    if action == 4 and passenger_look == 1 and (taxi_row, taxi_col) in stations:
        passenger_on_taxi = 1
    
    return action

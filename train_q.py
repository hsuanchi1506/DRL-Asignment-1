import gym
import numpy as np
import random
import pickle
from collections import defaultdict
from simple_custom_taxi_env import SimpleTaxiEnv

# --------------------------
# Q-Learning Parameters
# --------------------------
ALPHA = 0.01 
GAMMA = 0.99   
EPSILON = 1.0  
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99995
NUM_EPISODES = 100000
MAX_STEPS = 1000


env = SimpleTaxiEnv(fuel_limit=5000)
n_actions = 6
q_table = defaultdict(lambda: np.zeros(n_actions)) 

def get_discrete_state(obs, passenger_on_taxi):
    return tuple(obs) + (passenger_on_taxi,) 


success_count = 0
total_rewards = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    passenger_on_taxi = 0 
    state = get_discrete_state(obs, passenger_on_taxi)
    total_reward = 0
    action = 0
    reward = 0

    for step in range(MAX_STEPS):
        # epsilon greedy
        if random.uniform(0, 1) < EPSILON:
            action = random.randint(0, 5)  
        else:
            action = np.argmax(q_table[state])

        taxi_row, taxi_col, station_0_row, station_0_col, station_1_row, station_1_col, \
        station_2_row, station_2_col, station_3_row, station_3_col, \
        obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
        passenger_look, destination_look = obs
        # print('=' * 50)

        # print("passenger_look: ", passenger_look)
        # print("destination_look: ", destination_look)
        # print(f"(taxi_row, taxi_col): {taxi_row}, {taxi_col}")

        stations = [(station_0_row, station_0_col), (station_1_row, station_1_col),
                    (station_2_row, station_2_col), (station_3_row, station_3_col)]
        
        # print("stations: ", stations)

        # update `passenger_on_taxi`
        bonus = 0
        if action == 4:
            if passenger_look == 1 and (taxi_row, taxi_col) in stations:  
                if passenger_on_taxi == 0:
                    bonus = 5
                passenger_on_taxi = 1
            else:
                bonus = -6
        elif action == 5:
            if destination_look == 1 and (taxi_row, taxi_col) in stations:
                passenger_on_taxi = 0
            else:
                bonus = -6

        # print("(in train) passenger_on_taxi: ", passenger_on_taxi)

        next_obs, reward, done, _ = env.step(action)
        # if action == 4 and reward > -5: 
        #     reward += 0.06

        next_state = get_discrete_state(next_obs, passenger_on_taxi)

        # Q-Learning update
        q_table[state][action] = q_table[state][action] + ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action])
        
        obs = next_obs
        state = next_state
        reward += bonus
        total_reward += reward

        if reward > 10:
            success_count += 1

        if done:
            break
    
    total_rewards.append(total_reward)
    
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(total_rewards[-1000:])
        success_rate = (success_count / 1000) * 100
        print(f"Episode {episode+1}/{NUM_EPISODES}, Avg Reward (last 100 eps): {avg_reward:.2f}, Success Rate: {success_rate:.2f}%, Epsilon: {EPSILON:.4f}")
        success_count = 0

with open("q_table.pkl", "wb") as f:
    pickle.dump(dict(q_table), f)
print("Q-Learning Train done, Q-Table is saved to 'q_table.pkl'")

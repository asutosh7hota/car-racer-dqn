#!/usr/bin/env python

import gym
import random
import time
from collections import deque
import numpy as np
from keras import losses
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, TimeDistributed, Flatten
from keras.layers import Dense
from keras.optimizers import RMSprop
import tensorflow as tf
from skimage.transform import resize
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

def init_model():
    input_shape = (84, 84, 4)
    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(8,8),
                                     activation="relu", strides=4))
    model.add(Conv2D(64, kernel_size=(8,8), activation="relu", strides=2))
    model.add(Conv2D(64, kernel_size=(3,3), activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Flatten())
    model.add(Dense(num_actions))
    rms = RMSprop(lr=1e-4)
    model.compile(optimizer=rms, loss="mse")
    return model

def preprocess_frame(frame, prev):
    m = np.maximum(frame, prev)
    m = rgb2gray(m)
    return resize(m, (84, 84), anti_aliasing=True) 

def calculate_epsilon(decay_period, steps_taken, min_epsilon):
    step_size = (1 - min_epsilon) / decay_period
    return max(min_epsilon, 1. - (step_size * steps_taken))

def train_model(env):
    NUM_EPISODES = int(4)
    #NUM_EPISODES = int(4e5)
    TIME_STEPS = int(1e6)
    memories = deque(maxlen=int(1e6))
    NUM_ACTIONS = env.action_space.n
    GAMMA = 0.99
    EPSILON_DECAY_PERIOD = 2e5
    MIN_EPSILON = 0.1
    C = 1000
    BATCH_SIZE = 32
    SEQ_LEN = 4
    f_recent = deque(maxlen=SEQ_LEN)
    losses = []
    MAX_FRAMES = 10e6
    frames_played = 0
    
    q_model = init_model()
    target_q_model = init_model()
    out_file = open("car-{}.out".format(time.time()), "w+")

    for i_episode in range(NUM_EPISODES):
        if frames_played >= MAX_FRAMES:
            target_q_model.set_weights(q_model.get_weights())
            target_q_model.save('model.h5')
            break
        env.reset()
        action = env.action_space.sample()
        prev, _, _, _ = env.step(action)
        for i in range(SEQ_LEN):
            action = env.action_space.sample()
            observation, reward, terminal, info = env.step(action)
            s_t = preprocess_frame(observation, prev)
            f_recent.append(s_t)
            prev = observation
        phi_t = np.stack(np.array(f_recent), axis=-1)
        loss = 0
        for t in range(TIME_STEPS):
            frames_played += 1
            epsilon = calculate_epsilon(EPSILON_DECAY_PERIOD, frames_played, MIN_EPSILON)
            print("t:", t)
            # With some probability epsilon select random a. 
            if np.random.uniform() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_model.predict(np.expand_dims(np.array(phi_t), axis=0)))
            observation, reward, terminal, info = env.step(action)
            s_t_next = preprocess_frame(observation, prev)
            prev = observation
            f_recent.append(s_t_next)
            phi_t_next = np.stack(np.array(f_recent), axis=-1)
            mem = (phi_t, action, reward, phi_t_next, terminal)
            memories.append(mem)
            s_t = s_t_next
            phi_t = phi_t_next
            mem_batch = random.sample(memories, min(BATCH_SIZE, len(memories)))
            m_phi, m_action, m_reward, m_phi_next, m_terminal = zip(*mem_batch)
            y = q_model.predict(np.array(m_phi))
            q_s_next = target_q_model.predict(np.array(m_phi_next))
            for i, (mt, mr, ma) in enumerate(zip(m_terminal, m_reward, m_action)):
                if mt:
                    y[i][ma] = mr
                else:
                    y[i][ma] = mr + GAMMA * np.max(q_s_next[i])

            l = q_model.train_on_batch(np.array(m_phi), y)
            loss += l
            if t % C == 0:
                print("Adjusting target network")
                target_q_model.set_weights(q_model.get_weights())
                target_q_model.save('car_dqn_model.h5')
            print("epsilon: {}".format(epsilon))
            print("frames played: {}".format(frames_played))
            print("reward: {}".format(reward))
            print("terminal: {}".format(terminal))
            print("info: {}".format(info))
            print("---------------------")
            if terminal:
                losses.append(loss)
                loss_str = "Loss after {}th episode: {}".format(i_episode, loss)
                episode_str = "Episode finished after {} timesteps".format(t+1)
                print(loss_str)
                print(episode_str)
                out_file.write("{}\n{}\n".format(loss_str, episode_str))
                target_q_model.set_weights(q_model.get_weights())
                target_q_model.save('model.h5')
                break
    env.close()
    out_file.close()

    plt.plot(list(range(len(losses))), losses)

if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    num_actions = env.action_space.n
    print("Num actions:", num_actions)
    train_model(env)



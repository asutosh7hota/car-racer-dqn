#!/usr/bin/env python

import gym
import random
import time
import sys
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
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


def init_model(num_actions):
    input_shape = (92, 80, 4)
    model = Sequential()
    model.add(Conv2D(32, input_shape=input_shape, kernel_size=(8, 8),
                     activation="relu", strides=4))
    model.add(Conv2D(64, kernel_size=(4, 4), activation="relu", strides=2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(num_actions))
    rms = RMSprop(lr=1e-5)
    model.compile(optimizer=rms, loss="mse")
    return model


def preprocess_frame(frame, prev):
    m = np.maximum(frame, prev)
    m = rgb2gray(m)
    m = rescale_intensity(m, out_range=np.uint8).astype(np.uint8)
    m = m[26:, :]
    m = m[::2, ::2]
    return m


def calculate_epsilon(decay_period, steps_taken, min_epsilon):
    step_size = (1 - min_epsilon) / decay_period
    return max(min_epsilon, 1. - (step_size * steps_taken))


def preprocess_memory(states, idx, stack_size):
    _, action, reward, _, terminal = states[(idx + stack_size - 1) % len(states)]
    s_t = []
    s_t_next = []
    for i in range(stack_size):
        j = (idx + i) % len(states)
        s_t.append(states[j][0])
        s_t_next.append(states[j][3])
    phi_t = np.stack(np.array(s_t), axis=-1)
    phi_t_next = np.stack(np.array(s_t_next), axis=-1)
    return phi_t, action, reward, phi_t_next, terminal


def train_model(env):
    TIME_STEPS = int(1e6)
    STACK_SIZE = 4
    memories = deque(maxlen=int(1e6) + STACK_SIZE - 1)
    NUM_ACTIONS = env.action_space.n
    GAMMA = 0.99
    EPSILON_DECAY_PERIOD = int(5e5)
    MIN_EPSILON = 0.1
    C = 1000
    BATCH_SIZE = 32
    losses = []
    TRAINING_FRAMES = int(10e6)
    frames_played = 0

    q_model = init_model(NUM_ACTIONS)
    target_q_model = init_model(NUM_ACTIONS)
    curr_time = time.time()
    main_prefix = "main-{}".format(curr_time)
    f_main_out = open("{}.out".format(main_prefix), "w+", buffering=1)
    q_prefix = "q_vals-{}".format(curr_time)
    q_vals = []
    f_q_out = open("{}.out".format(q_prefix), "w+")

    for i_frame in range(TRAINING_FRAMES):
        env.reset()
        action = env.action_space.sample()
        prev, _, _, _ = env.step(action)
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
        s_t = preprocess_frame(observation, prev)
        for i in range(STACK_SIZE):
            action = env.action_space.sample()
            observation, reward, terminal, info = env.step(action)
            s_t_next = preprocess_frame(observation, prev)
            mem = (s_t, action, reward, s_t_next, terminal)
            memories.append(mem)
            s_t = s_t_next
            prev = observation
        phi_t = preprocess_memory(memories, 0, STACK_SIZE)[0]
        loss = 0
        i_episode = 0
        for t in range(TIME_STEPS):
            frames_played += 1
            epsilon = calculate_epsilon(EPSILON_DECAY_PERIOD, frames_played, MIN_EPSILON)
            # With some probability epsilon select random a.
            if np.random.uniform() <= epsilon:
                action = env.action_space.sample()
            else:
                preds = q_model.predict(np.expand_dims(np.array(phi_t), axis=0))
                action = np.argmax(preds)
                q_vals.append(np.max(preds))
                f_q_out.write("{}\n".format(np.max(preds)))

            observation, reward, terminal, info = env.step(action)
            s_t_next = preprocess_frame(observation, prev)
            prev = observation
            mem = (s_t, action, reward, s_t_next, terminal)
            memories.append(mem)
            s_t = s_t_next
            num_memories = len(memories) - STACK_SIZE
            idxs = random.sample(range(num_memories), min(BATCH_SIZE, num_memories))
            mem_batch = []
            for i in idxs:
                mem_batch.append(preprocess_memory(memories, i, STACK_SIZE))
            m_phi, m_action, m_reward, m_phi_next, m_terminal = zip(*mem_batch)
            q_s_next = target_q_model.predict(np.array(m_phi_next))
            y = q_model.predict(np.array(m_phi))
            for i, (mt, mr, ma) in enumerate(zip(m_terminal, m_reward, m_action)):
                if mt:
                    y[i][ma] = mr
                else:
                    y[i][ma] = mr + GAMMA * np.max(q_s_next[i])

            l = q_model.train_on_batch(np.array(m_phi), y)
            loss += l
            if t % C == 0:
                target_q_model.set_weights(q_model.get_weights())
                target_q_model.save('model.h5')
            # print("epsilon: {}".format(epsilon))
            # print("frames played: {}".format(frames_played))
            # print("reward: {}".format(reward))
            # print("terminal: {}".format(terminal))
            # print("info: {}".format(info))
            # print("---------------------")
            if terminal:
                losses.append(loss)
                loss_str = "Loss after {}th episode: {}".format(i_episode, loss)
                i_episode += 1
                episode_str = "Episode finished after {} timesteps".format(t + 1)
                print(loss_str)
                print(episode_str)
                # Save progress.
                f_main_out.write("{}\n{}\n".format(loss_str, episode_str))
                target_q_model.set_weights(q_model.get_weights())
                target_q_model.save('model.h5')
            break
    env.close()
    f_main_out.close()
    f_q_out.close()


if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    train_model(env)

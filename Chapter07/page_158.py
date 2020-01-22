# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Datetime : 2019/12/10 上午12:05
# @Author   : Fangyang
# @Software : PyCharm


import gym
import ptan
import numpy as np
import torch.nn as nn


env = gym.make('CartPole-v0')
net = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 256),
    nn.ReLU(),
    nn.Linear(256, env.action_space.n)
)

action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1)
agent = ptan.agent.DQNAgent(net, action_selector)

obs = np.array([env.reset()], dtype=np.float32)
act = agent(obs)

# Agent's experience
exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=0.99, steps_count=1)
it = iter(exp_source)
print(next(it))

print(1)


if __name__ == '__main__':
    pass



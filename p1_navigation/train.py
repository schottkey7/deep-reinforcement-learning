from unityagents import UnityEnvironment
from collections import namedtuple, deque
from time import time

from model import QNetwork
from agent import Agent

import numpy as np
import random

import matplotlib.pyplot as plt

import torch


def train(
    env,
    agent,
    brain_name,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):
    """Deep Q-Learning.

    Params
    ======
        env (UnityEnvironment): instantiated unity environment
        agent: agent for the network
        brain_name (str): brain name
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    durations = []

    eps = eps_start  # initialize epsilon
    for i_ep in range(1, n_episodes + 1):

        start = time()

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]  # get the current state
        score = 0  # initialize the score

        for t in range(max_t):
            action = agent.act(state, eps)  # select an action
            env_info = env.step(action)[
                brain_name
            ]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        durations.append(time() - start)

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        template = "Episode {}\tAverage Score: {:.2f}\tAverage Time: {:.2f}s"

        print(
            ("\r" + template).format(i_ep, np.mean(scores_window), np.mean(durations)),
            end="",
        )
        if i_ep % 100 == 0:
            print(
                ("\r" + template).format(
                    i_ep, np.mean(scores_window), np.mean(durations)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")

        if np.mean(scores_window) > 13.0:
            print(
                "\nProject solved in {:d} episodes!\tAverage Score: {:.2f}\tAverage Time {:.2f}".format(
                    i_ep - 100, np.mean(scores_window), np.mean(durations)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break

        if np.mean(scores_window) >= 200.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tAverage Time {:.2f}".format(
                    i_ep - 100, np.mean(scores_window), np.mean(durations)
                )
            )
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
    return scores


def main():
    # Initialise the unity environment
    env = UnityEnvironment(file_name="Banana.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of actions
    action_size = brain.vector_action_space_size

    # examine the state space
    state = env_info.vector_observations[0]
    state_size = len(state)

    print("\n--> Training\n")

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    scores = train(env=env, agent=agent, brain_name=brain_name)

    # Plot the scores after training
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("training.png")


if __name__ == "__main__":
    main()

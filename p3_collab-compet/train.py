from collections import deque
from time import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from agent import Agent


def train(env,
          agent,
          brain_name,
          num_agents,
          action_size,
          n_episodes=15000,
          max_t=1000):
    """Deep Q-Learning.

    Params
    ======
        env (UnityEnvironment): instantiated unity environment
        agent: agent for the network
        shared_memory: shared replay buffer
        env_info (UnityEnvironment): instance of the environment
        brain_name (str): brain name
        num_agents (int): number of agents
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    # initialize the score (for each agent)
    scores = []  # list containing scores from each episodes
    scores_window = deque(maxlen=100)
    avg_scores = []

    for i_ep in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()

        states = env_info.vector_observations  # get the current state
        score = np.zeros(num_agents)

        tstart = time()

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)

            score += rewards
            states = next_states

            if np.any(dones):
                break

        current_avg = np.mean(score)
        scores_window.append(current_avg)
        scores.append(current_avg)
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)

        template = "\rEpisode {} | Avg Score: {:.3f} | Score: {:.3f} | Time: {:.2f}s"

        print(template.format(i_ep, avg_score, current_avg, time() - tstart))

        if i_ep % 100 == 0:
            torch.save(agent.actor_local.state_dict(),
                       'saved/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(),
                       'saved/critic_checkpoint.pth')

        if avg_score >= 0.5:
            print("\nEnv solved in {:d} eps!\nAvg Score: {:.5f}\tTime {:.2f}".
                  format(i_ep - 100, avg_score, current_avg,
                         time() - tstart))
            torch.save(agent.actor_local.state_dict(),
                       'saved/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(),
                       'saved/critic_checkpoint.pth')
            break

    return scores, avg_scores


def main():
    # Random seed
    seed = 42

    # Initialise the unity environment
    env = UnityEnvironment(file_name="Tennis.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # number of actions
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(
        states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    print("\n--> Training\n")

    # Initializing agent
    agent = Agent(state_size, action_size, num_agents, seed)
    scores, avg_scores = train(env, agent, brain_name, num_agents, action_size)

    # Plot the scores after training
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("training.png")

    env.close()


if __name__ == "__main__":
    main()

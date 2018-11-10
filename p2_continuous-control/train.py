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
          n_episodes=300,
          max_t=3000):
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
    avg_scores = []
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_ep in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)
        agent.reset()

        tstart = time()

        t = 0

        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones, t)

            states = next_states
            scores += rewards
            t += 1

            if np.any(dones):
                break

        score = np.mean(scores)
        avg_scores.append(score)
        scores_window.append(score)  # save most recent score
        avg_score = np.mean(scores_window)

        template = "Episode {}\tAverage Score: {:.2f}\tTime: {:.2f}s"

        print(template.format(i_ep, avg_score, time() - tstart))
        if i_ep % 100 == 0:
            print(template.format(i_ep, avg_score, time() - tstart))
            torch.save(agent.actor_local.state_dict(),
                       'saved/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(),
                       'saved/critic_checkpoint.pth')

        if avg_score > 30.0:
            print(
                "\nProject solved in {:d} eps!\nAvg Score: {:.2f}\tTime {:.2f}".
                format(i_ep - 100, avg_score,
                       time() - tstart))
            torch.save(agent.actor_local.state_dict(),
                       'saved/actor_checkpoint.pth')
            torch.save(agent.critic_local.state_dict(),
                       'saved/critic_checkpoint.pth')
            break

    return avg_scores


def main():
    # Random seed
    seed = 42

    # Initialise the unity environment
    env = UnityEnvironment(file_name="Reacher-Multiple-Agents.app")

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

    agent = Agent(state_size, action_size, seed)
    scores = train(env, agent, brain_name, num_agents, action_size)

    # Plot the scores after training
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig("training.png")

    env.close()


if __name__ == "__main__":
    main()

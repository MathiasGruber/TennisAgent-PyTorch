import os
import time
from collections import deque

import torch
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def train(
    env, agents, brain_name=None, num_agents=1,
    n_episodes=2000, max_steps=1000,
    thr_score=13.0
):
    """Train agent in the environment
    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
    
    Keyword Arguments:
        brain_name {str} -- brain name for Unity environment (default: {None})
        num_agents {int} -- number of training episodes (default: {1})
        n_episodes {int} -- number of training episodes (default: {2000})
        max_steps {int} -- maximum number of timesteps per episode (default: {1000})
        thr_score {float} -- threshold score for the environment to be solved (default: {30.0})
    """

    # Scores for each episode
    scores = []

    # Last 100 scores
    scores_window = deque(maxlen=100)

    # Average scores & steps after each episode (within window)
    avg_scores = []

    # Best score so far
    best_avg_score = -np.inf

    # Loop over episodes
    time_start = time.time()
    for i in range(1, n_episodes+1):

        # Get initial state from environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        score = 0

        # Reset noise
        agents.reset()

        # Play an episode        
        for _ in range(max_steps):
            action = agents.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agents.step(state, action, rewards, next_state, dones)
            state = next_state
            score += np.max(rewards)
            if np.any(dones):
                break 

        # Update book-keeping variables
        scores_window.append(score)
        scores.append(score)
        avg_score = np.mean(scores_window)
        avg_scores.append(avg_score)
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Info for user every 100 episodes
        n_secs = int(time.time() - time_start)
        
        print(f'Episode {i:6}\t Score: {np.mean(score):.2f}\t Avg: {avg_score:.2f}\t Best Avg: {best_avg_score:.2f} \t Memory: {len(agents.memory):6}\t Seconds: {n_secs:4}')
        time_start = time.time()

        # Check if done
        if avg_score >= thr_score * 2:
            print(f'\nEnvironment solved in {i:d} episodes!\tAverage Score: {avg_score:.2f}')

            # Save the weights
            agents.save_model('./logs/', env.name)

            # Create plot of scores vs. episode
            _, ax = plt.subplots(1, 1, figsize=(7, 5))
            sns.lineplot(range(len(scores)), scores, label='Score', ax=ax)
            sns.lineplot(range(len(avg_scores)), avg_scores, label='Avg Score', ax=ax)
            ax.axhline(thr_score, color='red', lw=2, linestyle='--')
            ax.set_xlabel('Episodes')
            ax.set_xlabel('Score')
            ax.set_title('Environment: {}'.format(env.name))
            ax.legend()
            plt.savefig('./logs/scores_{}.png'.format(env.name))

            break

def test(env, agents, brain_name, checkpoint_actor, checkpoint_critic, num_agents=1):
    """Let pre-trained agent play in environment
    
    Arguments:
        env {UnityEnvironment} -- Unity Environment
        agent {object} -- Agent to traverse environment
        brain_name {str} -- brain name for Unity environment (default: {None})
        checkpoint_actor {str} -- filepath to load network weights for actor
        checkpoint_critic {str} -- filepath to load network weights for critic

    Keyword Arguments:
        num_agents {int} -- number of training episodes (default: {1})
    """

    # Load trained models
    agents.actor_local.load_state_dict(torch.load(checkpoint_actor))
    agents.critic_local.load_state_dict(torch.load(checkpoint_critic))

    # Reset noise
    agents.reset()

    # Initialize & interact in environment
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    for _ in range(600):

        # Get action & perform step
        action = agents.act(state)
        env_info = env.step(action)[brain_name]
        dones = env_info.local_done
        state = env_info.vector_observations
        if np.any(dones):
            break

        # Prevent too fast rendering
        time.sleep(1 / 60.)
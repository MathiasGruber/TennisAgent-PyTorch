import os
import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from libs.models import Actor, Critic
from libs.memory import ReplayBuffer, PrioritizedReplayMemory

BATCH_SIZE = 512        # Batch Size
BUFFER_SIZE = int(1e6)  # Memory capacity
GAMMA = 0.99            # Discount factor
LR_ACTOR = 5e-4         # Actor lerning rate
LR_CRITIC = 5e-4        # Critic learning rate
TAU = 5e-2              # Soft update of target networks
WEIGHT_DECAY = 0        # L2 weight decay for Critic
NOISE_SIGMA = 0.05      # sigma for Ornstein-Uhlenbeck noise


class DDPG():    
    def __init__(self, state_size, action_size, num_agents, random_state=42):
        """Initialize an Agent object.
        
        Arguments:
            state_size (int) -- dimension of each state
            action_size (int) -- dimension of each action
            num_agents (int) -- number of agents in total

        Keyword Arguments:        
            random_state {int} -- seed for torch random number generator (default: {42})
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random_state

        # Whether to use GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Actor Network (incl. target network)
        self.actor_local = Actor(state_size, action_size, random_state=self.seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_state=self.seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (incl. target network)
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents, random_state=self.seed).to(self.device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents, random_state=self.seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((action_size), self.seed)

    def act(self, obs, add_noise=True):
        """Determine agent action based on state
        
        Arguments:
            obs {array_like} -- observation for agent in the environment
        
        Keyword Arguments:
            add_noise {bool} -- whether to add Ornstain-Uhlenbeck noise to action-making
        
        Returns:
            [array-like] -- continous action(s) to take by agent
        """
        obs = torch.from_numpy(obs).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(obs).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the internal state of Ornstain-Uhlenbeck noise to mean"""
        self.noise.reset()

    def learn(self, experiences, next_actions, actions_pred, agent_id, memory, memory_type, idxs=None, is_weight=None):
        """
        Update actor and critic networks based on experience tuple.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Arguments:
            experiences (Tuple[torch.Tensor]) -- tuple of (s, a, r, s', done) 
            next_actions (Tuple[torch.Tensor]) -- next_actions predicted by each agent
            actions_pred (Tuple[torch.Tensor]) -- actions predicted by each agent
            agent_id (int) -- id of the agent
            memory {object} -- Python object for memory being used
            memory_type {str} -- Type of memory (options: replay or per)

        Keyword Arguments:        
            idxs {list} -- list of sample indexes (default: {None})
            is_weight {np.array} -- importance sampling weights (default: {None})            
        """

        # Unpack the tuple
        states, actions, rewards, next_states, dones = experiences

        # Reshape to get observations per agent. i.e. shape of (batchid, agentid, sample)
        rewards = rewards.reshape(-1, self.num_agents, 1)[:,agent_id,:]
        dones = dones.reshape(-1, self.num_agents, 1)[:,agent_id,:]      

        #print("=============", agent_id, not_last_agent)
        #print("states", states.shape, type(states))
        #print("actions", actions.shape, type(actions))
        #print("next_states", next_states.shape, type(next_states))
        #print("rewards", rewards.shape, type(rewards))
        #print("dones", dones.shape, type(dones))
        #print("=============")
        #print("next_actions", next_actions.shape, type(next_actions))
        #print("actions_pred", actions_pred.shape, type(actions_pred))
        #print("=============", agent_id, not_last_agent)

        # UPDATING THE CRITIC
        #####################

        # Get expected Q value for chosen action
        Q_expected = self.critic_local(states, actions)

        # Get Q values from target models for next states
        Q_targets_next = self.critic_target(next_states, next_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Update priorities in Prioritized Experience Replay
        if memory_type == 'per':
            errors = torch.abs(Q_expected - Q_targets).data.cpu().numpy()
            for i in range(len(errors)):
                memory.update(idxs[i], errors[i])

        # Calculate loss based on memory type
        if memory_type == 'per':
            critic_loss = (is_weight * nn.MSELoss(reduction='none')(Q_expected, Q_targets)).mean()
        elif memory_type == 'replay':
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # UPDATING THE ACTOR
        #####################

        # Compute loss
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # UPDATING TARGET NETWORKS
        ##########################
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Update the target model parameters (w_target) as follows:
        w_target = TAU*w_local + (1 - TAU)*w_target
        Arguments:
            local_model {PyTorch model} -- local model to copy from
            target_model {PyTorch model} -- torget model to copy to
            tau {float} -- interpolation parameter
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

class MADDPG():
    
    def __init__(self, state_size, action_size, num_agents, memory='replay', random_state=42):
        """Initialize an Agent object.
        
        Arguments:
            state_size (int) -- dimension of each state
            action_size (int) -- dimension of each action
            num_agents (int) -- number of agents
            random_seed (int) -- random seed

        Keyword Arguments:        
            memory {str} -- which memory type to use (default: {replay}, options: [replay, per])
            random_state {int} -- seed for torch random number generator (default: {42})
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random_state

        # Whether to use GPU or CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Instantiate the agents
        self.agents = [DDPG(state_size, action_size, num_agents) for i in range(num_agents)]

        # Replay memory
        self.memory_type = memory
        if self.memory_type == 'replay':
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        else:
            self.memory = PrioritizedReplayMemory(BUFFER_SIZE, BATCH_SIZE)

    def get_per_sample(self, state, action, reward, next_state, next_action, done):
        """Get the TD error for the current experience samples"""

        # Convert rewards to torch
        reward = Variable(torch.Tensor(reward)).float().to(self.device)

        # Get the temporal difference (TD) error for prioritized replay
        self.critic_local.eval()
        self.critic_target.eval()        
        with torch.no_grad():

            # Get old Q value. 
            old_q = self.critic_local(
                Variable(torch.FloatTensor(state)).to(self.device),
                Variable(torch.FloatTensor(action)).to(self.device)
            )

            # Get the new Q value.
            new_q = reward.unsqueeze(1)
            if not done:
                new_q += GAMMA * torch.max(
                    self.critic_target(
                        Variable(torch.FloatTensor(next_state)).to(self.device),
                        Variable(torch.FloatTensor(next_action)).to(self.device)
                    )
                )
            td_error = abs(old_q - new_q)

        self.critic_local.train()
        self.critic_target.train()

        return td_error
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory. Use random sample from buffer to learn."""

        # Reshape states
        state = state.reshape(1, -1)
        action = action.reshape(1, -1)
        next_state = next_state.reshape(1, -1)
        
        # Different behaviour depending on memory type
        if self.memory_type == 'replay':

            # Standard replay buffer
            self.memory.add(state, action, reward, next_state, done)
            if len(self.memory) > BATCH_SIZE:
                for i, agent in enumerate(self.agents):
                    experiences = self.memory.sample()
                    self.learn(experiences, i)

        elif self.memory_type == 'per':     

            # Prioritized experience replay
            next_action = self.act(next_state)
            td_errors = self.get_per_sample(state, action, reward, next_state, next_action, done)
            self.memory.add(td_errors, (state, action, reward, next_state, next_action, done))

            if len(self.memory) > BATCH_SIZE:
                for i, agent in enumerate(self.agents):
                    experiences, idxs, is_weight = self.memory.sample()
                    self.learn(experiences, i, idxs, is_weight)

        else:
            raise AttributeError(f'Following memory type has not been implemented: {self.memory_type}')

    def act(self, observations, add_noise=True):
        """Determine action for each agent given the agent observation
        
        Arguments:
            observations {array_like} -- observations for each agent in the environment
        
        Keyword Arguments:
            add_noise {bool} -- whether to add Ornstain-Uhlenbeck noise to action-making
        
        Returns:
            [array-like] -- continous action(s) to take by agent
        """
        actions = []
        for agent, observation in zip(self.agents, observations):
            action = agent.act(observation, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)

    def reset(self):
        """Reset the internal states of Ornstain-Uhlenbeck noise to mean for all agents"""
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, agent_id, idxs=None, is_weight=None):
        """
        Update actor and critic networks based on experience tuple.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Arguments:
            experiences (Tuple[torch.Tensor]) -- tuple of (s, a, r, s', done) 
            agent_id (int) -- the id for the agent

        Keyword Arguments:        
            idxs {list} -- list of sample indexes (default: {None})
            is_weight {np.array} -- importance sampling weights (default: {None})
        """

        # Unpack the tuple
        states, _, _, next_states, _ = experiences

        # Reshape to get observations per agent. i.e. shape of (batchid, agentid, sample)
        states = states.reshape(-1, self.num_agents, self.state_size)
        next_states = next_states.reshape(-1, self.num_agents, self.state_size)        

        # The individual agent critics need 'next_actions' and 'actions_pred' for all agents.
        # These quantities are thus calculated here, and passed to each individual agents
        # learning function subsequently
        next_actions = []
        actions_pred = []
        for i, agent in enumerate(self.agents):
            next_actions.append(agent.actor_target(next_states[:,i,:]))
            actions_pred.append(agent.actor_local(states[:,i,:]))
        next_actions = torch.cat(next_actions, dim=1).to(self.device)
        actions_pred = torch.cat(actions_pred, dim=1).to(self.device)

        # Have agent learn
        self.agents[agent_id].learn(
            experiences, next_actions, actions_pred,
            agent_id=i, memory=self.memory, memory_type=self.memory_type, idxs=idxs, is_weight=is_weight
        )

            
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, random_state, mu=0.0, theta=0.15, sigma=NOISE_SIGMA):
        """Initialize parameters and noise process."""
        random.seed(random_state)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()    

    def reset(self):
        """Reset the internal state to mean."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

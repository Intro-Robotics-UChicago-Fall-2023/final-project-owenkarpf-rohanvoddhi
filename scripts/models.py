import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

STATE_DIM = 36
ACTION_DIM = 2
MAX_LINEAR_VEL = .26
MIN_LINEAR_VEL = 0

MAX_ANGULAR_VEL = 1.86
MIN_ANGULAR_VEL = -1.86
ACTOR_LEARNING_RATE = .0001
CRITIC_LEARNING_RATE = .001
TAU = .001
BUFFER_SIZE = 1000000
MAX_EPISODES = 50000
MAX_EP_STEPS = 1000
GAMMA = .995
DEVICE = None

class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(ActorNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(state_shape, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, action_shape),
            nn.Tanh()
        )

    def forward(self, current_state):
        actions = self.actor_network(current_state)
        return actions


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(CriticNetwork, self).__init__()
        self.critic_network = nn.Sequential(
            nn.Linear(state_shape + action_shape, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, current_state, action_to_take):
        current_state = torch.flatten(current_state)
        action_to_take = torch.flatten(action_to_take)
        concatenated_state_and_action = torch.cat((current_state, action_to_take), dim=1)
        q_value = self.critic_network(concatenated_state_and_action)
        return q_value


class NavigationNetwork(object):
    def __init__(self, state_shape, action_shape):
        ## STILL NEED TO UNDERSTAND REPLAY BUFFER BTW

        self.actor_network = ActorNetwork(state_shape, action_shape).to(device)
        self.critic_network = CriticNetwork(state_shape, action_shape).to(device)
        self.actor_network_target = ActorNetwork(state_shape, action_shape).to(device)
        self.critic_network_target = CriticNetwork(state_shape, action_shape).to(device)

        self.actor_network_target.load_state_dict(self.actor_network.state_dict())
        self.critic_network_target.load_state_dict(self.critic_network.state_dict())

        self.actor_ADAM = torch.optim.Adam(self.actor_network.parameters())
        self.critic_ADAM = torch.optim.Adam(self.critic_network.parameters())

        self.writer = SummaryWriter()
        self.num_iter = 0
        self.num_critic_iters = 0
        self.num_actor_iters = 0

    def determine_current_action(self, current_state):
        current_state = torch.Tensor(current_state.reshape(-1)).to(DEVICE)
        actions = self.actor_network(current_state)
        extracted_actions = actions.cuda().data.numpy()
        return extracted_actions

    def train_navigational_ddpg(
            self,
            replay_buffer,
            iters,
            batch_size=32,
            gamma=1,
            policy_noise=0.2,
    ):
        reward_so_far = 0
        for iter in range(iters):
            # sample a batch from the replay buffer
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states \
                = replay_buffer.sample_batch(batch_size)

            current_state = torch.Tensor(batch_states).to(DEVICE)
            current_action = torch.Tensor(batch_actions).to(DEVICE)
            current_action_reward = torch.Tensor(batch_rewards).to(DEVICE)
            cycle_finished = torch.Tensor(batch_dones).to(DEVICE)
            next_state = torch.Tensor(batch_next_states).to(DEVICE)

            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(DEVICE)
            next_action = self.actor_network(next_state)
            target_Q = self.critic_network_target(next_state, next_action)
            target_Q = current_action_reward + (cycle_finished * gamma * target_Q).detach()

            current_Q = self.critic_network(current_state, current_action)

            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_ADAM.zero_grad()
            critic_loss.backward()
            self.critic_ADAM.step()
            actor_grad = -1 * self.critic_network(current_state, self.actor_network(current_state)).mean()
            self.actor_ADAM.zero_grad()
            actor_grad.backward()
            self.actor_ADAM.step()

            for actor_weight, target_actor_weight in zip(
                    self.actor_network.parameters(), self.actor_network_target.parameters()
            ):
                target_actor_weight.data.copy_(
                    TAU * actor_weight.data + (1 - TAU) * target_actor_weight.data
                )
                
            for critic_weight, target_critic_weight in zip(
                    self.critic_network.parameters(), self.critic_network_target.parameters()
            ):
                target_critic_weight.data.copy_(
                    TAU * critic_weight.data + (1 - TAU) * critic_weight.data
                )

            self.num_iter += 1


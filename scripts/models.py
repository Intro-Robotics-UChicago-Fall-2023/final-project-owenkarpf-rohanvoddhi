#!/usr/bin/env python3
import math
import os
import subprocess
import time
from os import path

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point, Pose, Vector3
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Empty
import cv2
import cv_bridge



PROJECT_NAME = 'final_project'
LAUNCH_FILE_NAME = 'modified_maze.launch'

class Environment:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.ranges = None
        self.rgb_img = None
        self.previous_time = None

        self.vel = Twist()
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.vel.linear = Vector3(0, 0, 0)
        self.vel.angular = Vector3(0, 0, 0)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.store_scan_vals)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',
                                    Image, self.image_callback)

        # Instantiate fields to pause and unpause gazebo & reset world to default values
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)


        rospy.init_node("environment")
        arguments_to_run_launchfile = ["roslaunch", PROJECT_NAME, LAUNCH_FILE_NAME]
        subprocess.Popen(arguments_to_run_launchfile)
        time.sleep(10)

    def store_scan_vals(self, scan_data):
        self.ranges = list(scan_data.ranges)
        for i in range(len(self.ranges)):
            if self.ranges[i] == float('inf'):
                self.ranges[i] = 0

    def image_callback(self, msg):
        # print('image callback occurred')
        """
        Image callback function that determines if the objects were seen and the x value for the objects is at the center

        @param msg: the msg received from the callback function
        """
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.rgb_img = image
        
    def is_at_wall(self):
        for dist in self.ranges:
            if dist == float('inf') or dist < .17:
                return True
        return False

    def is_at_target(self):
        image_x = self.rgb_img.shape[1]
        lower_blue = np.array([80, 40, 20])
        upper_blue = np.array([100, 255, 255])
        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        M_blue = cv2.moments(blue_mask)

        if M_blue['m00'] == 0:
            self.blue_x = 0
            self.blue_y = 0

        elif M_blue['m00'] != 0:
            print('mask detected')
            self.blue_x = int(M_blue['m10']/M_blue['m00'])
            self.blue_y = int(M_blue['m01']/M_blue['m00'])
        goal_x = image_x / 2
        dx = np.fabs(goal_x - self.blue_x)

        if dx < 50 and self.ranges[0] < .5:
            return True
        return False


    def determine_reward(self, linear_speed, absolute_turn_speed):
        if self.is_at_target():
            return 500000

        if self.is_at_wall():
            return -5000

        # Greatly penalize small turn values (slow == bad), penalize all turns & greater reward for large linear speeds!
        # if linear_speed < .1:
        #     linear_reward = -.05
        # else:
        linear_reward = linear_speed * 500
        turn_reward = -abs(absolute_turn_speed) * 50
        # if absolute_turn_speed == 0:
        #     turn_reward = 0
        # elif absolute_turn_speed < .2:
        #     turn_reward = -1 * 1/absolute_turn_speed
        # else:
        #     turn_reward = -absolute_turn_speed
        print(f"LINEAR REWARD: {linear_reward}, TURN: {turn_reward}")
        return turn_reward + linear_reward

    def perform_action(self, action):
        # perform the action, return the state after moving robot along with the world being done (robot hit wall or reached goal) and current reward
        new_linear_vel, new_angular_vel = action
        self.vel.linear.x = new_linear_vel
        self.vel.angular.z = new_angular_vel
        self.velocity_publisher.publish(self.vel)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(.05)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause()
        sim_ended_status = self.is_at_target() or self.is_at_wall()
        current_state = self.generate_state_from_scan()
        
        current_reward = self.determine_reward(new_linear_vel, new_angular_vel)

        return current_state,  sim_ended_status, current_reward

    def generate_state_from_scan(self):
        # Codense LIDAR into smaller subsample to reduce training time
        condensed_states = []
        for i in range(36):
            current_subsample = self.ranges[i * 10: i * 10 + 10]
            condensed_states.append(np.median(current_subsample))
        return condensed_states

    def reset_the_world(self):
        # returns initial state and that its not done simming
        self.unpause()
        self.vel.linear.x = 0
        self.vel.angular.z = 0
        self.velocity_publisher.publish(self.vel)
        time.sleep(1)
        self.reset_world()
        time.sleep(1)
        self.pause()
        time.sleep(1)
        current_state = self.generate_state_from_scan()
        return current_state, False
        



import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from numpy import inf

STATE_DIM = 36
ACTION_DIM = 2
MAX_LINEAR_VEL = .26
MIN_LINEAR_VEL = 0
seed = 96024
MAX_ANGULAR_VEL = 1.86
MIN_ANGULAR_VEL = -1.86
ACTOR_LEARNING_RATE = .0001
CRITIC_LEARNING_RATE =  .001
TAU = .05
BUFFER_SIZE = 1500
MAX_ITERATIONS = 50000
MAX_ITERATION_STEPS = 1000
GAMMA = .995
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(ActorNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(state_shape, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, action_shape),
            nn.Tanh()
        )

    def forward(self, current_state):
        actions = self.actor_network(current_state)
        return actions


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(CriticNetwork, self).__init__()
        self.critic_network = nn.Sequential(
            nn.Linear(state_shape + action_shape, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )

    def forward(self, current_state, action_to_take):
        # current_state = torch.flatten(current_state)
        # action_to_take = torch.flatten(action_to_take)
        concatenated_state_and_action = torch.cat((current_state, action_to_take),dim=-1)
        q_value = self.critic_network(concatenated_state_and_action)
        return q_value

class Rewards(Enum):
    GOAL = 500
    CRASH = -500

class ReplayBuffer:
    ''' A replay buffer is used to help train the model but remove the temporal connection between states.
        The idea is to randomly sample from the recent states (in the buffer) and then use those data points
        to train the network.

        size: should be the max number of steps in the training iteration

        state is a list of the condensed lidar values
        action is a tuple of linear and angular velocity
        reward is the value receieved from the environment
        new state is the state after the action
    '''

    def __init__(self, size: int) -> None:
        self.size = size
        self.states = [None for i in range(self.size)]
        self.actions = [None for i in range(self.size)]
        self.rewards = [None for i in range(self.size)]
        self.dones = [None for i in range(self.size)]
        self.new_states = [None for i in range(self.size)]
        self.idx = 0
        self.valid_indices = []
    
    def add(self, state, action, reward: float, done: bool, new_state) -> None:
        # circular buffers
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.new_states[self.idx] = new_state 
        self.valid_indices.append(self.idx)
        self.idx =  (self.idx + 1) % self.size
    
    def sample(self, n_samples: int) -> list:
        indices = np.random.choice(self.valid_indices, min(len(self.valid_indices), n_samples))
        # TODO: make sure no None in this list
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.dones[i] for i in indices],
            [self.new_states[i] for i in indices]
        )


class NavigationNetwork(object):
    def __init__(self, state_shape, action_shape):

        # initialize actor-critic network
        self.actor_network = ActorNetwork(state_shape, action_shape).to(DEVICE)
        self.critic_network = CriticNetwork(state_shape, action_shape).to(DEVICE)
        
        # init optimizers for actor-critic network
        self.actor_ADAM = torch.optim.Adam(self.actor_network.parameters(), lr=0.001)
        self.critic_ADAM = torch.optim.Adam(self.critic_network.parameters())

        # target network starts the same as regular networks
        self.actor_network_target = ActorNetwork(state_shape, action_shape).to(DEVICE)
        self.critic_network_target = CriticNetwork(state_shape, action_shape).to(DEVICE)
        self.actor_network_target.load_state_dict(self.actor_network.state_dict())
        self.critic_network_target.load_state_dict(self.critic_network.state_dict())

        self.num_iter = 0
        self.num_critic_iters = 0
        self.num_actor_iters = 0

    def determine_current_action(self, current_state):
        current_state = torch.Tensor(current_state).to(DEVICE)
        actions = self.actor_network(current_state)
        extracted_actions = actions.cpu().data.numpy()
        return extracted_actions

    def train_navigational_ddpg(
            self,
            replay_buffer,
            iters,
            batch_size=32,
            gamma=1,
            policy_noise=0.1,
    ):
        '''
            This is the function to update the actor critic weights inside the training loop.
            During this function, the world in gazebo is paused.
        '''

        reward_so_far = 0
        for iter in range(iters):
            # sample a batch from the replay buffer
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states \
                = replay_buffer.sample(batch_size)


            current_state = torch.Tensor(batch_states).to(DEVICE)
            current_action = torch.Tensor(batch_actions).to(DEVICE)
            current_action_reward = torch.Tensor(batch_rewards).to(DEVICE)
            cycle_finished = torch.Tensor(batch_dones).to(DEVICE)
            next_state = torch.Tensor(batch_next_states).to(DEVICE)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(DEVICE)
            next_action = self.actor_network(next_state)
            target_Q = self.critic_network_target(next_state, next_action)
            target_Q = current_action_reward.detach().cpu().numpy() + (cycle_finished.detach().cpu().numpy().flatten() *  target_Q.detach().cpu().numpy().flatten())
            target_Q = torch.unsqueeze(torch.tensor(target_Q), 1).to(DEVICE)
            current_Q = self.critic_network(current_state, current_action)
            assert target_Q.size()==current_Q.size()

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_ADAM.zero_grad()
            critic_loss.backward()
            self.critic_ADAM.step()

            # want to maximize the q value (opposite of gradient descent so we use a negative)
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

# add checkpointing

# todo: add exploration noise


def add_noise_and_scale_action(action):
    ang_range = 0.3
    print("init action:", action)
    action[0] = (action[0] + 1) * .13
    action[1] *= ang_range # 1.86
    lin_noise = 0 # np.random.normal(loc=0, scale=0.1)
    ang_noise = 0 # np.random.normal(loc=0, scale=0.05)
    action[0] += lin_noise
    action[1] += ang_noise

    print("scale and noise action:", action)
    action[0] = action[0].clip(0.03,0.26)
    action[1] = action[1].clip(-ang_range,ang_range)
    return action
    # TODO: do we want different distributions for the noise of linear vs angular?

def evaluate(env, model):
    state, done = env.reset_the_world()     
    reward = 0
    for it in range(1200):
        action_to_perform = model.determine_current_action(state)
        action_to_perform = add_noise_and_scale_action(action_to_perform)

        new_state, done, reward = env.perform_action(action_to_perform)
        if reward == Rewards.GOAL:
            print(f"DONE DONE DONE on iteration {it}")
        elif reward  == Rewards.CRASH:
            print(f"CRASH CRASH CRASH on iteration {it}")
        if done:
            break
        state = new_state
    if not done:
        print("Failed to complete in time")
    print(f"Total reward: {reward}")

def train():
    training_iteration = 0

    my_model = NavigationNetwork(STATE_DIM, ACTION_DIM)
    my_env = Environment()
    my_replay_buffer = ReplayBuffer(BUFFER_SIZE)
    time.sleep(2)
    while training_iteration < MAX_ITERATIONS:

        state, done = my_env.reset_the_world()     
        cumulative_reward = 0
        my_replay_buffer = ReplayBuffer(BUFFER_SIZE)

        for it in range(1200):
            action_to_perform = my_model.determine_current_action(state)
            action_to_perform = add_noise_and_scale_action(action_to_perform)
            print(f"it: {it} wall: {my_env.is_at_wall()}, target: {my_env.is_at_target()} action: {action_to_perform}")
            
            new_state, done, reward = my_env.perform_action(action_to_perform)
            if reward == Rewards.GOAL:
                print("DONE DONE DONE")
            elif reward  == Rewards.CRASH:
                print("CRASH CRASH CRASH")
            cumulative_reward += reward
            my_replay_buffer.add(state, action_to_perform, reward, done, new_state)
            if done:
                break
            state = new_state
        
        my_model.train_navigational_ddpg(my_replay_buffer, it)
        print(f"FINISHED AN ITERATION YAY!!! reward: {cumulative_reward}")

        if training_iteration % 100: 
            my_model.save(f"models/iteration{training_iteration}")

if __name__ == "__main__":
    train()
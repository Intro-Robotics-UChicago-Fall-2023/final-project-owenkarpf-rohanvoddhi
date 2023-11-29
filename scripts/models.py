#!/usr/bin/env python3
import math
import os
import subprocess
import time
from os import path
import os
import sys
from math import atan2

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point, Pose, Vector3, Quaternion
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import cv2
import cv_bridge

from tf.transformations import quaternion_from_euler, euler_from_quaternion


PROJECT_NAME = 'final_project'
LAUNCH_FILE_NAME = 'modified_maze.launch'
# TODO: make hidden layer smaller
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
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.odom = None
        self.goal = np.array([-1.2,1.3])
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
        
    def odometry_callback(self, msg):
        self.odom = msg.pose.pose

    def is_at_wall(self):
        for dist in self.ranges:
            if dist == float('inf') or dist <= .17:
                print("ITS THE WALL")
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
            return (False, 200)

        elif M_blue['m00'] != 0:
            self.blue_x = int(M_blue['m10']/M_blue['m00'])
            self.blue_y = int(M_blue['m01']/M_blue['m00'])
        goal_x = image_x / 2
        dx = np.fabs(goal_x - self.blue_x)
        print('mask detected', dx)


        if dx < 50 and self.ranges[0] < .5:
            print("REWARDED.")
            return (True, dx)
        return (False, dx)


    def determine_reward(self, linear_speed, absolute_turn_speed):

        dist_to_goal = np.linalg.norm(np.array([self.odom.position.x,self.odom.position.y])- self.goal,ord=2)
        
        theo_theta = atan2(self.goal[1] - self.odom.position.y, self.goal[0] - self.odom.position.x)

        theta = euler_from_quaternion([
            self.odom.orientation.x, 
            self.odom.orientation.y, 
            self.odom.orientation.z, 
            self.odom.orientation.w])[2]

        theta_diff = abs(theo_theta - theta)
        # print(f"THETA DIFF: {theta_diff}, THEO: {theo_theta}, ACC: {theta}")
        
        at_target, dx = self.is_at_target()
        if at_target:
            return 500000

        if self.is_at_wall():
            return -2000

        # Greatly penalize small turn values (slow == bad), penalize all turns & greater reward for large linear speeds!
        # if linear_speed < .1:
        #     linear_reward = -.05
        # else:
        linear_reward = linear_speed * 500
        turn_reward =  -abs(absolute_turn_speed) * 50
        goal_reward = max(200-dx,0)
        dist_reward = 0 # -dist_to_goal * 20
        theta_reward = 0 # -20 * theta_diff
        # if absolute_turn_speed == 0:
        #     turn_reward = 0
        # elif absolute_turn_speed < .2:
        #     turn_reward = -1 * 1/absolute_turn_speed
        # else:
        #     turn_reward = -absolute_turn_speed
        print(f"LINEAR REWARD: {linear_reward}, TURN: {turn_reward}, MASK: {goal_reward}")
        return turn_reward + linear_reward +  goal_reward

    def perform_action(self, action):
        # perform the action, return the state after moving robot along with the world being done (robot hit wall or reached goal) and current reward
        new_linear_vel, new_angular_vel = action
        self.vel.linear.x = new_linear_vel
        self.vel.angular.z = new_angular_vel
        self.velocity_publisher.publish(self.vel)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(.03)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause()
        sim_ended_status = self.is_at_target()[0] or self.is_at_wall()
        current_state = self.generate_state_from_scan()
        
        current_reward = self.determine_reward(new_linear_vel, new_angular_vel)

        return current_state,  sim_ended_status, current_reward

    def get_odom_list(self):
        return [self.odom.position.x, self.odom.position.y, self.odom.orientation.w,
                 self.odom.orientation.x, self.odom.orientation.y, self.odom.orientation.z]

    def generate_state_from_scan(self):
        # Codense LIDAR into smaller subsample to reduce training time
        return self.ranges[:] + self.get_odom_list()
        condensed_states = []
        for i in range(36):
            current_subsample = self.ranges[i * 10: i * 10 + 10]
            condensed_states.append(np.median(current_subsample))
        return condensed_states


    def reset_model_position_randomly(self):
        initial_state_one = ModelState()
        initial_state_one.model_name = 'turtlebot3'
        initial_state_one.pose.position.x = -.25
        initial_state_one.pose.position.y = 0
        initial_state_one.pose.position.z = 0
        initial_state_one.pose.orientation.x = 0
        initial_state_one.pose.orientation.y = 0
        initial_state_one.pose.orientation.w = 0

        initial_state_two = ModelState()
        initial_state_two.model_name = 'turtlebot3'
        initial_state_two.pose.position.x = -1.2
        initial_state_two.pose.position.y = -1.2
        initial_state_two.pose.position.z = 0
        initial_state_two.pose.orientation.x = 0
        initial_state_two.pose.orientation.y = 0
        initial_state_two.pose.orientation.w = 0

        cur_quat = quaternion_from_euler(0, 0, 0)
        initial_state_three = ModelState()
        initial_state_three.model_name = 'turtlebot3'
        initial_state_three.pose.position.x = -.3
        initial_state_three.pose.position.y = 1.3
        initial_state_three.pose.position.z = 0
        initial_state_three.pose.orientation.x = cur_quat[0]
        initial_state_three.pose.orientation.y = cur_quat[1]
        initial_state_three.pose.orientation.w = cur_quat[2]
        initial_state_three.pose.orientation.z = cur_quat[3]

        cur_quat = quaternion_from_euler(0, 0, math.pi/2)
        initial_state_four = ModelState()
        initial_state_four.model_name = 'turtlebot3'
        initial_state_four.pose.position.x = -.25
        initial_state_four.pose.position.y = -1.2
        initial_state_four.pose.position.z = 0
        initial_state_four.pose.orientation.x = cur_quat[0]
        initial_state_four.pose.orientation.y = cur_quat[1]
        initial_state_four.pose.orientation.w = cur_quat[2]
        initial_state_four.pose.orientation.z = cur_quat[3]

        cur_quat = quaternion_from_euler(0, 0, math.pi/2)
        initial_state_five = ModelState()
        initial_state_five.model_name = 'turtlebot3'
        initial_state_five.pose.position.x = -.1
        initial_state_five.pose.position.y = .90
        initial_state_five.pose.position.z = 0
        initial_state_five.pose.orientation.x = cur_quat[0]
        initial_state_five.pose.orientation.y = cur_quat[1]
        initial_state_five.pose.orientation.w = cur_quat[2]
        initial_state_five.pose.orientation.z = cur_quat[3]
        
        
        arr_of_states = [initial_state_one, initial_state_two, initial_state_three, initial_state_four, initial_state_five]
        
        self.model_state_pub.publish(arr_of_states[4])
        # self.model_state_pub.publish(np.random.choice(arr_of_states))



    def reset_the_world(self):
        # returns initial state and that its not done simming
        self.unpause()
        self.vel.linear.x = 0
        self.vel.angular.z = 0
        self.velocity_publisher.publish(self.vel)
        time.sleep(.1)
        self.reset_world()
        time.sleep(.5)
        self.reset_model_position_randomly()
        time.sleep(.5)
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

ODOM_DIM = 6
STATE_DIM = 360 + ODOM_DIM
ACTION_DIM = 2
MAX_LINEAR_VEL = .26
MIN_LINEAR_VEL = 0
seed = 96024
MAX_ANGULAR_VEL = 1.86
MIN_ANGULAR_VEL = -1.86
ACTOR_LEARNING_RATE = .0001
CRITIC_LEARNING_RATE =  .001
TAU = .001
BUFFER_SIZE = 1500
MAX_ITERATIONS = 50000
MAX_ITERATION_STEPS = 1000
GAMMA = .995
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_ITERATIONS = 250

class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(ActorNetwork, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(state_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, action_shape),
            nn.Tanh()
        )

    def forward(self, current_state):
        actions = self.actor_network(current_state)
        return actions


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super(CriticNetwork, self).__init__()
        self.critic_network = nn.Sequential(
            nn.Linear(state_shape + action_shape, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, current_state, action_to_take):
        # current_state = torch.flatten(current_state)
        # action_to_take = torch.flatten(action_to_take)
        concatenated_state_and_action = torch.cat((current_state, action_to_take),dim=-1)
        q_value = self.critic_network(concatenated_state_and_action)
        return q_value

class Rewards(Enum):
    GOAL = 500
    CRASH = -2000

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
        self.valid_indices = set()
    
    def add(self, state, action, reward: float, done: bool, new_state) -> None:
        # circular buffers
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.new_states[self.idx] = new_state 
        self.valid_indices.add(self.idx)
        self.idx =  (self.idx + 1) % self.size
    
    def sample(self, n_samples: int) -> list:
        indices = np.random.choice(list(self.valid_indices), min(len(self.valid_indices), n_samples))

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
        self.actor_ADAM = torch.optim.Adam(self.actor_network.parameters(), lr=0.003)
        self.critic_ADAM = torch.optim.Adam(self.critic_network.parameters(), lr=0.02)

        # target network starts the same as regular networks
        self.actor_network_target = ActorNetwork(state_shape, action_shape).to(DEVICE)
        self.critic_network_target = CriticNetwork(state_shape, action_shape).to(DEVICE)
        self.actor_network_target.load_state_dict(self.actor_network.state_dict())
        self.critic_network_target.load_state_dict(self.critic_network.state_dict())

        self.num_iter = 0
        self.num_critic_iters = 0
        self.num_actor_iters = 0

    def save(self, path):
        print(os.getcwd())
        # print("SAVING TO PATH:", path)
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor_network.state_dict(), path+"/actor")
        torch.save(self.critic_network.state_dict(), path+"/critic")
        torch.save(self.actor_ADAM.state_dict(), path+"/actor_adam")
        torch.save(self.critic_ADAM.state_dict(), path+"/critic_adam")
        torch.save(self.actor_network_target.state_dict(), path+"/actor_target")
        torch.save(self.critic_network_target.state_dict(), path+"/critic_target")

    def load(self, path):
        self.actor_network.load_state_dict(torch.load(path+"/actor"))
        self.critic_network.load_state_dict(torch.load(path+"/critic"))
        self.actor_ADAM.load_state_dict(torch.load(path+"/actor_adam"))
        self.critic_ADAM.load_state_dict(torch.load(path+"/critic_adam"))
        self.actor_network_target.load_state_dict(torch.load(path+"/actor_target"))
        self.critic_network_target.load_state_dict(torch.load(path+"/critic_target"))

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
            target_Q = current_action_reward.detach().cpu().numpy() + ( (1 - cycle_finished.detach().cpu().numpy().flatten()) * 1 *  target_Q.detach().cpu().numpy().flatten())
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
                    TAU * critic_weight.data + (1 - TAU) *  target_critic_weight.data
                )

            self.num_iter += 1

        for actor_weight, target_actor_weight in zip(
                    self.actor_network.parameters(), self.actor_network_target.parameters()
            ):
                actor_weight.data.copy_(
                    target_actor_weight.data
                )
                
        for critic_weight, target_critic_weight in zip(
                self.critic_network.parameters(), self.critic_network_target.parameters()
        ):
            critic_weight.data.copy_(
                target_critic_weight.data
            )

# add checkpointing

# todo: add exploration noise


def add_noise_and_scale_action(action, iteration, add_noise=True):
    ang_range = 0.6
    print("init action:", action)
    action[0] = (action[0] + 1) * .13
    action[1] *= ang_range # 1.86

    print("scale and noise action:", action)

    if add_noise:
        lin_noise =  np.random.normal(loc=0, scale=.1)
        ang_noise =  np.random.normal(loc=0, scale=1.2 if iteration < WARMUP_ITERATIONS else .6)
        # noise_scale = 1-(iteration/100)
        # lin_noise +=  np.random.uniform(-1,1) * noise_scale
        # ang_noise +=  np.random.uniform(-1,1) * noise_scale
        action[0] += lin_noise
        action[1] += ang_noise
    action[0] = action[0].clip(0.00,0.25)
    action[1] = action[1].clip(-ang_range,ang_range)
    return action
    # TODO: do we want different distributions for the noise of linear vs angular?

def evaluate(env, model):
    state, done = env.reset_the_world()     
    total_reward = 0
    for it in range(1200):
        action_to_perform = model.determine_current_action(state)
        action_to_perform = add_noise_and_scale_action(action_to_perform,it, add_noise=False)

        new_state, done, reward = env.perform_action(action_to_perform)
        total_reward += reward
        if reward == Rewards.GOAL:
            print(f"DONE DONE DONE on iteration {it}")
        elif reward  == Rewards.CRASH:
            print(f"CRASH CRASH CRASH on iteration {it}")
        if done:
            break
        state = new_state
    if not done:
        print("Failed to complete in time")
    print(f"Total reward: {total_reward}")

def eval(it):
    my_model = NavigationNetwork(STATE_DIM, ACTION_DIM)
    my_env = Environment()
    my_model.load(f"models/iteration{it}")
    evaluate(my_env, my_model)


NUM_ITS = 1000
def train():
    training_iteration = 0

    my_model = NavigationNetwork(STATE_DIM, ACTION_DIM)
    my_env = Environment()
    my_replay_buffer = ReplayBuffer(WARMUP_ITERATIONS * NUM_ITS)
    trained = False
    time.sleep(2)
    while training_iteration < MAX_ITERATIONS:

        state, done = my_env.reset_the_world()     
        cumulative_reward = 0
        base_lin_noise =  np.random.normal(loc=0, scale= .2)
        base_ang_noise =  np.random.normal(loc=0, scale= .5)

        if training_iteration < WARMUP_ITERATIONS:
            base_lin_noise =  np.random.normal(loc=0, scale= 1)
            base_ang_noise =  np.random.normal(loc=0, scale= 1)


        escaping = False
        escape_vol = None

        # if training_iteration > WARMUP_ITERATIONS:
        #     my_replay_buffer = ReplayBuffer(NUM_ITS)

        temp = []
        for it in range(NUM_ITS):
            # print("init action:", action_to_perform)
            # lin_noise =  np.random.normal(loc=0, scale=.1 if training_iteration < WARMUP_ITERATIONS else .01)
            # ang_noise =  np.random.normal(loc=0, scale=.2 if training_iteration < WARMUP_ITERATIONS else .03)
            # base_lin_noise += lin_noise
            # ang_noise += ang_noise
            # action_to_perform[0] += (base_lin_noise)
            # action_to_perform[1] += (base_ang_noise)
            action_to_perform = my_model.determine_current_action(state)
            action_to_perform = add_noise_and_scale_action(action_to_perform, iteration=training_iteration)


            # IF TOO CLOSE:
            if training_iteration < WARMUP_ITERATIONS:
                range_val = 20
                turn_range = .4
                if min(my_env.ranges[:range_val]) < turn_range or min(my_env.ranges[-range_val:]) < turn_range:
                    print(" ESCAPING ")
                    if not escaping:
                        escape_vol_lin = np.random.uniform(0,.05)
                        if sum(my_env.ranges[-90: -range_val]) > sum(my_env.ranges[range_val:90]):
                            escape_vol_ang = np.random.uniform(-.6,-.4)
                        else:
                            escape_vol_ang = np.random.uniform(.4,.6)
                        # escape_vol_ang = np.random.uniform(-.8,.8)
                        escaping = True
                    action_to_perform[0] = escape_vol_lin
                    action_to_perform[1] = escape_vol_ang
                elif escaping:
                    escaping = False
                    escape_vol_lin = None
                    escape_vol_ang = None

            if not escaping:
                print("MODEL NAVIGATION")


            print(f"it: {it} wall: {my_env.is_at_wall()}, target: {my_env.is_at_target()[0]} action: {action_to_perform}\n pos:{my_env.get_odom_list()}")
            
            new_state, done, reward = my_env.perform_action(action_to_perform)
            print("DONE:", done)
            if reward == Rewards.GOAL:
                print("DONE DONE DONE")
            elif reward  == Rewards.CRASH:
                print("CRASH CRASH CRASH")
            cumulative_reward += reward
            temp.append((state, action_to_perform, reward, done, new_state))
            # my_replay_buffer.add(state, action_to_perform, reward, done, new_state)
            # if reward > 1000:
            #     my_replay_buffer.add(state, action_to_perform, reward, done, new_state)

            if done:
                break
            state = new_state
        for (state, action_to_perform, reward, done, new_state) in temp:
            my_replay_buffer.add(state, action_to_perform, cumulative_reward, done, new_state)

        if training_iteration == WARMUP_ITERATIONS:
            my_model.train_navigational_ddpg(my_replay_buffer, WARMUP_ITERATIONS * 8, batch_size=32)
            my_replay_buffer = ReplayBuffer(size=300)
            # resample = my_replay_buffer.sample(1000)
            # for tup in zip(*resample):
            #     my_replay_buffer2.add(*tup)
            #     my_replay_buffer = my_replay_buffer2
        elif training_iteration > WARMUP_ITERATIONS + 5:# and training_iteration % 3 == 0:
            my_model.train_navigational_ddpg(my_replay_buffer, 2, batch_size=32) # 5 iterations of batc size 32
        print(f"FINISHED AN ITERATION {training_iteration} YAY!!! reward: {cumulative_reward}")

        if training_iteration % 5 == 0: 
            my_model.save(f"models/iteration{training_iteration}")
            # print("SAVED MODEL")
        training_iteration += 1
if __name__ == "__main__":
    if len(sys.argv) > 1:
        eval(int(sys.argv[1]))
    else:
        train()  
    # eval(170)        train()    # eval(170)
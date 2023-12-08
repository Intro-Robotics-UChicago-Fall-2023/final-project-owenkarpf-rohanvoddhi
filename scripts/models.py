#!/usr/bin/env python3
import math
import os
from os import path
import subprocess
import time
import sys
import cv2
import cv_bridge
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import quaternion_from_euler


# Declare relevant paths
PROJECT_NAME = 'final_project'
LAUNCH_FILE_NAME = 'modified_maze.launch'

class Environment:
    def __init__(self):
        # initialize the enviroment
        self.bridge = cv_bridge.CvBridge()
        self.ranges = None
        self.rgb_img = None
        self.previous_time = None

        # Set the initial velocity to 0s, initial subscribers to camera, lidar scan and odometer
        self.vel = Twist()
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.vel.linear = Vector3(0, 0, 0)
        self.vel.angular = Vector3(0, 0, 0)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.store_scan_vals)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',
                                    Image, self.image_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odometry_callback)
        self.odom = None

        # Instantiate coordinates of where the goal resides on the map
        self.goal = np.array([-1.2,1.3])

        # Instantiate fields to pause and unpause gazebo & reset world to default values
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        # create publisher to publish the model state, which enables movement of the model within gazebo
        self.model_state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)

        # Initialize our enviroment node and also open the gazebo process
        rospy.init_node("environment")
        arguments_to_run_launchfile = ["roslaunch", PROJECT_NAME, LAUNCH_FILE_NAME]
        subprocess.Popen(arguments_to_run_launchfile)
        time.sleep(10)

    def store_scan_vals(self, scan_data):
        # Callback function for the lidar that stores lidar ranges
        self.ranges = list(scan_data.ranges)
        for i in range(len(self.ranges)):
            if self.ranges[i] == float('inf'):
                self.ranges[i] = 0

    def image_callback(self, msg):
        # Callback function that sets the image field to an rgb image generated by the camera
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.rgb_img = image
        
    def odometry_callback(self, msg):
        # Set the odometer pose to the pose message recieved from odometer
        self.odom = msg.pose.pose

    def is_at_wall(self):
        # check if the robot is at the wall based on if any of the lidar scans are less than a threshold (.17)
        for dist in self.ranges:
            if dist == float('inf') or dist <= .17:
                return True
        return False

    def is_at_target(self):
        # Determine if the robot has reached the target by looking at the image recieved by the robot
        image_x = self.rgb_img.shape[1]
        lower_blue = np.array([80, 40, 20])
        upper_blue = np.array([100, 255, 255])
        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        M_blue = cv2.moments(blue_mask)

        if M_blue['m00'] == 0:
            self.blue_x = 0
            self.blue_y = 0
            return (False, 400)

        elif M_blue['m00'] != 0:
            self.blue_x = int(M_blue['m10']/M_blue['m00'])
            self.blue_y = int(M_blue['m01']/M_blue['m00'])
        goal_x = image_x / 2
        dx = np.fabs(goal_x - self.blue_x)
        print("FOUND COLOR", dx)


        if dx < 50 and self.ranges[0] < .6: # Reached the goal
            return (True, dx)
        return (False, dx)


    def determine_reward(self, linear_speed, absolute_turn_speed):
        '''
            This function is used to determine the reward based on the speeds.
            Previously, also used odometry and pose.
        '''
        at_target, dx = self.is_at_target()

        if self.is_at_wall():   # Penalize hitting the wall
            return -3000

        if at_target:           #  Massive reward for reaching the target
            return 500000

        linear_reward = linear_speed * 250              # Reward for higher linear speeds
        turn_reward =  -abs(absolute_turn_speed) * 30   # penalty for turning to much
        goal_reward = max(50-dx/6,0)                    # reward for having the target in sights
        return turn_reward + linear_reward +  goal_reward

    def perform_action(self, action):
        '''
            perform the action, return the state after moving robot along with the world being done
            (robot hit wall or reached goal) and current reward
        '''

        # Publish the action to the robot
        new_linear_vel, new_angular_vel = action
        self.vel.linear.x = new_linear_vel
        self.vel.angular.z = new_angular_vel
        self.velocity_publisher.publish(self.vel)

        # Resume the simulation for .01 seconds
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(.01)
        rospy.wait_for_service("/gazebo/pause_physics")
        self.pause()

        # Update and obtain the robots environment and state
        sim_ended_status = self.is_at_target()[0] or self.is_at_wall()
        current_state = self.generate_state_from_scan()
        current_reward = self.determine_reward(new_linear_vel, new_angular_vel)
        return current_state,  sim_ended_status, current_reward

    def get_odom_list(self):
        ''' 
            Helper function used to access the stored odometry 
        '''
        return [self.odom.position.x, self.odom.position.y, self.odom.orientation.w,
                 self.odom.orientation.x, self.odom.orientation.y, self.odom.orientation.z]

    def generate_state_from_scan(self):
        ''' 
            Codense LIDAR into smaller subsample to reduce training time
        '''
        condensed_states = []
        for i in range(36):
            current_subsample = self.ranges[i * 10: i * 10 + 10]
            condensed_states.append(np.min(current_subsample))
        return condensed_states


    def reset_model_position_randomly(self):
        '''
            This function randomly spawns the robot at one of 5 starting spots
        '''
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
        initial_state_five.pose.position.x = -0.05
        initial_state_five.pose.position.y = .90
        initial_state_five.pose.position.z = 0
        initial_state_five.pose.orientation.x = cur_quat[0]
        initial_state_five.pose.orientation.y = cur_quat[1]
        initial_state_five.pose.orientation.w = cur_quat[2]
        initial_state_five.pose.orientation.z = cur_quat[3]
        
        arr_of_states = [initial_state_one, initial_state_two, initial_state_three, initial_state_four, initial_state_five]
        self.model_state_pub.publish(np.random.choice(arr_of_states))



    def reset_the_world(self):
        ''' 
            Returns initial state and that its not done simming
        '''
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
        

# Training constants
ODOM_DIM = 6
STATE_DIM = 36
ACTION_DIM = 2
MAX_LINEAR_VEL = .2
MIN_LINEAR_VEL = 0
seed = 96024
MAX_ANGULAR_VEL = 1.4
MIN_ANGULAR_VEL = -1.2
ACTOR_LEARNING_RATE = .0004
CRITIC_LEARNING_RATE =  0.00001
TAU = .003
BUFFER_SIZE = 1500000
MAX_ITERATIONS = 50000
MAX_ITERATION_STEPS = 1500
GAMMA = .9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_ITERATIONS = 200
TRAIN_ITERATION_LAG = 10


class Scale(nn.Module):
    '''
        Created a scaling layer to scale the torch output to the desired range of 
        angular and linear velocities
    '''
    def __init__(self, add = [1,0], multiply=[MAX_LINEAR_VEL/2, MAX_ANGULAR_VEL]):
        super().__init__()

        # Store the scale tensors on the appropriate device for later use
        self.add =  torch.tensor(add).to(DEVICE)
        self.multiply =  torch.tensor(multiply).to(DEVICE)
    
    def forward(self, x):
        # apply the scale transformation
        x = torch.add(x, self.add)
        x = torch.mul(x, self.multiply)
        return x

class ActorNetwork(nn.Module):
    '''
        The actor network learns to produce the optimal action for a given state space
    '''
    def __init__(self, state_shape, action_shape):
        super(ActorNetwork, self).__init__()

        # define actor network structuer as a sequential model
        self.actor_network = nn.Sequential(
            nn.Linear(state_shape, 640),
            nn.ReLU(),
            nn.Linear(640, 16),
            nn.ReLU(),
            nn.Linear(16, action_shape),
            nn.Tanh(),
            Scale()
        )

    def forward(self, current_state):
        ''' 
            Calcualate actions from current state
        '''
        actions = self.actor_network(current_state)
        return actions


class CriticNetwork(nn.Module):
    '''
        The critic network learns to approximate the q function in continuous state space.
    '''
    def __init__(self, state_shape, action_shape):
        super(CriticNetwork, self).__init__()
        
        # define critic network architecture
        self.critic_network = nn.Sequential(
            nn.Linear(state_shape + action_shape, 640),
            nn.ReLU(),
            nn.Linear(640, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, current_state, action_to_take):
        ''' 
            Produce the estimated q value from the current state and action
        '''
        concatenated_state_and_action = torch.cat((current_state, action_to_take),dim=-1)
        q_value = self.critic_network(concatenated_state_and_action)
        return q_value

class Rewards(Enum):
    '''
        Enum for reward values
    '''
    GOAL = 500000
    CRASH = -3000

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
        '''
            Initialize the appropriate data structures and components for the replay buferr
        '''
        self.size = size
        self.states = [None for i in range(self.size)]
        self.actions = [None for i in range(self.size)]
        self.rewards = [None for i in range(self.size)]
        self.dones = [None for i in range(self.size)]
        self.new_states = [None for i in range(self.size)]
        self.idx = 0
        self.valid_indices = set()
    
    def add(self, state, action, reward: float, done: bool, new_state) -> None:
        '''
            Use a circular buffer and adds the values to the buffer
        '''
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.new_states[self.idx] = new_state 
        self.valid_indices.add(self.idx)        # add to valid indices to sample from
        self.idx =  (self.idx + 1) % self.size  # move index to next posiition in the buffer
    
    def sample(self, n_samples: int) -> list:
        '''
            Returns a random sample of transitions from the replay buffer to train on
        '''
        indices = np.random.choice(list(self.valid_indices), min(len(self.valid_indices), n_samples))
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            [self.rewards[i] for i in indices],
            [self.dones[i] for i in indices],
            [self.new_states[i] for i in indices]
        )


class NavigationNetwork(object):
    '''
        This is the combined actor-critic network that is trained
    '''
    def __init__(self, state_shape, action_shape):

        # initialize actor-critic network
        self.actor_network = ActorNetwork(state_shape, action_shape).to(DEVICE)
        self.critic_network = CriticNetwork(state_shape, action_shape).to(DEVICE)
        
        # initialize optimizers for actor-critic network
        self.actor_ADAM = torch.optim.Adam(self.actor_network.parameters(), lr=ACTOR_LEARNING_RATE)
        self.critic_ADAM = torch.optim.Adam(self.critic_network.parameters(), lr=CRITIC_LEARNING_RATE)

        # target network starts the same as regular networks
        self.actor_network_target = ActorNetwork(state_shape, action_shape).to(DEVICE)
        self.critic_network_target = CriticNetwork(state_shape, action_shape).to(DEVICE)
        self.actor_network_target.load_state_dict(self.actor_network.state_dict())
        self.critic_network_target.load_state_dict(self.critic_network.state_dict())

        self.num_iter = 0
        
    def save(self, path):
        '''
            Save the models to the corresponding path
        '''
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor_network.state_dict(), path+"/actor")
        torch.save(self.critic_network.state_dict(), path+"/critic")
        torch.save(self.actor_ADAM.state_dict(), path+"/actor_adam")
        torch.save(self.critic_ADAM.state_dict(), path+"/critic_adam")
        torch.save(self.actor_network_target.state_dict(), path+"/actor_target")
        torch.save(self.critic_network_target.state_dict(), path+"/critic_target")

    def load(self, path):
        '''
            Load the models from the corresponding path
        '''
        self.actor_network.load_state_dict(torch.load(path+"/actor"))
        self.critic_network.load_state_dict(torch.load(path+"/critic"))
        self.actor_ADAM.load_state_dict(torch.load(path+"/actor_adam"))
        self.critic_ADAM.load_state_dict(torch.load(path+"/critic_adam"))
        self.actor_network_target.load_state_dict(torch.load(path+"/actor_target"))
        self.critic_network_target.load_state_dict(torch.load(path+"/critic_target"))

    def determine_current_action(self, current_state):
        '''
            Obtain the action from the actor network.
        '''
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
            # Sample a batch from the replay buffer to train on
            batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states \
                = replay_buffer.sample(batch_size)

            # Move the current state variabels to the appropriate device
            current_state = torch.Tensor(batch_states).to(DEVICE)
            current_action = torch.Tensor(batch_actions).to(DEVICE)
            current_action_reward = torch.Tensor(batch_rewards).to(DEVICE)
            cycle_finished = torch.Tensor(batch_dones).to(DEVICE)
            next_state = torch.Tensor(batch_next_states).to(DEVICE)
            

            # Use actor network to determine the action from the current state
            next_action = self.actor_network(next_state)

            # Calculate the target q from the target critic network
            target_Q = self.critic_network_target(next_state, next_action)
            target_Q = current_action_reward.detach().cpu().numpy() + ((1 - cycle_finished.detach().cpu().numpy().flatten()) * GAMMA *  target_Q.detach().cpu().numpy().flatten())
            target_Q = torch.unsqueeze(torch.tensor(target_Q), 1).to(DEVICE)

            # Calculate q from the regular critic network
            current_Q = self.critic_network(current_state, current_action)

            # Define critic loss as MSE between Current and Target critic Q values
            critic_loss = F.mse_loss(current_Q, target_Q)

            # Backprop the loss
            self.critic_ADAM.zero_grad()
            critic_loss.backward()
            self.critic_ADAM.step()

            # Want to maximize the q value (opposite of gradient descent so we use a negative)
            actor_grad = -1 * self.critic_network(current_state, self.actor_network(current_state)).mean()
            self.actor_ADAM.zero_grad()
            actor_grad.backward()
            self.actor_ADAM.step()

            # Copy a fraction (TAU) of the weights to the target network
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

        # Copy back the weights from the critic to the online netowrk
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



def add_noise_and_scale_action(action, iteration=30000, add_noise=True):
    '''
        Add exploration noise to the action vector
    '''
    new_action = np.zeros(2)
    new_action[0] = action[0]
    new_action[1] = action[1]

    if add_noise:
        lin_noise =  np.random.normal(loc=0, scale=.05)
        ang_noise =  np.random.normal(loc=0, scale=.1)
        new_action[0] += lin_noise
        new_action[1] += ang_noise
    new_action[0] = new_action[0].clip(0.00,MAX_LINEAR_VEL)
    new_action[1] = new_action[1].clip(-MAX_ANGULAR_VEL,MAX_ANGULAR_VEL)
    return new_action

def get_random_action():
    '''
        Obtain a random action from the action space (chosen uniformly).
        Used for the first phase of training.
    '''
    new_action = np.zeros(2)
    new_action[0] = np.random.uniform(0,MAX_LINEAR_VEL)
    new_action[1] = np.random.uniform(-MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
    return new_action



def evaluate(env, model):
    '''
        This function is used to evaluate a model in gazebo simulation.
    '''
    state, done = env.reset_the_world()         # Reset the world
    total_reward = 0
    
    for it in range(1200):                      # Max iterations is 1200 based on sim
        action_to_perform = model.determine_current_action(state)
        new_state, done, reward = env.perform_action(action_to_perform)
        total_reward += reward
        if reward == Rewards.GOAL:
            print(f"Reached target on iteration {it}")
        elif reward  == Rewards.CRASH:
            print(f"Crashed on iteration {it}")
        if done:
            break
        state = new_state
    if not done:
        print("Failed to complete in time")
    print(f"Total reward: {total_reward}")

def eval(path: str):
    '''
        Function used to evaluate a model saved at path
    '''
    my_model = NavigationNetwork(STATE_DIM, ACTION_DIM)
    my_env = Environment()
    print("Loading model stored at:",path)
    my_model.load(path)
    evaluate(my_env, my_model)


def train():

    # setup environment and buffers
    my_model = NavigationNetwork(STATE_DIM, ACTION_DIM)
    my_env = Environment()
    my_replay_buffer = ReplayBuffer(BUFFER_SIZE)
    my_success_buffer = ReplayBuffer(40000)
    time.sleep(2)       # sleep to give time for environment to setup in background
    training_iteration = 0
    while training_iteration < MAX_ITERATIONS:

        state, done = my_env.reset_the_world()     
        cumulative_reward = 0

        temp_buffer = []    # use a temp buffer to post process the rewards before adding to buffer
        for it in range(1200):
            if training_iteration > WARMUP_ITERATIONS:
                action_to_perform = my_model.determine_current_action(state)    # Retrieve action from model based on current state
                action_to_perform = add_noise_and_scale_action(action_to_perform, iteration=training_iteration)   # add exploration noise    
            else:
                # uniformly sample from action space to give more data to start up the model
                action_to_perform = get_random_action()
            
            new_state, done, reward = my_env.perform_action(action_to_perform)

            if reward == Rewards.GOAL:
                print("DONE DONE DONE")
            elif reward  == Rewards.CRASH:
                print("CRASH CRASH CRASH")
            cumulative_reward += reward
            temp_buffer.append((state, action_to_perform, reward, done, new_state))
            
            if done:
                break
            state = new_state
        
        if cumulative_reward > 5e3: # add to success buffer if good enough reward because these are rare
            for (state, action_to_perform, reward, done, new_state) in temp_buffer:
                my_success_buffer.add(state, action_to_perform, reward, done, new_state)

        # add transitions to replay buffer
        for (state, action_to_perform, reward, done, new_state) in temp_buffer:
            my_replay_buffer.add(state, action_to_perform, reward, done, new_state)

        # Only train after warmup iterations are done every TRAIN_ITERATION_LAG iterations
        if training_iteration > WARMUP_ITERATIONS and training_iteration % TRAIN_ITERATION_LAG == 0:
            if my_success_buffer.idx >= 5:  # train from success buffer to address class imbalance
                my_model.train_navigational_ddpg(my_success_buffer, 5, batch_size=128)
                
            my_model.train_navigational_ddpg(my_replay_buffer, TRAIN_ITERATION_LAG, batch_size=128)
        print(f"Finished iteration {training_iteration} with reward: {cumulative_reward}")

        # Save model every 25 iterations
        if training_iteration % 25 == 0:
            my_model.save(f"models/iteration{training_iteration}")

        training_iteration += 1


if __name__ == "__main__":
    if len(sys.argv) > 1:   # check command line args for path to evaluate model
        eval(sys.argv[1])
    else:
        train()  

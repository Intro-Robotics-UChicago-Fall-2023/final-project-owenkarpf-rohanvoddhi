#!/usr/bin/env python3
import math
import os
import subprocess
import time
from os import path
import os
import sys
import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan, Image
import cv2
import cv_bridge
from models import ActorNetwork
import torch

MAX_ANGULAR = 1.1
STATE_DIM = 36
ACTION_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RunModelRealWorld():
    '''
        This object is used to keep track of robots state and determine the
        appropriate action to take based on the trained actor model.
    '''

    def __init__(self, path):
        self.bridge = cv_bridge.CvBridge()

        # initialize velocity publisher and objects
        self.vel = Twist()
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel.linear = Vector3(0, 0, 0)
        self.vel.angular = Vector3(0, 0, 0)

        # initialize subscribers
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',
                                    Image, self.image_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.store_scan_vals)
        self.ranges = None

        
        # initialize and load model from disk
        self.actor_network = ActorNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
        self.actor_network.load_state_dict(torch.load(path + "/actor"))

        rospy.sleep(5)  # sleep to give time to setup
        rospy.init_node("RealWorld")


    def run_model(self):
        '''
            This is the action loop for the robot. Continuosly takes in the
            robots state and deterrmines the appropriate action to take. Then 
            publishes the action to the robot.
        '''
        while True:
            # determine action from state
            current_state = self.generate_state_from_scan()
            current_state = torch.Tensor(current_state).to(DEVICE)
            action = self.actor_network(current_state)
            scaled_action = self.scale_action(action)

            # publish velocity
            new_linear_vel, new_angular_vel = scaled_action
            self.vel.linear.x = new_linear_vel
            self.vel.angular.z = new_angular_vel
            self.velocity_publisher.publish(self.vel)

            time.sleep(.1) # give time for the robot to execute the action

            # check robot status to see if over
            at_wall = self.is_at_wall()
            at_target = self.is_at_target()[0]
            if at_target or at_wall:
                if at_target:
                    print("target reached")
                if at_wall:
                    print("at wall")
                return

    def scale_action(self, action):
        '''
            Function to scale the model output. Previous
            iterations of the actor model didn't produce scaled
            actions so is used.
        '''
        new_action = np.zeros(2)
        new_action[0] = (action[0] + 1) * .13
        new_action[1] = action[1] * MAX_ANGULAR
        return new_action

    def store_scan_vals(self, scan_data):
        '''
            Callback function for the lidar that stores lidar ranges
        '''
        self.ranges = list(scan_data.ranges)
        for i in range(len(self.ranges)):
            if self.ranges[i] == float('inf'):
                self.ranges[i] = 0

    def image_callback(self, msg):
        '''
            Callback function that sets the image field to an rgb image generated by the camera
        '''
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.rgb_img = image

    def is_at_wall(self):
        '''
            check if the robot is at the wall based on if any of the lidar scans are less than
              a threshold (.17)
        '''
        for dist in self.ranges:
            if dist == float('inf') or dist <= .17:
                return True
        return False

    def is_at_target(self):
        '''
            Determine if the robot has reached the target by looking at the image recieved by the robot
        '''        
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


        if dx < 50 and self.ranges[0] < .6:
            return (True, dx)
        return (False, dx)


    def generate_state_from_scan(self):
        '''
            Condense LIDAR into smaller subsample to reduce training time
        '''
        while not self.ranges:
            print("No ranges")
            time.sleep(1)

        condensed_states = []
        for i in range(36):
            current_subsample = self.ranges[i * 10: i * 10 + 10]
            condensed_states.append(np.min(current_subsample))
        return condensed_states


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "good_models/iteration3200"

    # instantiate and run model from path
    my_model = RunModelRealWorld(path)
    my_model.run_model()

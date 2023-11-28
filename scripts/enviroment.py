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
        print('image callback occurred')
        """
        Image callback function that determines if the objects were seen and the x value for the objects is at the center

        @param msg: the msg received from the callback function
        """
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.rgb_img = image
        print(self.is_at_target())

    def is_at_wall(self):
        for range in self.ranges:
            if range == float('inf') or range == 0:
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
        print(dx)

        if dx < 50 and self.ranges[0] < .3:
            return True
        return False


    def determine_reward(self, linear_speed, absolute_turn_speed):
        if self.is_at_target():
            return 500

        if self.is_at_wall():
            return -500

        # Greatly penalize small turn values (slow == bad), penalize all turns & greater reward for large linear speeds!
        if linear_speed < .1:
            linear_reward = -.05
        else:
            linear_reward = linear_speed
        if absolute_turn_speed == 0:
            turn_reward = 0
        elif absolute_turn_speed < .2:
            turn_reward = -1 * 1/absolute_turn_speed
        else:
            turn_reward = -absolute_turn_speed

        return turn_reward + linear_reward

    def perform_action(self, action):
        # perform the action, return the state after moving robot along with the world being done (robot hit wall or reached goal) and current reward
        new_linear_vel, new_angular_vel = action
        self.vel.linear.x = new_linear_vel
        self.vel.angular.z = new_angular_vel
        self.velocity_publisher.publish(self.vel)
        rospy.wait_for_service("/gazebo/unpause_physics")
        self.unpause()
        time.sleep(.1)
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
        

Environment()

#!/usr/bin/env python3
import math
import os
import subprocess
import time
from os import path

import numpy as np
import rospy
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist, Point, Pose, Vector3, Quaternion
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import LaserScan, Image
from std_srvs.srv import Empty
import cv2
import cv_bridge



PROJECT_NAME = 'particle_filter_project'
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

        # state_msg = ModelState()
        # state_msg.model_name = 'turtlebot3'
        # state_msg.pose.position.x = .2
        # state_msg.pose.position.y = .2
        # state_msg.pose.position.z = 0.3
        # state_msg.pose.orientation.x = .4
        # state_msg.pose.orientation.y = .3
        # state_msg.pose.orientation.w = .2
        # self.model_state_pub.publish(state_msg)
        # print("State set theoretically!")


    def store_scan_vals(self, scan_data):
        print('ranges given')
        self.ranges = list(scan_data.ranges)
        for i in range(len(self.ranges)):
            if self.ranges[i] == 0:
                self.ranges[i] = float('inf')

    def image_callback(self, msg):
        print('image callback occurred')
        """
        Image callback function that determines if the objects were seen and the x value for the objects is at the center

        @param msg: the msg received from the callback function
        """
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        self.rgb_img = image

    def wall_detected(self):
        for range in self.ranges:
            if range == float('inf') or range == 0:
                return True
        return False

    def at_target(self):
        image_x = self.rgb_img.shape[1]
        lower_blue = np.array([90, 120, 100])
        upper_blue = np.array([95, 255, 255])
        hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        M_blue = cv2.moments(blue_mask)

        if M_blue['m00'] == 0:
            self.blue_x = None
            self.blue_y = None

        elif M_blue['m00'] != 0:
            self.blue_x = int(M_blue['m10']/M_blue['m00'])
            self.blue_y = int(M_blue['m01']/M_blue['m00'])
        goal_x = image_x / 2
        dx = np.fabs(image_x - goal_x)

        if dx < 50 and self.ranges[0] < .5:
            return True
        return False


    def determine_reward(self, linear_speed, absolute_turn_speed):
        if self.at_target():
            return 500

        if self.wall_detected():
            return -500

        # Greatly penalize small turn values (slow == bad), penalize all turns & greater reward for large linear speeds!
        if linear_speed < .1:
            linear_reward = -.05
        else:
            linear_reward = linear_speed
        if absolute_turn_speed == 0:
            turn_reward = 0
        elif absolute_turn_speed < .4:
            turn_reward = -1 * 1/absolute_turn_speed
        else:
            turn_reward = -absolute_turn_speed

        return turn_reward + linear_reward

    def perform_action(self, action):
        new_linear_vel, new_angular_vel = action
        self.vel.linear.x = new_linear_vel
        self.vel.angular.z = new_angular_vel
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(.1)
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

    def reset_the_world(self):
        self.reset_world()
        

Environment()

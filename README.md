# Minimizing Time For Maze Completition with Deep-Deterministic Policy Gradients

## Owen Karpf & Rohan Voddhi


# Project Description


### Project Goal

The general idea for our project is that we wanted to use reinforcement learning to constantly set linear and angular velocities for a turtlebot3 robot such that the linera and angular velocities minimized the time necessary for the turtlebot to reach the goal. Due to impracticalities in regards to training time (navigation of the entire maze would take thousands upon thousands of iterations, if it worked at all), along with issues in actually tuning our model, we decided to constrain ourself towards having the robot learn to reach the end goal from specific starting positions. Although this means the task doesn't generalize well (e.g. we likely can't have the robot finish the maze from a random position), it the idea for our project actually feasible.


### Why We Chose This Idea

We chose this project idea for two primary reasons. First, we thought it served as an evolutionary step of the q-learning project as it required us to go more in-depth with reinforcement learning and work extensively with deep learning variants of reinforcement algorithms. This enabled us to learn more about actual implementation of deep-learning systems, along with working extensively with simulated enviroments. Secondly, we thought the task was pretty cool as once the robot was able to succesfully navigate the maze from a start-point in an optimal (or at least semi-optimal) fashion, we should be able to train the robot to do the same task on a race-track. As a result, the task would generalize well to other enviroments so long as the robot was then trained within those enviroments first.


### Project Components

#### Environment

In order to train our model, we needed to create a simulation enviroment. The actual enviroment will be described further in the System Architecture section, but the core idea was that we needed an enviroment that enabled us to pause and unpause the world to allow gradient updates to take place and for the model to take the input and determine the next action that should be taken.

#### DDPG Algorithm Implementation

We use Deep Deterministic Policy Gradients as our core algorithm to determine the linear and angular velocities at a given point in time. We chose this algorithm as it allows for a continous state-space to be used. 

#### Training Loop

The training loop is used in tandem with the model implementation and the enviroment in order to train our model to complete the task from various start points within the maze.

### What We Were Able to Make Our Robot Do


Sadly, we had limited success in actually getting our implementation working correctly. After many, many different attempts at training the model and modifying the model, we succeeded at having the robot go relatively close to the goal (but not reach the goal) after making a singular turn and in having our robot reward hack its way into spinning in circles and never crashing into a wall.

#### GIFS 

##### GIF of Robot Almost Reaching Goal

![](https://github.com/Intro-Robotics-UChicago-Fall-2023/final-project-owenkarpf-rohanvoddhi/blob/main/almost_480p.gif)


##### GIF of Robot Spinning in Circles

![](https://github.com/Intro-Robotics-UChicago-Fall-2023/final-project-owenkarpf-rohanvoddhi/blob/main/spinning_480p.gif)



# System Architecture

### Gazebo Environment


##### General Overview of The Environment

We used a Gazebo environment in order to generate a space in which we could run training simulations. The environment lives inside the models.py file in the Enviroment class. The general idea behind the environment was that we wanted to be able to update the robot's linear and angular velocities within Gazebo following an action being generated by our DDPG model, keep track of the state of the model (get LIDAR readings, determine if wall was hit or if we reached the goal object). To do so, we had the environment run as a ROS node with publishers for the velocity and subscribers and associated callback functions for the lidar, camera and then also odometer (even though the odometer wasn't used in our final version of the DDPG model). Additionally, we determined the reward for an action within the environment, reset the world after the conclusion of a training iteration, and actually launched the Gazebo world.


##### Resetting the World and Determining if a Wall was Reached or if Goal was Reached

To reset the world, we used a Gazebo service proxy for resetting Gazebo worlds, which returned the world to its initial state that was used in the launch file. Additionally, we implemented capabilities to randomly spawn in a set of pre-determined different locations upon world reset, but chose to have the robot spawn only in one location to narrow the scope of our project to a feasible one. All of this was done in the reset_world function of the Environment class.

Going hand-in-hand with resetting world were the functions to determine if a wall or the goal was reached. For determining if a wall was reached, we used the is_at_wall method within the Environment class which checked to see if any LIDAR ranges were less than a threshold of .17 meters, and if they were, used that to say the robot was at the wall. For determining if the goal was reached, we used a combination of checking for if an object was directly in front of the robot and using the camera output to see if the colored wall the robot was trying to detect was directly in front of it. This was all within the is_at_target method within the Environment class.


##### Performing an Action

To perform the action, we began by publishing the linear and angular velocity of the action determined by our model. With that done, we then unpaused the physics of the Gazebo world, enabling the action to actually be carried out, slept for a very small amount of time, and then paused the world physics again. State was then recorded (both the LIDAR state, whether the given trial was completed and the reward for performing the given action). All of this was within our perform_action function in the Environment class.

##### Determining Reward

The method for determining the reward can be found in the determine_reward function of the Environment class. The general idea of the reward is that we wanted to incentivize high linear speeds, penalize turning, penalize hitting a wall, reward seeing the end goal and reward reaching the end goal. Additionally, we wanted the reward for reaching the goal to be remarkably high so that the robot would be able to realize that it did indeed want to reach the end goal. We penalized turning as we wanted the robot to take as direct a path as possible.


##### Launching the Gazebo World

Launching the Gazebo World was done in the initialization of the Environment Class. We used the subprocess module to run the command line arguments to launch a modified version of the maze (modified with the addition of a blue end goal) as seen below.

<img src="screenshot.png" height="300" width="800">


### DDPG Algorithm and Training Loop


##### DDPG Algorithm Implementation

**Rohan pls fill in below sections ty**

###### Actor Network


###### Critic Network


###### Replay Buffer


###### Navigation Network


##### Training Implementation

**RORO WRITES**

###### Scaling and Adding Noise

###### Training Loop



### Running in the Real World

Our implementation for running in the real world was relatively simple. All code for this resides in the RunModelWorld. It begins by setting up the velocity publisher and LIDAR and camera subscribers. Additionally, a trained actor network is loaded in with the correct weights.  After initialization, the run_model function is called, which runs the model until competition (either wall is hit or goal is reached). Within this function, at each iteration, the state is recorded and then passed to the actor network, which determines the action to take. This action is then scaled before being published. The action is then performed while the script sleeps for a small amount of time, before we check for if a wall or the goal is reached. If so, we terminate, otherwise keep running this perform action loop.


# ROS Node Diagram
<img src="rosgraph" height="300" width="800">


# Execution

First, "owenkarpf" or "rvoddhi" will have to be replaced in file paths to your username within the world file "modified_world.world". From there, you can just run the runner shell script within the main directory, which will train the model. If you want to train the model a second time, you will have to run the killer shell script as otherwise there are weird gazebo errors that arise. For testing within gazebo, run "./test.sh {ITER_NUM}" with ITER_NUM corresponding to the iteration at which the model was saved at. For example, for the model within the repo, run "./test.sh 395". Finally, for running in the real world with a trained model, run roscore, connect to the robot (run both bringup and bringup_cam) and then run. Next, change the path to the actor model within the real_world.py file to the correct one. Finally, run the command "rosrun final_project real_world.py". 


# Challenges

### Gazebo Challenges


##### Setting up the Gazebo World

Initially, we wanted the robot to receive a reward for following a general path by denoting the path as a certain color. Unfortunately, Gazebo provides a very poor world-editing experience, meaning that we couldn't directly apply the color to the floor. Following that, we tried adding an additional layer on-top of the default floor and then coloring that, but physics interactions between objects led to bumps in the floor that the robot either couldn't drive over or would drive over in a weird way that led to it turning in a way that it wasn't supposed to. As a result, we had to abandon the colored floor idea and instead decided to simply add a colored wall to denote the reward.

##### Gazebo World Glitches 

Gazebo had a glitch within it that led to weird simulation behavior. If the robot ran into a wall at a high enough speed, it would see what was on the other side of the wall through its camera. This resulted in our robot sometimes being mis-rewarded with having reached the goal or seeing the goal when it did not in fact achieve such a thing. Below is a GIF of such a thing occurring.


### Training Challenges


##### Narrow Maze

In an ideal world, we'd want our robot to easily explore the entire maze with just a small bit of noise being added in for the sake of exploration. Unfortunately, we made the mistake of working with the maze that had been created for the particle filter. Using this maze proved problematic as the areas in which the robot would drive through were thin, leading to the robot frequently crashing with a wall, ending a training simulation and preventing the robot from exploring further. This hampered our robot's exploration, limiting the Q-values it was able to achieve and the different states it was able to experience.

##### Hyper-Parameter Tuning

In an ideal world, we'd be able to perform a grid search to optimize the extensive list of hyper-parameters we are working with (layer sizes, amount of the actual networks we want to copy to the target networks, learning rates, etc) by running variants of the training in parallel across many different computing clusters. In practice, this wasn't feasible, so hyper-parameter tuning had to be done by hand. This was both inefficient and also had the added problem of being nowhere near as precise as running many different trainings at the same time with the only difference being the hyper-parameters.


##### Training Time

Unsurprisingly, training time for our model was quite high. Not only did each iteration take a decent amount of time due to it needing to be physically run in Gazebo, but training to convergence took many thousands of iterations. This not only led to a training time bottleneck, but also slowed down our iteration process, as we couldn't determine if a given set of hyper-parameters or a given change was working as intended after only a short number of trials.

 
##### Reward Hacking

The final challenge we faced was the issue of reward hacking. Throughout our time attempting to train the model to successfully navigate the maze in as fast a time as possible, we ran into continuous issues with reward hacking. For example, the robot spinning in circles below is an example of reward hacking, where the robot figured out that by spinning in circles ad infinitum, it would receive a high reward as the linear reward and reward for seeing the end goal outweighed the penalty for turning. Another example was when the robot's camera glitched through the wall in the Gazebo world by running into the wall really fast and then subsequently achieved a high reward for seeing the goal, the robot attempted to do this continuously.


# Future Work

In terms of future work, we'd like to go back and actually get the model implementation working correctly. Ideally, we'd begin by attempting to use another robotics simulator that is more geared towards running a large number of trials in an efficient way. Additionally, we'd like to sample a large number of different reward functions to see if any of those achieve better results. Finally, we'd like to create an environment with less narrow passage-ways so that the robot would be able to explore more of the environment before hitting a wall and having to reset. 


# Takeaways

1. Deep Learning is often very inefficient and time consuming without extensive compute resources.

2. Model Optimization is very difficult to do in practice, and can be a real performance bottleneck even when your general architectural structure is correct.

3. Pair programming isn't an efficient use of time when working on training a model, as it's much better for each to try different set-ups and different model variations with a bunch of small changes implemented. 

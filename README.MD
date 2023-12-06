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


Sadly, we had limited success in actually getting our implementation working correctly. <bold></bold>
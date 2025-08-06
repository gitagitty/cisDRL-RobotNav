#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion
import math
import time
import threading

class RobotNavigationEnv(gym.Env, Node):
    """
    Custom Gymnasium Environment for Robot Navigation to Target Point (10,10)
    
    Observation Space: [robot_x, robot_y, robot_yaw, distance_to_target, angle_to_target, 
                       lidar_data (10 rays), linear_vel, angular_vel]
    Action Space: [linear_velocity, angular_velocity] (continuous)
    """
    
    def __init__(self):
        # Initialize ROS2 node first
        Node.__init__(self, 'robot_navigation_env')
        gym.Env.__init__(self)
        
        # Target position
        self.target_x = 10.0
        self.target_y = 10.0
        self.target_tolerance = 0.5  # meters
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.lidar_data = np.full(10, 10.0)  # 10 lidar rays, max range 10m
        
        # Sensor data for checking availability
        self.current_pose = None
        self.current_lidar = None
        self.current_velocity = None
        
        # Episode management
        self.max_episode_steps = 1000
        self.current_step = 0
        self.episode_start_time = time.time()
        self.max_episode_time = 120.0  # 2 minutes
        
        # Rewards
        self.previous_distance = None
        self.min_distance_achieved = float('inf')
        
        # Define action and observation spaces
        # Actions: [linear_velocity (-1 to 1 m/s), angular_velocity (-2 to 2 rad/s)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -2.0], dtype=np.float32),
            high=np.array([1.0, 2.0], dtype=np.float32),
            shape=(2,)
        )
        
        # Observations: [x, y, yaw, dist_to_target, angle_to_target, 10 lidar rays, lin_vel, ang_vel]
        self.observation_space = gym.spaces.Box(
            low=np.array([-20.0, -20.0, -np.pi, 0.0, -np.pi] + [0.0]*10 + [-2.0, -2.0], dtype=np.float32),
            high=np.array([20.0, 20.0, np.pi, 30.0, np.pi] + [10.0]*10 + [2.0, 2.0], dtype=np.float32),
            shape=(17,)
        )
        
        # ROS2 Publishers and Subscribers
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        
        # Service client for resetting simulation
        self.reset_client = self.create_client(Empty, '/reset_world')
        
        # Remove threading - use direct ROS integration like your circle trainer
        # self.ros_thread = threading.Thread(target=self._spin_ros, daemon=True)
        # self.ros_thread.start()
        
        self.get_logger().info("Robot Navigation Environment initialized!")
    
    def _spin_ros(self):
        """Spin ROS in a separate thread"""
        rclpy.spin(self)
    
    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Get velocities
        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z
        
        # Mark that pose data is available
        self.current_pose = msg
    
    def lidar_callback(self, msg):
        """Process LIDAR data"""
        # Downsample to 10 rays from the original scan
        ranges = np.array(msg.ranges)
        ranges[ranges == 0] = msg.range_max  # Replace 0s with max range
        ranges[np.isinf(ranges)] = msg.range_max  # Replace inf with max range
        ranges[np.isnan(ranges)] = msg.range_max  # Replace nan with max range
        
        # Downsample to 10 evenly spaced rays
        step = len(ranges) // 10
        self.lidar_data = ranges[::step][:10]
        
        # Clip to reasonable range
        self.lidar_data = np.clip(self.lidar_data, 0.1, 10.0)
        
        # Mark that lidar data is available
        self.current_lidar = msg
    
    def get_observation(self):
        """Get current observation"""
        # Calculate distance and angle to target
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx) - self.robot_yaw
        
        # Normalize angle to [-pi, pi]
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        
        obs = np.array([
            self.robot_x,
            self.robot_y, 
            self.robot_yaw,
            distance_to_target,
            angle_to_target,
            *self.lidar_data,
            self.linear_vel,
            self.angular_vel
        ], dtype=np.float32)
        
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_start_time = time.time()
        self.previous_distance = None
        self.min_distance_achieved = float('inf')
        
        # Reset the simulation world (this will reset robot to default position)
        self._reset_robot_position(0, 0, 0)  # Parameters ignored with reset_world
        
        # Wait for robot to settle
        time.sleep(2.0)
        
        # Send zero velocity
        self._send_action([0.0, 0.0])
        
        # Wait for sensor data to be available
        timeout = 10.0
        start_time = time.time()
        while (self.current_pose is None or self.current_lidar is None) and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.current_pose is None or self.current_lidar is None:
            self.get_logger().warn("Timeout waiting for sensor data during reset")
        
        observation = self.get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Send action to robot
        self._send_action(action)
        
        # Wait for action to take effect
        time.sleep(0.1)
        
        # Get observation
        observation = self.get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation, action)
        
        # Check if episode is done
        terminated, truncated, info = self._check_episode_end(observation)
        
        return observation, reward, terminated, truncated, info
    
    def _send_action(self, action):
        """Send velocity commands to robot"""
        twist = Twist()
        # Use similar action scaling as your circle trainer
        twist.linear.x = float(np.clip(action[0], 0.0, 1.0)) * 0.5   # Forward only, max 0.5 m/s
        twist.angular.z = float(np.clip(action[1], -1.0, 1.0)) * 1.0  # Angular: -1.0 to 1.0 rad/s
        self.cmd_vel_pub.publish(twist)
    
    def _calculate_reward(self, observation, action):
        """Calculate reward based on current state"""
        distance_to_target = observation[3]
        angle_to_target = abs(observation[4])
        min_lidar_distance = min(observation[5:15])
        
        reward = 0.0
        
        # Distance-based reward (progress toward target)
        if self.previous_distance is not None:
            distance_progress = self.previous_distance - distance_to_target
            reward += distance_progress * 5.0  # Scale reward for progress
        
        self.previous_distance = distance_to_target
        
        # Keep track of minimum distance achieved
        if distance_to_target < self.min_distance_achieved:
            self.min_distance_achieved = distance_to_target
            reward += 2.0  # Bonus for getting closer than ever before
        
        # Large reward for reaching target
        if distance_to_target < self.target_tolerance:
            reward += 100.0
            self.get_logger().info(f"🎯 TARGET REACHED! Distance: {distance_to_target:.2f}m")
        
        # Penalty for being far from target
        reward -= distance_to_target * 0.1
        
        # Reward for facing target
        reward -= angle_to_target * 0.5
        
        # Collision penalty
        if min_lidar_distance < 0.3:
            reward -= 50.0
            self.get_logger().warn(f"⚠️ Collision risk! Min distance: {min_lidar_distance:.2f}m")
        
        # Small penalty for high angular velocity (encourage smooth motion)
        reward -= abs(action[1]) * 0.1
        
        # Time penalty (encourage efficiency)
        reward -= 0.01
        
        return reward
    
    def _check_episode_end(self, observation):
        """Check if episode should end"""
        distance_to_target = observation[3]
        min_lidar_distance = min(observation[5:15])
        elapsed_time = time.time() - self.episode_start_time
        
        terminated = False
        truncated = False
        info = {}
        
        # Success condition
        if distance_to_target < self.target_tolerance:
            terminated = True
            info['success'] = True
            info['reason'] = 'target_reached'
        
        # Collision condition
        elif min_lidar_distance < 0.2:
            terminated = True
            info['success'] = False
            info['reason'] = 'collision'
        
        # Timeout conditions
        elif self.current_step >= self.max_episode_steps:
            truncated = True
            info['success'] = False
            info['reason'] = 'max_steps'
        
        elif elapsed_time > self.max_episode_time:
            truncated = True
            info['success'] = False
            info['reason'] = 'timeout'
        
        # Add final stats
        if terminated or truncated:
            info['final_distance'] = distance_to_target
            info['min_distance_achieved'] = self.min_distance_achieved
            info['steps'] = self.current_step
            info['time'] = elapsed_time
        
        return terminated, truncated, info
    
    def _reset_robot_position(self, x, y, yaw):
        """Reset robot position using reset_world service"""
        try:
            # Use reset_world service to reset the entire simulation
            request = Empty.Request()
            future = self.reset_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            
            # Wait for reset to complete
            time.sleep(0.5)
            
            self.get_logger().info("World reset successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to reset world: {e}")
    
    def close(self):
        """Clean up resources"""
        # Stop the robot
        self._send_action([0.0, 0.0])
        
        # Destroy ROS node
        self.destroy_node()
        
        self.get_logger().info("Environment closed")

def make_env():
    """Factory function to create environment"""
    return RobotNavigationEnv()

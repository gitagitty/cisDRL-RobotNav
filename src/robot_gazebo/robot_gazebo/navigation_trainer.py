#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import time
import os

class SimpleNavigationNet(nn.Module):
    def __init__(self, input_dim=5, action_dim=2, hidden_dim=64):
        super(SimpleNavigationNet, self).__init__()
        
        # Simple network for navigation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) - outputs actions
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        shared = self.shared(x)
        action = self.actor(shared)
        value = self.critic(shared)
        return action, value

class NavigationTrainer(Node):
    def __init__(self):
        super().__init__('navigation_trainer')
        
        # PPO parameters (simplified)
        self.lr = 3e-4
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.epochs = 5
        self.max_episode_steps = 1000
        self.action_std = 0.2
        
        # Navigation parameters
        self.target_x = 10.0
        self.target_y = 10.0
        self.target_tolerance = 0.5
        
        # Initialize network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = SimpleNavigationNet().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        # ROS2 setup - using correct Gazebo Classic topics 
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        
        self.lidar_sub = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10
        )
        
        # State variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_linear_vel = 0.0
        self.robot_angular_vel = 0.0
        self.min_lidar_distance = 10.0
        
        # Episode tracking
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.start_time = time.time()
        
        # Training data storage
        self.reset_buffers()
        
        # Start training loop
        self.training_timer = self.create_timer(0.1, self.training_step)
        
        self.get_logger().info("Navigation Trainer initialized - Learning to navigate to (10,10)!")
        
    def reset_buffers(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def odom_callback(self, msg):
        # Extract robot position and orientation  
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Convert quaternion to yaw
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Extract velocities
        self.robot_linear_vel = msg.twist.twist.linear.x
        self.robot_angular_vel = msg.twist.twist.angular.z
        
    def lidar_callback(self, msg):
        """Process LIDAR data - get minimum distance"""
        ranges = np.array(msg.ranges)
        ranges[ranges == 0] = msg.range_max
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        
        self.min_lidar_distance = float(np.min(ranges))
        
    def get_observation(self):
        """
        Simple observation: [distance_to_target, angle_to_target, min_lidar, linear_vel, angular_vel]
        """
        # Distance to target
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # Angle to target
        angle_to_target = math.atan2(dy, dx) - self.robot_yaw
        # Normalize angle to [-pi, pi]
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        
        observation = np.array([
            distance_to_target / 20.0,      # Normalized distance (0-1 for 20m)
            angle_to_target / math.pi,      # Normalized angle [-1, 1]
            self.min_lidar_distance / 10.0, # Normalized lidar (0-1 for 10m)
            self.robot_linear_vel / 1.0,    # Normalized linear velocity
            self.robot_angular_vel / 2.0    # Normalized angular velocity
        ], dtype=np.float32)
        
        return observation
        
    def calculate_navigation_reward(self, action):
        """
        Improved navigation reward function
        """
        reward = 0.0
        
        # Distance to target
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # 1. Progress reward - PRIMARY INCENTIVE (encourage getting closer)
        if not hasattr(self, 'previous_distance') or self.previous_distance is None:
            self.previous_distance = distance_to_target
            progress = 0.0
        else:
            progress = self.previous_distance - distance_to_target
        
        # Strong reward for progress toward target
        reward += progress * 50.0  # Increased from 10.0 to 50.0
        self.previous_distance = distance_to_target
        
        # 2. SUCCESS reward - huge bonus for reaching target
        if distance_to_target < self.target_tolerance:
            reward += 1000.0  # Increased from 100.0
            self.get_logger().info(f"🎯 TARGET REACHED! Distance: {distance_to_target:.2f}m")
        
        # 3. Distance-based reward (positive for being close)
        max_distance = 20.0  # Maximum expected distance
        distance_reward = (max_distance - distance_to_target) / max_distance
        reward += distance_reward * 5.0
        
        # 4. Collision penalty (keep this strong)
        if self.min_lidar_distance < 0.3:
            reward -= 100.0  # Increased penalty
            
        # 5. Orientation reward - encourage facing target
        angle_to_target = math.atan2(dy, dx) - self.robot_yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))
        angle_reward = 1.0 - abs(angle_to_target) / math.pi
        reward += angle_reward * 3.0  # Increased from 2.0
        
        # 6. Forward motion reward (encourage movement toward target)
        if self.robot_linear_vel > 0 and abs(angle_to_target) < math.pi/2:
            reward += self.robot_linear_vel * 2.0  # Reward forward motion when facing target
        
        # 7. REMOVED: Action penalty (was discouraging movement)
        # 8. REMOVED: Time penalty (was creating constant negative pressure)
        
        return reward
        
    def select_action(self, observation):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean, value = self.network(obs_tensor)
        
        # Add noise for exploration
        action_std_tensor = torch.ones_like(action_mean) * self.action_std
        dist = torch.distributions.Normal(action_mean, action_std_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Scale actions for navigation (similar to your circle trainer)
        action_scaled = action.clone()
        action_scaled[0, 0] = torch.clamp(action_scaled[0, 0], 0.0, 1.0) * 0.5   # Linear: 0 to 0.5 m/s
        action_scaled[0, 1] = torch.clamp(action_scaled[0, 1], -1.0, 1.0) * 1.0  # Angular: -1.0 to 1.0 rad/s
        
        return action_scaled.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
        
    def training_step(self):
        # Get current observation
        observation = self.get_observation()
        
        # Select action
        action, log_prob, value = self.select_action(observation)
        
        # Execute action - send to cmd_vel topic
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)
        
        # Calculate reward
        reward = self.calculate_navigation_reward(action)
        self.episode_reward += reward
        self.episode_step += 1
        
        # Store transition
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        
        # Check if episode is done
        dx = self.target_x - self.robot_x
        dy = self.target_y - self.robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        done = (
            self.episode_step >= self.max_episode_steps or 
            distance_to_target < self.target_tolerance or
            self.min_lidar_distance < 0.2 or
            distance_to_target > 25.0  # Add condition to prevent robot from going too far
        )
        self.dones.append(done)
        
        if done or self.episode_step % 100 == 0:  # Log progress
            self.get_logger().info(
                f"Step {self.episode_step}: Pos=({self.robot_x:.2f}, {self.robot_y:.2f}), "
                f"Target=({self.target_x}, {self.target_y}), Dist={distance_to_target:.2f}m, "
                f"Reward={reward:.2f}, Total={self.episode_reward:.2f}"
            )
        
        if done:
            self.episode_count += 1
            avg_reward = self.episode_reward / self.episode_step
            
            success = distance_to_target < self.target_tolerance
            self.get_logger().info(
                f"Episode {self.episode_count} completed: "
                f"{'SUCCESS' if success else 'FAILED'}, "
                f"Steps={self.episode_step}, Final Dist={distance_to_target:.2f}m, "
                f"Total Reward={self.episode_reward:.2f}, Avg Reward={avg_reward:.4f}"
            )
            
            # Train if we have enough data
            if len(self.observations) >= 64:
                self.train_ppo()
                self.reset_buffers()
            
            # Reset episode
            self.episode_step = 0
            self.episode_reward = 0
            self.previous_distance = None
            
            # Save model periodically
            if self.episode_count % 10 == 0:
                self.save_model(f"navigation_model_episode_{self.episode_count}.pth")
                
    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
        
    def train_ppo(self):
        """Simplified PPO training"""
        if len(self.observations) < 32:
            return
            
        # Convert to tensors
        observations = torch.FloatTensor(np.array(self.observations)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values = self.values
        
        # Compute advantages
        advantages, returns = self.compute_gae(self.rewards, values, self.dones)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training
        for epoch in range(self.epochs):
            action_means, current_values = self.network(observations)
            
            # New log probabilities
            action_std = torch.ones_like(action_means) * self.action_std
            dist = torch.distributions.Normal(action_means, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            # Fix tensor shape mismatch by ensuring consistent dimensions
            value_loss = nn.MSELoss()(current_values.squeeze(), returns.squeeze())
            total_loss = policy_loss + 0.5 * value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.get_logger().info(f"Training completed. Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
        
    def save_model(self, filename):
        """Save the trained model"""
        model_dir = "/home/evan/hiwonder/src/robot_gazebo/models"
        os.makedirs(model_dir, exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
        }, os.path.join(model_dir, filename))
        
        self.get_logger().info(f"Navigation model saved: {filename}")

def main(args=None):
    rclpy.init(args=args)
    trainer = NavigationTrainer()
    
    try:
        rclpy.spin(trainer)
    except KeyboardInterrupt:
        trainer.save_model("navigation_model_final.pth")
        trainer.get_logger().info("Navigation training stopped and model saved")
    finally:
        trainer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import time
import rclpy
import torch
from ros_nodes import (
    ScanSubscriber,
    OdomSubscriber,
    ResetWorldClient,
    SetModelStateClient,
    CmdVelPublisher,
    MarkerPublisher,
    PhysicsClient,
    SensorSubscriber,
)
import numpy as np
import torch.nn.functional as F
from geometry_msgs.msg import Pose, Twist
from squaternion import Quaternion
import rospkg
import os
import json
from torch.utils.tensorboard import SummaryWriter
from SAC.SAC_actor import DiagGaussianActor as actor_model

class DEP_env:
    def __init__(
        self,
        device,
        max_action,
        model_path,
        state_dim=45,
        action_dim=2,
        init_target_distance=2.0,
        target_dist_increase=0.001,
        max_target_dist=8.0,
        target_reached_delta=0.2,
        collision_delta=0.1,
        args=None,
    ):
        rclpy.init(args=args)
        super().__init__()
        
        
        self.cmd_vel_publisher = CmdVelPublisher()
        self.scan_subscriber = ScanSubscriber()
        self.odom_subscriber = OdomSubscriber()
        self.robot_state_publisher = SetModelStateClient()
        self.world_reset = ResetWorldClient()
        self.physics_client = PhysicsClient()
        self.publish_target = MarkerPublisher()
        self.sensor_subscriber = SensorSubscriber()
        self.target_dist = init_target_distance
        self.target_dist_increase = target_dist_increase
        self.max_target_dist = max_target_dist
        self.target_reached_delta = target_reached_delta
        self.collision_delta = collision_delta
        # self.target = self.set_target_position(0.0, 0.0)
        self.element_positions = []
        self.model_path = model_path
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = (-max_action, max_action)
        self.device = torch.device(device)
        self.writer = SummaryWriter('deploy/{}'.format(time.strftime("%m%d-%H%M")))
        self.actor = actor_model(
            obs_dim=self.state_dim,
            action_dim=action_dim,
            hidden_dim=1024,
            hidden_depth=2,
            log_std_bounds=[-5, 2],
        ).to(self.device)
        
        self.load_model()
        # Load configurations
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('robot_gazebo')
        config_path = os.path.join('src', 'robot_gazebo', 'configs', 'configs.json')
        
        with open(config_path, 'r') as f:
            self.configs = json.load(f)
        
        self.current_config_index = 0
        
    def load_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'actor_state_dict' in state_dict:
                state_dict = state_dict['actor_state_dict']
            # 也可能是直接的状态字典
            elif all(key.startswith('trunk.') or key.startswith('log_std_bounds') for key in state_dict.keys()):
                pass  # 已经是正确的状态字典
        
        # 加载到模型
        self.actor.load_state_dict(state_dict)
        self.actor.to(self.device)
        self.actor.eval()  # 设置为评估模式
        print(f'Model loaded from {self.model_path}')


    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    
        
        # 部署时使用确定性策略
        with torch.no_grad():
            action = self.actor(state).mean.cpu().numpy().flatten()
        
        return action.clip(self.action_range[0], self.action_range[1])
    
    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        self.cmd_vel_publisher.publish_cmd_vel(lin_velocity, ang_velocity)
        self.physics_client.unpause_physics()
        time.sleep(0.1)
        rclpy.spin_once(self.sensor_subscriber)
        self.physics_client.pause_physics()

        (
            latest_scan,
            latest_position,
            latest_orientation,
        ) = self.sensor_subscriber.get_latest_sensor()

        distance, cos, sin, _ = self.get_dist_sincos(
            latest_position, latest_orientation
        )
        collision = self.sensor_subscriber.has_collision()
        # collision_count = self.sensor_subscriber.get_collision_count()
        if collision:
            print("Collision detected!")
            self.sensor_subscriber.collision_count += 1
            print(f"Collision count: {self.sensor_subscriber.collision_count}")
        crash = self.sensor_subscriber.has_crash()
        if crash:
            print("Crash detected!")
        goal = self.check_target(distance, collision)
        if goal:
            print("Target reached!")
        action = [lin_velocity, ang_velocity]
        reward = self.get_reward(goal, collision, action, latest_scan, crash, cos)
        collision_count = self.sensor_subscriber.collision_count

        return latest_scan, distance, cos, sin, collision, goal, action, reward, collision_count,crash

    def reset(self):
        self.world_reset.reset_world()
        self.sensor_subscriber.reset_collision_count()
        action = [0.0, 0.0]
        self.cmd_vel_publisher.publish_cmd_vel(
            linear_velocity=action[0], angular_velocity=action[1]
        )

        self.element_positions = []
        self.set_positions()

        self.publish_target.publish(self.target[0], self.target[1])

        latest_scan, distance, cos, sin, _, _, action, reward, collision_count, crash = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    def set_target_position(self, x, y):
        self.element_positions.append([x, y])
        return [x, y]
    
    def set_random_position(self, name):
        angle = np.random.uniform(-np.pi, np.pi)
        pos = False
        while not pos:
            x = np.random.uniform(-4.0, 4.0)
            y = np.random.uniform(-4.0, 4.0)
            pos = self.check_position(x, y, 1.8)
        self.element_positions.append([x, y])
        self.set_position(name, x, y, angle)

    def set_robot_position(self, x, y, angle):
        self.set_position("robot", x, y, angle)
        return x, y
    
   
    def set_position(self, name, x, y, angle):
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.x = quaternion.x
        pose.orientation.y = quaternion.y
        pose.orientation.z = quaternion.z
        pose.orientation.w = quaternion.w

        self.robot_state_publisher.set_state(name, pose)
        rclpy.spin_once(self.robot_state_publisher)

    def set_positions(self):
        # for i in range(4, 8):
        #     name = "obstacle" + str(i + 1)
        #     self.set_random_position(name)
        config = self.configs[self.current_config_index]
        
        # Set robot position
        start_pose = config['start_pose']
        self.set_robot_position(start_pose[0], start_pose[1], start_pose[2])
        
        # Set target position
        target_pos = config['target_position']
        self.set_target_position(target_pos[0], target_pos[1])
        
        # Update internal state
        self.target = target_pos
        if self.element_positions:
            self.element_positions[-1] = target_pos
        else:
            self.element_positions.append(target_pos)
        
        # print(f"Target position: {target_pos[0]}, {target_pos[1]}")
        # print(f"Robot position: {start_pose[0]}, {start_pose[1]}, angle: {start_pose[2]}")
        
        # Move to next configuration for next episode
        self.current_config_index = (self.current_config_index + 1) % len(self.configs)
        
        # robot_position = self.set_robot_position()
        # self.target = self.set_target_position(robot_position)

    def check_position(self, x, y, min_dist):
        pos = True
        for element in self.element_positions:
            distance_vector = [element[0] - x, element[1] - y]
            distance = np.linalg.norm(distance_vector)
            if distance < min_dist:
                pos = False
        return pos

    """ def check_collision(self, laser_scan):
        # if min(laser_scan) < self.collision_delta:
        #     return True
        # return False
        return self.sensor_subscriber.has_collision() """

    def check_target(self, distance, collision):
        if distance < self.target_reached_delta and not collision:
            self.target_dist += self.target_dist_increase
            if self.target_dist > self.max_target_dist:
                self.target_dist = self.max_target_dist
            return True
        return False

    def get_dist_sincos(self, odom_position, odom_orientation):
        # Calculate robot heading from odometry data
        odom_x = odom_position.x
        odom_y = odom_position.y
        quaternion = Quaternion(
            odom_orientation.w,
            odom_orientation.x,
            odom_orientation.y,
            odom_orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)
        pose_vector = [np.cos(angle), np.sin(angle)]
        goal_vector = [self.target[0] - odom_x, self.target[1] - odom_y]

        distance = np.linalg.norm(goal_vector)
        cos, sin = self.cossin(pose_vector, goal_vector)

        return distance, cos, sin, angle

    @staticmethod
    def get_reward(goal, collision, action, laser_scan, crash, cos):
        crash_reward = 0.0
        col_reward = 0.0
        goal_reward = 0.0
        if crash:
            crash_reward = -500.0  # Severe penalty for crashes
        if collision:
            col_reward = -100.0  # Penalize collisions
        if goal:
            goal_reward = 500.0  # Large reward for reaching goal
        
        # Handle None or empty laser scans
        if laser_scan is None or len(laser_scan) == 0:
            return 0.0  # Neutral reward when no scan data
            
        target_reward = action[0] * cos
        base_reward = abs(action[0])+abs(action[1]) # Base reward for moving forward
            
        return base_reward + goal_reward + col_reward + crash_reward + target_reward



    @staticmethod
    def cossin(vec1, vec2):
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        cos = np.dot(vec1, vec2)
        sin = np.cross(vec1, vec2).item()

        return cos, sin
    
    def prepare_state(self, latest_scan, distance, cos, sin, goal, action, collision_count, crash):
        latest_scan = np.array(latest_scan)
        
        # 1. Preprocess laser scan
        inf_mask = np.isinf(latest_scan) | (latest_scan > 4.0)
        latest_scan[inf_mask] = 4.0
        
        
        max_bins = (self.state_dim - 5) / 2
        bin_size = int(np.ceil(len(latest_scan) / max_bins))

        # Initialize the list to store the minimum values of each bin
        min_values = []
        max_values = []

        # Loop through the data and create bins
        for i in range(0, len(latest_scan), bin_size):
            # Get the current bin
            bin = latest_scan[i : i + min(bin_size, len(latest_scan) - i)]
            # Find the minimum value in the current bin and append it to the min_values list
            min_values.append(min(bin))
            max_values.append(max(bin))
        state = min_values + max_values + [distance, cos, sin] + [action[0], action[1]]
        
        
        # 6. Terminal conditions (goal, collisions, or crash)
        terminal = 1 if goal or (collision_count >= 15) or crash else 0
        
        # 7. Verify state dimension matches expected size
        assert len(state) == self.state_dim
        
        return state, terminal

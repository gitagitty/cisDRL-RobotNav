#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import rclpy
from stable_baselines3 import PPO

# Import our custom environment
sys.path.append('/home/ubuntu/ros2_ws/src/robot_gazebo/robot_gazebo')
from robot_nav_env import RobotNavigationEnv

def run_trained_model(model_path=None):
    """Run the trained PPO model in real-time"""
    
    print("🤖 Running Trained PPO Model")
    print("=" * 50)
    print("Target: Navigate to point (10, 10)")
    print("Controls: Ctrl+C to stop")
    print("=" * 50)
    
    # Initialize ROS 2
    rclpy.init()
    
    try:
        # Find the latest model if none specified
        if model_path is None:
            models_dir = "/home/ubuntu/ros2_ws/src/robot_gazebo/models"
            if os.path.exists(models_dir):
                # Find the latest model directory
                model_dirs = [d for d in os.listdir(models_dir) if d.startswith("ppo_robot_navigation_")]
                if model_dirs:
                    latest_model_dir = sorted(model_dirs)[-1]
                    model_path = f"{models_dir}/{latest_model_dir}/best_model/best_model"
                    print(f"📂 Using latest model: {model_path}")
                else:
                    print("❌ No trained models found!")
                    return
            else:
                print("❌ Models directory not found!")
                return
        
        # Load the trained model
        if not os.path.exists(f"{model_path}.zip"):
            print(f"❌ Model file not found: {model_path}.zip")
            return
        
        print(f"📥 Loading model from: {model_path}")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully!")
        
        # Create environment
        print("🌍 Creating environment...")
        env = RobotNavigationEnv()
        
        # Reset environment
        print("🔄 Resetting environment...")
        obs, _ = env.reset()
        
        print("\\n🚀 Starting navigation...")
        print("📍 Target position: (10.0, 10.0)")
        print("=" * 50)
        
        episode_reward = 0
        step_count = 0
        start_time = time.time()
        
        while True:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Get current state
            robot_x = obs[0]
            robot_y = obs[1] 
            robot_yaw = obs[2]
            distance_to_target = obs[3]
            angle_to_target = obs[4]
            min_lidar_distance = min(obs[5:15])
            
            # Print status every 20 steps
            if step_count % 20 == 0:
                elapsed_time = time.time() - start_time
                print(f"Step {step_count:4d} | "
                      f"Pos: ({robot_x:5.2f}, {robot_y:5.2f}) | "
                      f"Target dist: {distance_to_target:5.2f}m | "
                      f"Obstacle: {min_lidar_distance:4.2f}m | "
                      f"Reward: {episode_reward:6.1f} | "
                      f"Time: {elapsed_time:5.1f}s")
            
            # Check if episode ended
            if terminated or truncated:
                elapsed_time = time.time() - start_time
                
                print("\\n" + "=" * 50)
                if 'success' in info and info['success']:
                    print("🎉 SUCCESS! Robot reached the target!")
                    print(f"✅ Final distance to target: {distance_to_target:.2f}m")
                else:
                    reason = info.get('reason', 'unknown')
                    print(f"❌ Episode ended: {reason}")
                    print(f"📍 Final distance to target: {distance_to_target:.2f}m")
                
                print(f"📊 Episode Statistics:")
                print(f"   Total steps: {step_count}")
                print(f"   Total reward: {episode_reward:.1f}")
                print(f"   Episode time: {elapsed_time:.1f}s")
                print(f"   Min distance achieved: {info.get('min_distance_achieved', 'N/A'):.2f}m")
                print("=" * 50)
                
                # Ask if user wants to continue
                try:
                    continue_training = input("\\n🔄 Run another episode? (y/n): ").lower().strip()
                    if continue_training != 'y':
                        break
                    
                    # Reset for new episode
                    print("\\n🔄 Starting new episode...")
                    obs, _ = env.reset()
                    episode_reward = 0
                    step_count = 0
                    start_time = time.time()
                    
                except KeyboardInterrupt:
                    break
        
        print("\\n🛑 Stopping robot...")
        env.close()
        
    except KeyboardInterrupt:
        print("\\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trained PPO model for robot navigation")
    parser.add_argument("--model", type=str, help="Path to trained model (without .zip extension)", default=None)
    
    args = parser.parse_args()
    
    run_trained_model(args.model)

if __name__ == "__main__":
    main()

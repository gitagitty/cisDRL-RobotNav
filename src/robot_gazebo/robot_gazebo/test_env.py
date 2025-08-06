#!/usr/bin/env python3

import sys
import rclpy
import time

# Import our custom environment
sys.path.append('/home/ubuntu/ros2_ws/src/robot_gazebo/robot_gazebo')
from robot_nav_env import RobotNavigationEnv

def test_environment():
    """Test the environment creation and basic functionality"""
    
    print("🧪 Testing Robot Navigation Environment")
    print("=" * 50)
    
    # Initialize ROS 2
    rclpy.init()
    
    try:
        # Create environment
        print("🌍 Creating environment...")
        env = RobotNavigationEnv()
        print("✅ Environment created successfully!")
        
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test reset
        print("\\n🔄 Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Robot position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
        print(f"   Distance to target: {((env.target_x - env.robot_x)**2 + (env.target_y - env.robot_y)**2)**0.5:.2f}m")
        
        # Test a few actions
        print("\\n🎮 Testing actions...")
        for i in range(5):
            action = env.action_space.sample()  # Random action
            print(f"   Action {i}: linear={action[0]:.2f}, angular={action[1]:.2f}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Reward: {reward:.2f}, Position: ({env.robot_x:.2f}, {env.robot_y:.2f})")
            
            if terminated or truncated:
                print(f"   Episode ended: {info}")
                break
        
        # Clean up
        env.close()
        
        print("\\n" + "=" * 50)
        print("✅ Environment test completed successfully!")
        print("🚀 Ready for PPO training!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    test_environment()

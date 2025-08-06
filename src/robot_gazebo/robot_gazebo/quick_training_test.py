#!/usr/bin/env python3

import os
import sys
import time
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import our custom environment
sys.path.append('/home/ubuntu/ros2_ws/src/robot_gazebo/robot_gazebo')
from robot_nav_env import RobotNavigationEnv

def quick_training_test(total_timesteps=5000):
    """Quick training test to verify everything works"""
    
    print("🚀 Quick PPO Training Test")
    print("=" * 50)
    print(f"Target: Navigate to point (10, 10)")
    print(f"Training steps: {total_timesteps}")
    print(f"This is a quick test - for full training, use ppo_training.py")
    print("=" * 50)
    
    # Initialize ROS 2
    rclpy.init()
    
    try:
        # Create environment
        print("🌍 Creating environment...")
        env = RobotNavigationEnv()
        
        # Test environment
        print("🧪 Testing environment...")
        obs, info = env.reset()
        print(f"✅ Environment created successfully!")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        
        # Create model
        print("🤖 Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            tensorboard_log="/home/ubuntu/ros2_ws/src/robot_gazebo/logs/"
        )
        
        print("✅ Model created successfully!")
        
        # Create output directory
        quick_test_dir = "/home/ubuntu/ros2_ws/src/robot_gazebo/quick_test"
        os.makedirs(quick_test_dir, exist_ok=True)
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=quick_test_dir,
            name_prefix="ppo_quick_test"
        )
        
        # Start training
        print(f"🏋️ Starting training for {total_timesteps} steps...")
        start_time = time.time()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.1f} seconds!")
        
        # Save final model
        model_path = f"{quick_test_dir}/final_model"
        model.save(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Quick test of trained model
        print("\\n🧪 Testing trained model...")
        obs, _ = env.reset()
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                distance = obs[3]  # distance to target
                print(f"   Step {step}: Distance to target = {distance:.2f}m")
            
            if terminated or truncated:
                if 'success' in info and info['success']:
                    print(f"🎉 Success! Reached target in {step} steps!")
                else:
                    print(f"Episode ended after {step} steps")
                break
        
        # Clean up
        env.close()
        
        print("\\n" + "=" * 50)
        print("✅ Quick training test completed successfully!")
        print(f"📁 Files saved in: {quick_test_dir}")
        print("💡 For full training, run: python3 ppo_training.py")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print("🧹 Cleanup completed")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick PPO training test")
    parser.add_argument("--steps", type=int, default=5000, help="Number of training steps")
    
    args = parser.parse_args()
    
    quick_training_test(args.steps)

if __name__ == "__main__":
    main()

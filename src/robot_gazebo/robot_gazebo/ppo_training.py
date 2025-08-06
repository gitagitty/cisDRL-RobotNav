#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ROS 2 imports
import rclpy
from rclpy.executors import MultiThreadedExecutor

# RL imports
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Import our custom environment
sys.path.append('/home/ubuntu/ros2_ws/src/robot_gazebo/robot_gazebo')
from robot_nav_env import RobotNavigationEnv

class PPOTrainer:
    def __init__(self):
        self.model_name = f"ppo_robot_navigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = f"/home/ubuntu/ros2_ws/src/robot_gazebo/models/{self.model_name}"
        self.log_dir = f"/home/ubuntu/ros2_ws/src/robot_gazebo/logs/{self.model_name}"
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/eval", exist_ok=True)
        
        print(f"🚀 PPO Training Setup")
        print(f"📁 Model save directory: {self.save_dir}")
        print(f"📊 Log directory: {self.log_dir}")
        
        # Initialize ROS 2
        rclpy.init()
        
        # Training parameters
        self.total_timesteps = 100000  # Adjust based on your needs
        self.eval_freq = 5000
        self.save_freq = 10000
        
        # Training results
        self.training_rewards = []
        self.eval_rewards = []
        self.success_rates = []
        
    def create_env(self, env_id=0):
        """Create and wrap environment"""
        def _init():
            env = RobotNavigationEnv()
            env = Monitor(env, f"{self.log_dir}/train_env_{env_id}")
            return env
        return _init
    
    def create_eval_env(self):
        """Create evaluation environment"""
        env = RobotNavigationEnv()
        env = Monitor(env, f"{self.log_dir}/eval")
        return env
    
    def setup_callbacks(self, eval_env):
        """Setup training callbacks"""
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.save_dir}/best_model",
            log_path=f"{self.log_dir}/eval",
            eval_freq=self.eval_freq,
            deterministic=True,
            render=False,
            verbose=1,
            n_eval_episodes=5
        )
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=f"{self.save_dir}/checkpoints",
            name_prefix="ppo_robot_nav"
        )
        
        return CallbackList([eval_callback, checkpoint_callback])
    
    def create_model(self, env):
        """Create PPO model with optimized hyperparameters"""
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=self.log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            device='auto',
            seed=42
        )
        
        print("🧠 PPO Model created with hyperparameters:")
        print(f"   Learning rate: 3e-4")
        print(f"   Batch size: 64")
        print(f"   N steps: 2048")
        print(f"   Gamma: 0.99")
        print(f"   Device: {model.device}")
        
        return model
    
    def train(self):
        """Main training loop"""
        print("\n🏋️ Starting PPO Training...")
        print(f"Target: Navigate robot to point (10, 10)")
        print(f"Total timesteps: {self.total_timesteps:,}")
        
        try:
            # Create training environment
            print("🌍 Creating training environment...")
            env = DummyVecEnv([self.create_env()])
            
            # Create evaluation environment
            print("📊 Creating evaluation environment...")
            eval_env = self.create_eval_env()
            
            # Create model
            print("🧠 Creating PPO model...")
            model = self.create_model(env)
            
            # Setup callbacks
            print("⚙️ Setting up callbacks...")
            callbacks = self.setup_callbacks(eval_env)
            
            # Start training
            print(f"\\n🚦 Training started at {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 60)
            
            start_time = time.time()
            
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                tb_log_name="PPO",
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            
            print("=" * 60)
            print(f"✅ Training completed in {training_time/60:.1f} minutes!")
            
            # Save final model
            final_model_path = f"{self.save_dir}/final_model"
            model.save(final_model_path)
            print(f"💾 Final model saved to: {final_model_path}")
            
            # Clean up environments
            env.close()
            eval_env.close()
            
            return model
            
        except KeyboardInterrupt:
            print("\\n⏹️ Training interrupted by user")
            return None
        except Exception as e:
            print(f"\\n❌ Training failed with error: {e}")
            return None
    
    def test_model(self, model_path=None):
        """Test the trained model"""
        print("\\n🧪 Testing trained model...")
        
        try:
            # Load model
            if model_path is None:
                model_path = f"{self.save_dir}/best_model/best_model"
            
            if not os.path.exists(f"{model_path}.zip"):
                print(f"❌ Model not found at: {model_path}")
                return
            
            model = PPO.load(model_path)
            print(f"📂 Loaded model from: {model_path}")
            
            # Create test environment
            env = self.create_eval_env()
            
            # Run test episodes
            n_episodes = 5
            success_count = 0
            total_rewards = []
            
            print(f"🎮 Running {n_episodes} test episodes...")
            
            for episode in range(n_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                done = False
                
                print(f"\\nEpisode {episode + 1}:")
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    done = terminated or truncated
                    
                    # Print progress every 50 steps
                    if steps % 50 == 0:
                        distance = obs[3] if len(obs) > 3 else "unknown"
                        print(f"  Step {steps}: Distance to target = {distance:.2f}m")
                
                total_rewards.append(episode_reward)
                
                if 'success' in info and info['success']:
                    success_count += 1
                    print(f"  ✅ SUCCESS! Reward: {episode_reward:.1f}, Steps: {steps}")
                else:
                    reason = info.get('reason', 'unknown')
                    final_dist = info.get('final_distance', 'unknown')
                    print(f"  ❌ Failed ({reason}). Final distance: {final_dist:.2f}m, Reward: {episode_reward:.1f}")
            
            # Print summary
            success_rate = success_count / n_episodes * 100
            avg_reward = np.mean(total_rewards)
            
            print("\\n" + "=" * 50)
            print("📈 TEST RESULTS SUMMARY:")
            print(f"Success rate: {success_rate:.1f}% ({success_count}/{n_episodes})")
            print(f"Average reward: {avg_reward:.1f}")
            print(f"Reward std: {np.std(total_rewards):.1f}")
            print("=" * 50)
            
            env.close()
            
        except Exception as e:
            print(f"❌ Testing failed with error: {e}")
    
    def plot_training_results(self):
        """Plot training results if available"""
        try:
            # This would require parsing the log files
            # For now, just print a message
            print("📊 Training plots can be viewed using TensorBoard:")
            print(f"   tensorboard --logdir {self.log_dir}")
            print("   Then open http://localhost:6006 in your browser")
        except Exception as e:
            print(f"Failed to create plots: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            rclpy.shutdown()
            print("🧹 Cleanup completed")
        except:
            pass

def main():
    print("🤖 PPO Robot Navigation Training")
    print("=" * 50)
    print("Objective: Train robot to navigate to point (10, 10)")
    print("Algorithm: Proximal Policy Optimization (PPO)")
    print("Environment: Gazebo Classic with ROS 2")
    print("=" * 50)
    
    trainer = PPOTrainer()
    
    try:
        # Train the model
        model = trainer.train()
        
        if model is not None:
            # Test the trained model
            trainer.test_model()
            
            # Show how to view training plots
            trainer.plot_training_results()
            
            print(f"\\n🎉 Training completed successfully!")
            print(f"📁 Results saved in: {trainer.save_dir}")
            print(f"📊 Logs saved in: {trainer.log_dir}")
            
            # Instructions for running the trained model
            print("\\n" + "=" * 60)
            print("🚀 TO RUN THE TRAINED MODEL:")
            print("1. Start Gazebo simulation:")
            print("   ros2 launch robot_gazebo worlds.launch.py world_name:=empty")
            print("2. Run the trained model:")
            print(f"   python3 /home/ubuntu/ros2_ws/src/robot_gazebo/robot_gazebo/run_trained_model.py")
            print("=" * 60)
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()

from pathlib import Path

from SAC.SAC import SAC
from ros_python import ROS_env
from deploy_utils import DEP_env
from replay_buffer import ReplayBuffer
import torch
import numpy as np
from utils import record_eval_positions
from pretrain_utils import Pretraining

max_rad = 0.645  # maximum angle in radians for the angular velocity output
H = 0.21  # distance from the robot front wheels to the back wheels








def main(args=None):
    """Main training function"""
    action_dim = 2  # number of actions produced by the model
    max_action = 1  # maximum absolute value of output actions
    state_dim = 40+5  # number of input values in the neural network (vector length of state input)
    model_path = "src/drl_navigation_ros2/models/SAC/SAC_actor.pth"  # path to the trained model
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # using cuda if it is available, cpu otherwise
    if device.type == "cuda":
        print("Using GPU for deployment")
    else:
        print("Using CPU for deployment")
    episodes_per_epoch = 70  # how many episodes to run in single epoch
    episode = 0  # starting episode number
    train_every_n = 2  # train and update network parameters every n episodes
    training_iterations = 500  # how many batches to use for single training cycle
    batch_size = 40  # batch size for each training iteration
    max_steps = 500  # maximum number of steps in single episode
    steps = 0  # starting step number
    totalreward = 0.0
    totalcol = 0.0
    totalgoal = 0.0

    print("Loading model...")

    ros = DEP_env(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        model_path=model_path,
        )  # instantiate ROS environment

    print("Model loaded. Starting deployment...")
    ros.reset()  # reset the ROS environment to the initial state
    latest_scan, distance, cos, sin, collision, goal, a, reward, collision_count, crash = ros.step(
        lin_velocity=0.0, ang_velocity=0.0
    )  # get the initial step state
    print("Initial step done.")

    while True:
        state, terminal = ros.prepare_state(
            latest_scan, distance, cos, sin, goal, a, collision_count, crash
        )  # get state a state representation from returned data from the environment
        action = ros.get_action(state)  # get an action from the model
        a_in = [
            action[0],
            # (action[0]+1) * 0.5,
            np.tan(action[1] * max_rad) * action[0] / H,
            # action[1],
        ]  # clip linear velocity to [-0.5, 0.5] m/s range

        latest_scan, distance, cos, sin, collision, goal, a, reward, collision_count, crash = ros.step(
            lin_velocity=a_in[0], ang_velocity=a_in[1]
        )  # get data from the environment
        totalreward += reward
        totalcol += collision
        totalgoal += goal
        _, terminal = ros.prepare_state(
            latest_scan, distance, cos, sin, goal, a, collision_count, crash
        )  # get a next state representation
        
        if (
            terminal or steps == max_steps
        ):
                          
            latest_scan, distance, cos, sin, collision, goal, a, reward = ros.reset()
            steps = 0
            episode += 1
            print(f"Episode {episode} finished")
            avg_reward = totalreward / episode
            avg_col = totalcol / episode
            avg_goal = totalgoal / episode
            print(f"Average Reward: {avg_reward}")
            print(f"Average Collision rate: {avg_col}")
            print(f"Average Goal rate: {avg_goal}")
            print("..............................................")
            ros.writer.add_scalar("deploy/avg_reward", avg_reward, episode)
            ros.writer.add_scalar("deploy/avg_col", avg_col, episode)
            ros.writer.add_scalar("deploy/avg_goal", avg_goal, episode)
        else:
            steps += 1


if __name__ == "__main__":
    main()

## Commands for Hiwonder DRL Simulator

<!-- export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311 -->
export GAZEBO_RESOURCE_PATH=/workspace/src/multi_robot_scenario/launch
export LIDAR_TYPE=A1
export ROBOT_HOST=hiwonder
source /workspace/devel/setup.bash
python train_velodyne_td3_hiwonder.py

rosparam set robot_description "$(xacro /home/developer/workspace/src/hiwonder_description/urdf/jetacker.gazebo.xacro)"
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/hiwonder/cmd_vel

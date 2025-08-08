#!/bin/bash

colcon build

echo "🤖 Simple PPO Navigation Training Starter"
echo "=========================================="

# Clean up previous processes
echo "🧹 Cleaning up..."
pkill -9 -f gazebo
pkill -9 -f ignition  
pkill -9 -f gz
pkill -9 -f navigation_trainer
sleep 3

# Check if ports are still in use and wait
while netstat -tulpn 2>/dev/null | grep -q ":11345"; do
    echo "⏳ Waiting for Gazebo port to be free..."
    sleep 2
done

# Setup environment
# cd /home/evan/hiwonder
export LIDAR_TYPE=A1
source install/setup.bash

# Build touch plugin if not already built
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:touch_plugin/build


echo "🌍 Starting Gazebo..."
# Start Gazebo in background
ros2 launch robot_gazebo worlds.launch.py world_name:=all_training &
GAZEBO_PID=$!

echo "⏳ Waiting 20 seconds for Gazebo to fully start..."
sleep 10

# Check if Gazebo started successfully
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "❌ Gazebo failed to start! Check the logs above."
    exit 1
fi

# Verify robot topics are available
# echo "🔍 Checking robot topics..."
# timeout 15 ros2 topic list | grep -E "(cmd_vel|odom)" || {
#     echo "❌ Robot topics not available! Let's check what topics exist:"
#     echo "📋 Available topics:"
#     timeout 10 ros2 topic list || echo "Failed to get topic list"
#     echo ""
#     echo "🔍 Checking if differential drive plugin loaded:"
#     timeout 5 ros2 topic list | grep -E "(cmd_vel|odom|joint_states)" || echo "No movement/state topics found"
    
#     kill $GAZEBO_PID 2>/dev/null
#     exit 1
# }

# echo "🎯 Starting PPO Navigation Training to (10,10)..."

# Quick robot movement test
# echo "🧪 Testing robot movement..."
# ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" &
# sleep 2

# Check if robot position changes
# echo "📍 Checking robot position..."
# ros2 topic echo --once /odom | grep -A3 "position:" || echo "⚠️  No odometry data available"

# Start training
# python3 src/robot_gazebo/robot_gazebo/navigation_trainer.py
python3 src/drl_navigation_ros2/train.py 

# # Cleanup
# echo "🛑 Cleaning up..."
# kill $GAZEBO_PID
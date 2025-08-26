#!/bin/bash

colcon build

echo "🤖 Simple SAC Navigation Training Starter"
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
ros2 launch robot_gazebo worlds.launch.py world_name:=all_training gui:=True &
GAZEBO_PID=$!

echo "⏳ Waiting 20 seconds for Gazebo to fully start..."
sleep 10

# Check if Gazebo started successfully
if ! ps -p $GAZEBO_PID > /dev/null; then
    echo "❌ Gazebo failed to start! Check the logs above."
    exit 1
fi

python3 src/drl_navigation_ros2/deploy.py 


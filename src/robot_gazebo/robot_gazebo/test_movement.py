#!/usr/bin/env python3

import sys
import rclpy
import time
from geometry_msgs.msg import Twist

def test_robot_movement():
    """Test robot movement directly"""
    
    print("🤖 Testing Direct Robot Movement")
    print("=" * 50)
    
    # Initialize ROS 2
    rclpy.init()
    
    try:
        from rclpy.node import Node
        
        class TestNode(Node):
            def __init__(self):
                super().__init__('test_movement')
                self.cmd_pub = self.create_publisher(Twist, 'controller/cmd_vel', 10)
                self.odom_sub = self.create_subscription(
                    __import__('nav_msgs.msg', fromlist=['Odometry']).Odometry,
                    '/odom', self.odom_callback, 10
                )
                self.robot_x = 0.0
                self.robot_y = 0.0
                self.robot_yaw = 0.0
                
            def odom_callback(self, msg):
                self.robot_x = msg.pose.pose.position.x
                self.robot_y = msg.pose.pose.position.y
                
                # Convert quaternion to yaw
                orientation = msg.pose.pose.orientation
                import math
                siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
                cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
                self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        node = TestNode()
        
        print("📡 Waiting for initial odometry...")
        for i in range(50):  # Wait up to 5 seconds
            rclpy.spin_once(node, timeout_sec=0.1)
            if hasattr(node, 'robot_x'):
                break
        
        print(f"📍 Initial position: ({node.robot_x:.2f}, {node.robot_y:.2f}, {node.robot_yaw:.2f})")
        
        # Send movement command
        print("\\n🚀 Sending forward movement command...")
        twist = Twist()
        twist.linear.x = 0.5  # 0.5 m/s forward
        twist.angular.z = 0.0
        
        for i in range(20):  # Send for 2 seconds
            node.cmd_pub.publish(twist)
            time.sleep(0.1)
            rclpy.spin_once(node, timeout_sec=0.01)
            if i % 5 == 0:
                print(f"   Position: ({node.robot_x:.2f}, {node.robot_y:.2f})")
        
        # Stop robot
        print("\\n🛑 Stopping robot...")
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        node.cmd_pub.publish(twist)
        
        # Wait and check final position
        time.sleep(0.5)
        rclpy.spin_once(node, timeout_sec=0.1)
        print(f"📍 Final position: ({node.robot_x:.2f}, {node.robot_y:.2f})")
        
        distance_moved = ((node.robot_x)**2 + (node.robot_y)**2)**0.5
        print(f"📏 Distance moved: {distance_moved:.2f}m")
        
        if distance_moved > 0.1:
            print("✅ Robot movement successful!")
        else:
            print("❌ Robot did not move - check controller configuration")
        
        node.destroy_node()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    test_robot_movement()

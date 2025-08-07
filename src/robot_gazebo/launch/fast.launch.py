import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import UnlessCondition  # ADDED MISSING IMPORT
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
import xacro

# Set critical Gazebo environment variables for performance
os.environ['GAZEBO_PLUGIN_PATH'] = os.environ.get('GAZEBO_PLUGIN_PATH', '') + ':/opt/ros/foxy/lib'
os.environ['GAZEBO_MODEL_PATH'] = os.environ.get('GAZEBO_MODEL_PATH', '') + ':/usr/share/gazebo-11/models'
os.environ['GZ_SIM_RESOURCE_PATH'] = os.environ.get('GZ_SIM_RESOURCE_PATH', '') + ':/usr/share/gazebo-11'
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'  # Compatibility override

def launch_setup(context):
    # Parse launch arguments with default values
    use_sim_time = LaunchConfiguration('use_sim_time', default='true').perform(context)
    world_name = LaunchConfiguration('world_name', default='empty').perform(context)
    nav = LaunchConfiguration('nav', default='false').perform(context)
    moveit_unite = LaunchConfiguration('moveit_unite', default='false').perform(context)
    real_time_factor = LaunchConfiguration('real_time_factor', default='3.0').perform(context)
    
    # Keep headless as substitution for condition
    headless = LaunchConfiguration('headless', default='true')  # REMOVED .perform()

    # Declare configurable parameters
    return [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('world_name', default_value='all_training'),
        DeclareLaunchArgument('nav', default_value='false'),
        DeclareLaunchArgument('moveit_unite', default_value='false'),
        DeclareLaunchArgument('headless', default_value='true'),
        DeclareLaunchArgument('real_time_factor', default_value='3.0'),
        
        # Optimized Gazebo launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gzserver.launch.py'  # Headless server only
                ])
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('robot_gazebo'),
                    'worlds',
                    world_name + '.sdf'
                ]),
                'verbose': 'false',  # Reduce console output
                'extra_gazebo_args': f'--physics-engine ode '
                                     f'--iterations 20 '
                                     f'--profile /tmp/gz_profile '
                                     f'--server-plugin libPhysicsSpeedControl.so '
            }.items()
        ),
        
        # Conditionally include GUI client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gzclient.launch.py'
                ])
            ]),
            # Only launch GUI if headless=false
            condition=UnlessCondition(headless)  # FIXED CONDITION
        ),
        
        # Physics speed controller node
        Node(
            package='physics_speed_control',
            executable='speed_controller',
            name='physics_speed_controller',
            parameters=[{'real_time_factor': real_time_factor}],
            output='screen'
        ),
        
        # Robot spawner
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('robot_gazebo'),
                    'launch',
                    'spwan_model.launch.py'
                ])
            ),
            launch_arguments={
                'moveit_unite': moveit_unite,
                'world_name': world_name,
                'use_sim_time': use_sim_time,
            }.items(),
        ),
    ]

def generate_launch_description():
    # Set environment variables for performance
    SetEnvironmentVariable('GAZEBO_MASTER_URI', 'http://localhost:11345'),
    SetEnvironmentVariable('GAZEBO_IP', '127.0.0.1'),
    SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
    SetEnvironmentVariable('GAZEBO_RESOURCE_PATH', ''),
    
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_CONSOLE_OUTPUT_FORMAT', '[{severity}] [{name}]: {message}'),
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        OpaqueFunction(function=launch_setup)
    ])

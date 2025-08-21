to generate world
```
python world_generation.py
```


to start training 
```
./start.sh
```


start tensorboard
```
tensorboard --logdir runs
```

to see the process of training
change the gui = LaunchConfiguration('gui', default='True').perform(context) in src/robot_gazebo/launch/worlds.launch.py    to
```
gui = LaunchConfiguration('gui', default='False').perform(context)
```
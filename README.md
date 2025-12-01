The origin repository: https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2.git

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

## Our work is now release on arxiv
to cite our paper
```
@misc{xiong2025nonholonomicnarrowdeadendescape,
      title={Nonholonomic Narrow Dead-End Escape with Deep Reinforcement Learning}, 
      author={Denghan Xiong and Yanzhe Zhao and Yutong Chen and Zichun Wang},
      year={2025},
      eprint={2511.22338},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.22338}, 
}
```
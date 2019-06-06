#!/bin/bash
nohup python train.py Reacher-v1 -n 60000 -b 50 > reacher.log 2>&1 &
nohup python train.py InvertedPendulum-v1 > inverted_pendulum.log 2>&1 &
nohup python train.py InvertedDoublePendulum-v1 -n 12000 > inverted_double_pendulum.log 2>&1 &
nohup python train.py Swimmer-v1 -n 2500 -b 5 > swimmer.log 2>&1 &
nohup python train.py Hopper-v1 -n 30000 > hopper.log 2>&1 &
nohup python train.py HalfCheetah-v1 -n 3000 -b 5 > half_cheetah.log 2>&1 &
nohup python train.py Walker2d-v1 -n 25000 > walker2d.log 2>&1 &
nohup python train.py Ant-v1 -n 100000 > ant.log 2>&1 &
nohup python train.py Humanoid-v1 -n 200000 > humanoid.log 2>&1 &
nohup python train.py HumanoidStandup-v1 -n 200000 -b 5 > humanoid_standup.log 2>&1 &
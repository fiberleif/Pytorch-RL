# Pytorch-RL
Pytorch implementation of popular deep reinforcement learning algorithms towards SOA performance.

Implemented algorithms:
* Proximal Policy Optimization (PPO)

To be implemented algorithms:
* Deep Deterministic Policy Gradient (DDPG)
* (Double/Dueling) Deep Q-Learning (DQN)

## Dependency Installation
* Python 3.6
* Numpy 1.15
* Scipy 1.1.0
* Mujoco-py 0.5.7
* Gym 0.9.0
* sklearn 0.0
* PyTorch v0.4.0

## Code Usage
### How to run PPO algorithm
```
cd ppo
python ppo.py --env_name Reacher-v1 -n 60000 -b 50
python ppo.py --env_name InvertedPendulum-v1
python ppo.py --env_name InvertedDoublePendulum-v1 -n 12000
python ppo.py --env_name Swimmer-v1 -n 2500 -b 5
python ppo.py --env_name Hopper-v1 -n 30000
python ppo.py --env_name HalfCheetah-v1 -n 3000 -b 5
python ppo.py --env_name Walker2d-v1 -n 25000
python ppo.py --env_name Ant-v1 -n 100000
python ppo.py --env_name Humanoid-v1 -n 200000
python ppo.py --env_name HumanoidStandup-v1 -n 200000 -b 5
```

## References
* [Human Level Control through Deep Reinforcement Learning](https://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
* [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
* [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
* [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
* [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)
* Some hyper-parameters are from [DeepMind Control Suite](https://arxiv.org/abs/1801.00690), [OpenAI Baselines](https://github.com/openai/baselines) and [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)


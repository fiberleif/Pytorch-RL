# Pytorch-RL
Pytorch implementation of popular deep reinforcement learning algorithms towards SOA performance.

Implemented algorithms:
* Proximal Policy Optimization (PPO)
* Deep Deterministic Policy Gradient (DDPG) 

To be implemented algorithms:
* (Double/Dueling) Deep Q-Learning (DQN)

## Dependency
* Python 3.6
* Numpy 1.15
* Scipy 1.1.0
* Mujoco-py 0.5.7
* Gym 0.9.0
* sklearn 0.0
* PyTorch v0.4.0

## Code Usage
### Run PPO algorithm in MuJoCo Suite
```
cd ppo
python ppo_train.py --e Reacher-v1 -n 60000 -b 50
python ppo_train.py --e InvertedPendulum-v1
python ppo_train.py --e InvertedDoublePendulum-v1 -n 12000
python ppo_train.py --e Swimmer-v1 -n 2500 -b 5
python ppo_train.py --e Hopper-v1 -n 30000
python ppo_train.py --e HalfCheetah-v1 -n 3000 -b 5
python ppo_train.py --e Walker2d-v1 -n 25000
python ppo_train.py --e Ant-v1 -n 100000
python ppo_train.py --e Humanoid-v1 -n 200000
python ppo_train.py --e HumanoidStandup-v1 -n 200000 -b 5
```
### Run DDPG algorithm in MuJoCo Suite
```
cd ddpg
python ddpg_train.py --e Reacher-v1 --start_timesteps 1000
python ddpg_train.py --e InvertedPendulum-v1 --start_timesteps 1000
python ddpg_train.py --e InvertedDoublePendulum-v1 --start_timesteps 1000
python ddpg_train.py --e Swimmer-v1 --start_timesteps 1000
python ddpg_train.py --e Hopper-v1 --start_timesteps 1000
python ddpg_train.py --e HalfCheetah-v1 --start_timesteps 10000
python ddpg_train.py --e Walker2d-v1 --start_timesteps 1000
python ddpg_train.py --e Ant-v1 --start_timesteps 10000
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
* Github Repository with a lot helpful implementations: [Pat-coady](https://github.com/pat-coady/trpo), [OpenAI Baselines](https://github.com/openai/baselines) and [Ilya Kostrikov](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)


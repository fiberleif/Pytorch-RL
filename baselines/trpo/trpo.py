from baselines.common.models.policies.mlp_policy import MLPPolicy
from baselines.common.models.value_functions.mlp_value_function import MLPValueFunction
from baselines.common.utils.running_mean_std import RunningMeanStd
from baselines.trpo.sampler import Sampler
from baselines import logger
import os
import torch
import random
import numpy as np


def evaluate_policy(eval_env, policy, eval_n_episodes=10):
    eval_returns = []
    traj_max_length = 1000
    for _ in range(eval_n_episodes):
        obs = eval_env.reset()
        current_return = 0.0
        for t in range(traj_max_length):
            action = policy.select_action(obs, stochastic=False)
            obs, reward, done = eval_env.step(action)
            assert reward.size == 1
            current_return += reward[0]
            if done[0]:
                eval_returns.append(current_return)
                break
    return np.mean(eval_returns), np.std(eval_returns)


def add_adv_and_vtarg(seg, gamma, lam, value_estimator="MC"):
    """ Add value target and advantage function into segment. """
    mb_next_values = np.concatenate((seg["mb_values"], seg["last_value"]), axis=0)[1:]
    mb_deltas = seg["mb_rewards"] + (1 - seg["mb_dones"]) * gamma * mb_next_values - seg["mb_values"]

    mb_size = seg["mb_obs"].shape[0]
    seg["mb_advs"] = mb_advantages = np.empty(mb_deltas.shape, "float32")
    mb_value_target_mc = np.empty(mb_deltas.shape, "float32")
    current_adv = 0.0
    current_value_target_mc = 0.0
    for t in reversed(range(mb_size)):
        mb_advantages[t] = current_adv = gamma * lam * (1 - seg["mb_dones"]) * current_adv + mb_deltas[t]
        mb_value_target_mc[t] = current_value_target_mc = gamma * (1 - seg["mb_dones"]) * current_value_target_mc + seg["mb_rewards"][t]

    if value_estimator == "MC":
        seg["mb_value_target"] = mb_value_target_mc
    elif value_estimator == "GAE":
        seg["mb_value_target"] = seg["mb_advs"] + seg["mb_values"]


def learn(
        env,
        eval_env,
        env_id=None,
        seed=None,
        num_epochs=1000,
        timesteps_per_batch=1000, # what to train on
        gamma=0.99,
        lam=0.98,
        evaluate_freq = 10,
        network="mlp",
        network_hidden_sizes=[32, 32],
        network_activation='tanh',
        state_dependent_var=True,
        normalize_observations=True,
        max_kl=0.001,
        cg_iters=10,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        ):

    # Configure log directory.
    log_dir = os.path.join("log", "trpo", env_id, str(seed))
    logger.configure(dir=log_dir)

    # Set all seeds.
    env.seed(seed)
    eval_env.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    assert network == "mlp"
    obs_dim = env.observation_space.shape[0]  # env.observation_space.shape = (11,)
    act_dim = env.action_spave.shape[0]

    if normalize_observations:
        obs_normalizer = RunningMeanStd(shape=env.observation_space.shape)
    else:
        obs_normalizer = None

    old_policy_net = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer)
    policy_net = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer)
    value_net = MLPValueFunction(obs_dim, hidden_sizes=network_hidden_sizes, activation=network_activation, rms=obs_normalizer)

    sampler = Sampler(env, old_policy_net, value_net, timesteps_per_batch)

    for epoch in range(num_epochs):
        logger.log("********** Epoch  %i ************" % epoch)
        old_policy_net.load_state_dict(policy_net.state_dict())  # align the old policy with current policy

        seg = sampler.sample()  # sample basic data, e.g. state, action, reward, done, value, last_value.
        add_adv_and_vtarg(seg, gamma, lam)  # extract value target and advantage data from the basic data.


        if epoch % evaluate_freq == 0:
            return_mean, return_std = evaluate_policy(eval_env, policy_net)
            logger.record_tabular('return-average', return_mean)
            logger.record_tabular('return-std', return_std)
            logger.dump_tabular()



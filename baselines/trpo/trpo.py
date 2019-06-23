from baselines.common.models.policies.mlp_policy import MLPPolicy
from baselines.common.models.value_functions.mlp_value_function import MLPValueFunction
from baselines.common.utils.running_mean_std import RunningMeanStd
from baselines.trpo.sampler import Sampler
from baselines import logger
import os
import torch
import random
import numpy as np


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

    old_policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer)
    policy = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer)
    policy.load_state_dict(old_policy.state_dict())
    vf = MLPValueFunction(obs_dim, hidden_sizes=network_hidden_sizes, activation=network_activation, rms=obs_normalizer)

    sampler = Sampler(env, policy, vf, timesteps_per_batch)
    for epoch in range(num_epochs):
        logger.log("********** Epoch  %i ************" % epoch)
        old_policy.load_state_dict(policy.state_dict())  # align the old policy with current policy
        # sample batch by sampler.

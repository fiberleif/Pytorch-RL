from baselines.common.models.policies.mlp_policy import MLPPolicy
from baselines.common.models.value_functions.mlp_value_function import MLPValueFunction
from baselines.common.utils.running_mean_std import RunningMeanStd
from baselines.trpo.sampler import Sampler
from baselines.common.utils.utils import get_flat_params, set_flat_params
from baselines.common.utils.cg import conjugate_gradients
from baselines import logger
import os
import torch
import torchsnooper
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_policy(eval_env, policy, eval_n_episodes=10):
    eval_returns = []
    traj_max_length = 1000
    for _ in range(eval_n_episodes):
        obs = eval_env.reset()
        obs = obs.astype(np.float32)
        current_return = 0.0
        for t in range(traj_max_length):
            action = policy.select_action(torch.from_numpy(obs).to(device), stochastic=False)
            obs, reward, done, _ = eval_env.step(action)
            obs = obs.astype(np.float32)
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
        mb_advantages[t] = current_adv = gamma * lam * (1 - seg["mb_dones"][t]) * current_adv + mb_deltas[t]
        mb_value_target_mc[t] = current_value_target_mc = gamma * (1 - seg["mb_dones"][t]) * current_value_target_mc + seg["mb_rewards"][t]

    if value_estimator == "MC":
        seg["mb_value_targets"] = mb_value_target_mc
    elif value_estimator == "GAE":
        seg["mb_value_targets"] = seg["mb_advs"] + seg["mb_values"]


def reshape_segment_values(seg):
    assert len(seg["mb_obs"].shape) == 3
    assert len(seg["mb_actions"].shape) == 3
    obs_dim = seg["mb_obs"].shape[2]
    action_dim = seg["mb_actions"].shape[2]
    seg["mb_obs"] = seg["mb_obs"].reshape(-1, obs_dim)
    seg["mb_actions"] = seg["mb_actions"].reshape(-1, action_dim)
    seg["mb_advs"] = seg["mb_advs"].reshape(-1, 1)
    seg["mb_value_targets"] = seg["mb_value_targets"].reshape(-1, 1)


def update_value_net(seg, value_net, value_net_optimizer, vf_iters, vf_batchsize):
    seg_size = seg["mb_obs"].shape[0]
    for _ in range(vf_iters):
        idx = np.arange(seg_size)
        np.random.shuffle(idx)
        value_net_optimizer.zero_grad()
        batch_obs = torch.from_numpy(seg["mb_obs"][idx[:vf_batchsize]]).to(device)
        batch_value_target = torch.from_numpy(seg["mb_value_targets"][idx[:vf_batchsize]]).to(device)
        value_loss = torch.mean((value_net(batch_obs) - batch_value_target) ** 2)
        value_loss.backward()
        value_net_optimizer.step()


def fisher_vector_product(policy_net, get_kl_and_loss, cg_damping):
    def compute_fvp(v):
        kl, _, _ = get_kl_and_loss()
        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        kl_flat_grad = torch.cat([grad.view(-1) for grad in grads])
        kl_v = torch.sum(kl_flat_grad * v)
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        kl_flat_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
        return kl_flat_grad_grad + v * cg_damping
    return compute_fvp


# def line_search(policy_net, old_policy_net, fullstep, expected_improve_rate, max_kl, get_kl_and_loss, max_backtracks=10,):
#     prev_params = get_flat_params(old_policy_net)
#     _, surro_before, _ = get_kl_and_loss()
#     surro_before = surro_before.data
#     for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
#         new_params = prev_params + stepfrac * fullstep
#         set_flat_params(policy_net, new_params)
#         kl, surro, _ = get_kl_and_loss()
#         improve = surro - surro_before
#         expected_improve = expected_improve_rate * stepfrac
#         logger.log("Expected: %.3f Actual: %.3f" % (expected_improve, improve))
#
#         if kl > max_kl * 1.5:
#             logger.log("violated KL constraint. shrinking step.")
#         elif improve < 0:
#             logger.log("surrogate didn't improve. shrinking step.")
#         else:
#             logger.log("Stepsize OK!")
#             break
#     else:
#         set_flat_params(policy_net, prev_params)

# @torchsnooper.snoop()
def line_search(policy_net, old_policy_net, fullstep, expected_improve_rate, get_kl_and_loss, max_backtracks=10, accept_ratio=.1):
    prev_params = get_flat_params(old_policy_net)
    _, fval, _ = get_kl_and_loss()
    # fval = fval.cpu().data
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        # xnew = prev_params + torch.tensor(stepfrac).to(device) * fullstep
        xnew = prev_params + fullstep * stepfrac
        set_flat_params(policy_net, xnew)
        _, newfval, _ = get_kl_and_loss()
        # newfval = newfval.cpu().data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True
    set_flat_params(policy_net, prev_params)
    return False


def update_policy_net(seg, policy_net, old_policy_net, max_kl, cg_iters, cg_damping, ent_coef):
    # Normalize the advantages.
    mb_advs = (seg["mb_advs"] - seg["mb_advs"].mean()) / seg["mb_advs"].std()
    batch_advs = torch.from_numpy(mb_advs).to(device)
    batch_obs = torch.from_numpy(seg["mb_obs"]).to(device)
    batch_actions = torch.from_numpy(seg["mb_actions"]).to(device)

    def get_kl_and_loss():
        action_dist = policy_net(batch_obs)
        old_action_dist = old_policy_net(batch_obs)

        logp = action_dist.log_prob(batch_actions)
        old_logp = old_action_dist.log_prob(batch_actions)
        action_mean = action_dist.mean
        old_action_mean = old_action_dist.mean
        action_std = action_dist.stddev
        old_action_std = old_action_dist.stddev
        kl = torch.log(action_std) - torch.log(old_action_std) + \
             (old_action_std ** 2 + (old_action_mean - action_mean) ** 2) / (2.0 * action_std ** 2) - 0.5
        kl = torch.mean(kl)

        ratio = torch.exp(logp - old_logp)
        surro_loss = torch.mean(ratio * batch_advs)
        ent_bonus = ent_coef * action_dist.entropies()
        policy_loss = surro_loss + ent_bonus

        return kl, surro_loss, policy_loss

    kl, surro_loss, policy_loss = get_kl_and_loss()
    grads = torch.autograd.grad(policy_loss, policy_net.parameters())
    flat_gradient = torch.cat([grad.view(-1) for grad in grads])

    stepdir = conjugate_gradients(fisher_vector_product(policy_net, get_kl_and_loss, cg_damping), -flat_gradient, cg_iters)
    shs = 0.5 * (stepdir * fisher_vector_product(policy_net, get_kl_and_loss, cg_damping)(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-flat_gradient * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier:", lm[0], "grad_norm:", flat_gradient.norm()))

    # line_search(policy_net, old_policy_net, fullstep, neggdotstepdir / lm[0], max_kl, get_kl_and_loss)
    success = line_search(policy_net, old_policy_net, fullstep, neggdotstepdir / lm[0], get_kl_and_loss)
    print(success)


def learn(
        env,
        eval_env,
        env_id=None,
        seed=0,
        num_epochs=1000,
        timesteps_per_batch=1000,  # what to train on
        gamma=0.99,
        lam=0.98,
        evaluate_freq=10,
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
        vf_batchsize=128,
        ):

    # Configure log directory.
    log_dir = os.path.join("log", "trpo", env_id, str(seed))
    logger.configure(dir=log_dir)

    # Set all seeds.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    assert network == "mlp"
    obs_dim = env.observation_space.shape[0]  # env.observation_space.shape = (11,)
    act_dim = env.action_space.shape[0]

    if normalize_observations:
        obs_normalizer = RunningMeanStd(shape=env.observation_space.shape)
    else:
        obs_normalizer = None

    old_policy_net = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer).to(device)
    policy_net = MLPPolicy(obs_dim, act_dim, hidden_sizes=network_hidden_sizes, activation=network_activation,
                       state_dependent_var=state_dependent_var, rms=obs_normalizer).to(device)
    value_net = MLPValueFunction(obs_dim, hidden_sizes=network_hidden_sizes, activation=network_activation, rms=obs_normalizer).to(device)

    value_net_optimizer = torch.optim.Adam(value_net.parameters(), lr=vf_stepsize)
    sampler = Sampler(env, old_policy_net, value_net, timesteps_per_batch)

    for epoch in range(num_epochs):
        logger.log("********** Epoch  %i ************" % epoch)
        old_policy_net.load_state_dict(policy_net.state_dict())  # align the old policy with current policy

        segment = sampler.sample()  # sample basic data, e.g. state, action, reward, done, value, last_value.
        add_adv_and_vtarg(segment, gamma, lam)  # extract value target and advantage data from the basic data.
        reshape_segment_values(segment)  # reshape some values in segment (dict class) for future training.
        update_value_net(segment, value_net, value_net_optimizer, vf_iters, vf_batchsize)
        update_policy_net(segment, policy_net, old_policy_net, max_kl, cg_iters, cg_damping, ent_coef)

        if epoch % evaluate_freq == 0:
            return_mean, return_std = evaluate_policy(eval_env, policy_net)
            logger.record_tabular('return-average', return_mean)
            logger.record_tabular('return-std', return_std)
            logger.dump_tabular()



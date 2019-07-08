from baselines.common.models.policies.mlp_policy import MLPPolicy
from baselines.gail.mujoco_dset import Mujoco_Dset
from baselines.common.utils.running_mean_std import RunningMeanStd
import baselines.logger as logger
from tqdm import tqdm
import torch.optim as optim
import torch
import numpy as np
import argparse
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='dataset/hopper.npz')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    parser.add_argument('--subsample_freq', type=int, default=20)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--bc_lr', type=float, default=3e-4)
    parser.add_argument('--max_iter', help='Max iteration for training BC', type=float, default=1e5)
    return parser.parse_args()


def compute_policy_loss(policy, obs, acs):
    """ [Build torch Graph] """
    """ Compute the BC loss of the policy. """
    obs = torch.from_numpy(obs.astype(np.float32)).to(device)
    acs = torch.from_numpy(acs.astype(np.float32)).to(device)
    loss = -policy(obs).log_prob_mean(acs)
    return loss


def learn(env, policy, policy_optimizer, dataset, max_iters=1e4, optim_batch_size=128):
    """ Learn the policy with expert state-action pairs, based on BC."""
    val_per_iter = int(max_iters/10)
    for iter_so_far in tqdm(range(int(max_iters))):
        policy_optimizer.zero_grad()
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        loss = compute_policy_loss(policy, ob_expert, ac_expert)
        loss.backward()
        policy_optimizer.step()
        if iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            val_loss = compute_policy_loss(policy, ob_expert, ac_expert)

            ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
            train_loss = compute_policy_loss(policy, ob_expert, ac_expert)
            logger.log("Training loss: {}, Validation loss: {}".format(train_loss.cpu().data.numpy(), val_loss.cpu().data.numpy()))


def evaluate(env, policy, number_trajs=10):
    """ Evaluate the learned policy via perform multiple rollouts."""
    rets = []
    horizon = 1000
    for _ in range(number_trajs):
        traj_ret = 0.0
        obs = env.reset()
        obs = torch.from_numpy(obs.astype(np.float32)).to(device)
        for t in tqdm(range(horizon)):
            action = policy.select_action(obs.view(-1, obs.shape[0]), stochastic=False)[0]
            obs, reward, done, _ = env.step(action)
            obs = torch.from_numpy(obs.astype(np.float32)).to(device)
            traj_ret += reward
            if done:
                rets.append(traj_ret)
                break
    print("average return:", np.mean(rets))
    print("std of return:", np.std(rets))
    print("max of return", np.max(rets))
    print("min of return", np.min(rets))


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.env_id)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    acs_dim = env.action_space.shape[0]

    rms = RunningMeanStd(shape=env.observation_space.shape)
    policy = MLPPolicy(obs_dim, acs_dim, hidden_sizes=(100, 100), rms=rms).to(device)
    policy_optimizer = optim.Adam(policy.parameters(), lr=args.bc_lr)
    dataset = Mujoco_Dset(expert_path=args.expert_path, traj_limitation=args.traj_limitation, data_subsample_freq=args.subsample_freq)
    rms.update(dataset.obs)  # set observation normalization
    learn(env, policy, policy_optimizer, dataset, max_iters=args.max_iter)  # train policy by BC
    evaluate(env, policy, number_trajs=10)  # evaluate policy


if __name__ == "__main__":
    args = argsparser()
    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
sys.path.append('..')
import utils.logger as logger


class Policy(nn.Module):
    """ NN-based approximation of policy mean """
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult, init_policy_logvar, clipping_range=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            hid1_mult: size of first hidden layer, multiplier of obs_dim
        """
        super(Policy, self).__init__()
        # hyper-parameters
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid1_mult = hid1_mult
        self.beta = 1.0
        self.eta = 50
        self.kl_targ = kl_targ
        self.init_policy_logvar = init_policy_logvar
        self.epochs = 20
        self.lr = None
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.clipping_range = clipping_range

        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        self.hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        self.hid3_size = self.act_dim * 10  # 10 empirically determined
        self.hid2_size = int(np.sqrt(self.hid1_size * self.hid3_size))
        logger.info('Policy Params -- h1: {}, h2: {}, h3: {}'.format(self.hid1_size, self.hid2_size, self.hid3_size))

        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 9e-4 / np.sqrt(self.hid2_size)  # 9e-4 empirically determined

        # NN components of policy mean
        self.fc1 = nn.Linear(self.obs_dim, self.hid1_size)
        self.fc2= nn.Linear(self.hid1_size, self.hid2_size)
        self.fc3 = nn.Linear(self.hid2_size, self.hid3_size)
        self.fc4 = nn.Linear(self.hid3_size, self.act_dim)

        # policy logvars
        # logvar_speed is used to 'fool' gradient descent into making faster updates
        # to log-variances. heuristic sets logvar_speed based on network size.
        logvar_speed = (10 * self.hid3_size) // 48
        self.log_vars = torch.Tensor(np.ones((logvar_speed, self.act_dim)) * self.init_policy_logvar)
        self.log_vars.requires_grad = True

        # set optimizer
        self._set_optimizer()

    def forward(self, state):
        viewed_state = state.view(-1, self.obs_dim)
        # 3 hidden layers with tanh activations
        x = F.tanh(self.fc1(viewed_state))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def _set_optimizer(self):
        self.policy_mean_optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.policy_logvars_optimizer = optim.Adam([self.log_vars], lr= self.lr)

    def sample(self, state):
        state = torch.Tensor(state)
        # compute mean action
        mean_action = self.forward(state)
        # compute action noise
        policy_logvar = torch.sum(self.log_vars, dim=0)
        action_noise = torch.exp(policy_logvar / 2.0) * torch.randn(policy_logvar.size(0))
        action = mean_action + action_noise
        return action.detach().numpy()

    def update(self, observes, actions, advantages):
        # placeholder
        observes_tensor = torch.Tensor(observes)
        actions_tensor = torch.Tensor(actions)
        advantages_tensor = torch.Tensor(advantages)
        old_means_np = self.forward(observes_tensor).detach().numpy()
        old_logvar_np = torch.sum(self.log_vars, dim=0).detach().numpy()
        old_means_tensor = torch.Tensor(old_means_np)
        old_logvar_tensor = torch.Tensor(old_logvar_np)
        for e in range(self.epochs):
            policy_logvar = torch.sum(self.log_vars, dim=0)
            # logp
            logp1 = -0.5 * torch.sum(policy_logvar)
            logp2 = -0.5 * torch.sum((actions_tensor - self.forward(observes_tensor)) ** 2 / torch.exp(policy_logvar),
                                     dim=1)
            logp = logp1 + logp2
            logp_old1 = -0.5 * torch.sum(old_logvar_tensor)
            logp_old2 = -0.5 * torch.sum((actions_tensor - old_means_tensor) ** 2 / torch.exp(old_logvar_tensor),dim=1)
            logp_old = logp_old1 + logp_old2
            # kl & entropy
            log_det_cov_old = torch.sum(old_logvar_tensor)
            log_det_cov_new = torch.sum(policy_logvar)
            tr_old_new = torch.sum(torch.exp(old_logvar_tensor - policy_logvar))
            kl = 0.5 * torch.sum(log_det_cov_new - log_det_cov_old + tr_old_new +
                torch.sum((self.forward(observes_tensor) - old_means_tensor) ** 2 / torch.exp(policy_logvar), dim=1)
                - policy_logvar.shape[0])
            if self.clipping_range is not None:
                # logger.info('setting up loss with clipping objective')
                pg_ratio = torch.exp(logp - logp_old)
                clipped_pg_ratio = torch.clamp(pg_ratio, 1 - self.clipping_range[0],
                                               1 + self.clipping_range[1])
                surrogate_loss = torch.min(advantages_tensor * pg_ratio, advantages_tensor * clipped_pg_ratio)
                loss = -torch.sum(surrogate_loss)
            else:
                # logger.info('setting up loss with KL penalty')
                loss1 = -torch.sum(advantages_tensor * torch.exp(logp - logp_old))
                loss2 = kl * self.beta
                loss3 = ((torch.max(torch.Tensor([0.0]), kl - 2.0 * self.kl_targ)) ** 2) * self.beta
                loss = loss1 + loss2 + loss3
            self.policy_mean_optimizer.zero_grad()
            self.policy_logvars_optimizer.zero_grad()

            # adjust learning rate
            lr = self.lr * self.lr_multiplier
            for param_group in self.policy_mean_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.policy_logvars_optimizer.param_groups:
                param_group['lr'] = lr
            loss.backward()
            self.policy_mean_optimizer.step()
            self.policy_logvars_optimizer.step()

            # compute new kl
            policy_logvar = torch.sum(self.log_vars, dim=0)
            log_det_cov_old = torch.sum(old_logvar_tensor)
            log_det_cov_new = torch.sum(policy_logvar)
            tr_old_new = torch.sum(torch.exp(old_logvar_tensor - policy_logvar))
            kl = 0.5 * torch.sum(log_det_cov_new - log_det_cov_old + tr_old_new +
                torch.sum((self.forward(observes_tensor) - old_means_tensor) ** 2 / torch.exp(policy_logvar), dim=1)
                                 - policy_logvar.shape[0])
            kl_np = kl.detach().numpy()

            # break
            if kl_np > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl_np > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl_np < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5



if __name__ == "__main__":
    """ unit test for models """
    policy = Policy(20, 10, 0, 10, 20, 0)
    for name, value in policy.state_dict().items():
        print(name)






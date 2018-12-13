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
        self.log_vars = torch.Tensor(np.zeros((logvar_speed, self.act_dim)))
        self.log_vars.requires_grad = True

        # set optimizer
        self._set_optimizer()

    def forward(self, state):
        viewed_state = state.view(-1, self.obs_dim)
        # 3 hidden layers with tanh activations
        x = torch.tanh(self.fc1(viewed_state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x

    def _set_optimizer(self):
        self.policy_mean_optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.policy_logvars_optimizer = optim.Adam([self.log_vars], lr= self.lr)

    def sample(self, state):
        state = torch.Tensor(state)
        # compute mean action
        mean_action = self(state)
        # compute action noise
        policy_logvar = torch.sum(self.log_vars, dim=0) + self.init_policy_logvar
        action_noise = torch.exp(policy_logvar / 2.0) * torch.randn(policy_logvar.size(0))
        action = mean_action + action_noise
        return action.detach().numpy()

    def _placeholders(self, observes, actions, advantages):
        # placeholders
        self.observes_tensor = torch.Tensor(observes)
        self.actions_tensor = torch.Tensor(actions)
        self.advantages_tensor = torch.Tensor(advantages)
        self.old_means_tensor = self(self.observes_tensor).detach()
        self.old_logvar_tensor = (torch.sum(self.log_vars, dim=0) + self.init_policy_logvar).detach()

    def _logprob(self):
        # logp
        logp1 = -0.5 * torch.sum(self.policy_logvar)
        logp2 = -0.5 * torch.sum((self.actions_tensor - self(self.observes_tensor)) ** 2 / torch.exp(self.policy_logvar), dim=1)
        self.logp = logp1 + logp2
        logp_old1 = -0.5 * torch.sum(self.old_logvar_tensor)
        logp_old2 = -0.5 * torch.sum((self.actions_tensor - self.old_means_tensor) ** 2 / torch.exp(self.old_logvar_tensor), dim=1)
        self.logp_old = logp_old1 + logp_old2

    def _kl_entropy(self):
        # kl & entropy
        log_det_cov_old = torch.sum(self.old_logvar_tensor)
        log_det_cov_new = torch.sum(self.policy_logvar)
        tr_old_new = torch.sum(torch.exp(self.old_logvar_tensor - self.policy_logvar))
        self.kl = torch.mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                        torch.sum((self(self.observes_tensor) - self.old_means_tensor) ** 2 / torch.exp(self.policy_logvar), dim=1)
                        - self.policy_logvar.shape[0]) * 0.5
        self.entropy = (torch.sum(self.policy_logvar) + self.act_dim * (np.log(2 * np.pi) + 1)) * 0.5

    def _loss_train(self):
        if self.clipping_range is not None:
            # logger.info('setting up loss with clipping objective')
            pg_ratio = torch.exp(self.logp - self.logp_old)
            clipped_pg_ratio = torch.clamp(pg_ratio, 1 - self.clipping_range[0],
                                           1 + self.clipping_range[1])
            surrogate_loss = torch.min(self.advantages_tensor * pg_ratio, self.advantages_tensor * clipped_pg_ratio)
            self.loss = -torch.mean(surrogate_loss)
        else:
            # logger.info('setting up loss with KL penalty')
            loss1 = -torch.mean(self.advantages_tensor * torch.exp(self.logp - self.logp_old))
            loss2 = torch.mean(self.kl * self.beta)
            loss3 = ((torch.max(torch.Tensor([0.0]), self.kl - 2.0 * self.kl_targ)) ** 2) * self.eta
            self.loss = loss1 + loss2 + loss3

        self.policy_mean_optimizer.zero_grad()
        self.policy_logvars_optimizer.zero_grad()
        # adjust learning rate
        lr = self.lr * self.lr_multiplier
        for param_group in self.policy_mean_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.policy_logvars_optimizer.param_groups:
            param_group['lr'] = lr
        self.loss.backward()
        self.policy_mean_optimizer.step()
        self.policy_logvars_optimizer.step()

    def update(self, observes, actions, advantages):
        # placeholder
        self._placeholders(observes, actions, advantages)
        for e in range(self.epochs):
            # train policy
            self.policy_logvar = torch.sum(self.log_vars, dim=0) + self.init_policy_logvar
            self._logprob()
            self._kl_entropy()
            self._loss_train()

            # test
            self.policy_logvar = torch.sum(self.log_vars, dim=0) + self.init_policy_logvar
            # compute new kl
            self._kl_entropy()
            kl_np = self.kl.detach().numpy()
            entropy_np = self.entropy.detach().numpy()
            # compute new loss
            self._logprob()
            self._loss_train()
            loss_np = self.loss.detach().numpy()
            # break
            if kl_np > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        #print("epoch-{0}-finished".format(e))
        #print("PolicyLoss:", loss_np)
        #print("KL:", kl_np)
        #print("entropy:", entropy_np)

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






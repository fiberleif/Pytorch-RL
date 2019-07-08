import torch.distributions as dist


# ReinforcedNormal
class ReinforcedNormal(dist.Normal):
    def __init__(self, loc, scale, validate_args=None):
        super(ReinforcedNormal, self).__init__(loc, scale, validate_args=validate_args)

    def log_prob_mean(self, actions):
        return self.log_prob(actions).mean()

    def log_probs(self, actions):
        return self.log_prob(actions).sum()

    def entropies(self):
        return self.entropy().sum()

    def mode(self):
        return self.mean


# ReinforcedCategorical
class ReinforceCategorical(dist.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None):
        super(ReinforceCategorical, self).__init__(probs=probs, logits=logits, validate_args=validate_args)

    def new_sample(self):
        return self.sample().unsqueeze(-1) # N * num_actions -> N * 1 * num_actions

    def log_probs(self, actions):
        return self.log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

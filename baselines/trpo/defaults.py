
def atari():
    return dict(
        # network=cnn_small(),
        timesteps_per_batch=512,
        max_kl=0.001,
        cg_iters=10,
        cg_damping=1e-3,
        gamma=0.98,
        lam=1.0,
        vf_iters=3,
        vf_stepsize=1e-4,
        entcoeff=0.00,
    )


def mujoco():
    return dict(
        # run hyper-parameters.
        num_epochs=1000,
        timesteps_per_batch=1000,
        gamma=0.99,
        lam=0.98,
        evaluate_freq=10,

        # network hyper-parameters.
        network_hidden_sizes=[32, 32],
        network_activation='tanh',
        normalize_observations=True,

        # algorithm hyper-parameters.
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        vf_iters=5,
        vf_stepsize=1e-3,
    )

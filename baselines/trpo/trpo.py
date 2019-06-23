
def learn(
        env,
        seed=None,
        num_epochs=1000,
        timesteps_per_batch=1000, # what to train on
        gamma = 0.99,
        lam = 0.98,
        evaluate_freq = 10,
        network="mlp",
        network_hidden_sizes = [32, 32],
        network_activation = 'tanh',
        normalize_observations = True,
        max_kl=0.001,
        cg_iters=10,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        callback=None,
        load_path=None,
        ):

import sys
import re
import multiprocessing
import gym
from collections import defaultdict
from baselines.common.vec_env import VecFrameStack, VecNormalize
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from importlib import import_module


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def build_env(args, train=True):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    alg = args.alg
    seed = args.seed
    env_type, env_id = get_env_type(args)

    if env_type in {'atari'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        else:
            frame_stack_size = 4
            if train:
                env = make_vec_env(env_id, env_type, args.num_env or ncpu, seed, reward_scale=args.reward_scale)
            else:
                env = make_vec_env(env_id, env_type, 1, seed, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        flatten_dict_observations = alg not in {'her'}
        if train:
            env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)
        else:
            env = make_vec_env(env_id, env_type, 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        # if env_type == 'mujoco':
        #     env = VecNormalize(env, use_tf=True)

    return env


# Parse the gym registry.
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    # Search corresponding env_type for env_id
    env_type=None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    if ':' in env_id:
        env_type = re.sub(r':.*', '', env_id)
    assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def train(args, extra_args):
    # Get env type (e.g. Atari, Mujoco).
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    eval_env = build_env(args, train=False)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    alg_kwargs['env_id'] = env_id

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        eval_env=eval_env,
        seed=seed,
        **alg_kwargs
    )

    return model, env, eval_env


def main(args):
    # Parse arguments.
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    # Train with DRL algorithms.
    model, env, eval_env = train(args, extra_args)

    # Close environments.
    env.close()
    eval_env.close()
    return model


if __name__ == '__main__':
    main(sys.argv)

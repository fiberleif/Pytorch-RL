import argparse
import torch


class Hyper(object):
    env_name = "CartPole-v0"
    num_epochs = 500
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    batch_size = 128
    max_path_length = 200
    discount = 0.99
    epsilon = 0.2
    tau = 0.001
    hard_update_period = 1000
    replay_buffer_size = 1000000
    is_double = True
    device = "cpu"


    @ classmethod
    def parse_arguments(cls):
        parser = argparse.ArgumentParser()
        bool_mapper = lambda str: True if 'True' in str or 'true' in str else False
        for key, value in vars(Hyper).items():
            if '__' not in key and 'parse_arguments' not in key and 'help' not in key:
                if isinstance(value, bool):
                    parser.add_argument('--' + key, type=bool_mapper, default=value)
                else:
                    parser.add_argument('--' + key, type=type(value), default=value)
        args = parser.parse_args()
        print(vars(args))
        for key, value in vars(args).items():
            setattr(cls, key, value)
        # special attribute
        cls.device = torch.device("cuda" if torch.cuda.is_available() and "gpu" in cls.device else "cpu")


    @ classmethod
    def help(cls):
        for key, value in vars(Hyper).items():
            if '__' not in key and 'parse_arguments' not in key and 'help' not in key:
                print(key, value)


if __name__ == '__main__':
    Hyper.help()
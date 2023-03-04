import os
import argparse
from shutil import rmtree
from time import strftime, localtime
import yaml
from train import train


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", "-t", action="store_true",
                        help="train mode, you must give the --output and --config")
    parser.add_argument("--eval", "-e", action="store_true",
                        help="eval mode, you must give the --output and --config and the --checkpoint")

    parser.add_argument('--output', '-o', type=str, default=None,
                        help='if the mode is train: the dir to store the file;'
                             'if the mode is eval or ave: the path of output')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='used in all the modes, the path of the config yaml')

    parser.add_argument('--checkpoint', type=str,
                        help='used in the eval, ave and test mode, the path of the checkpoint')
    parser.add_argument('--test_config', type=str, default='configs/soma_nuclei_rev_test.yaml',
                        help='the test config yaml file, used in the test mode')
    args = parser.parse_args()

    return args


def get_basedir(base_dir, start_new_model=False):
    # init the output folder structure
    if base_dir is None:
        base_dir = os.path.join("./", strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    if start_new_model and os.path.exists(base_dir):
        rmtree(base_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not os.path.exists(os.path.join(base_dir, 'logs')):
        os.mkdir(os.path.join(base_dir, 'logs'))  ##tensorboard
    if not os.path.exists(os.path.join(base_dir, 'checkpoint')):
        os.mkdir(os.path.join(base_dir, 'checkpoint'))  ##checkpoint
    return base_dir


def main():
    args = get_args()
    with open(args.config, "r", encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["TrainConfig"]["gpu"]
    basedir = get_basedir(args.output, config["TrainConfig"]["start_new_model"])
    if args.train:
        train(config, basedir, args.config)


if __name__ == "__main__":
    main()

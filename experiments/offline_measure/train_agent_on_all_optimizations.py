import argparse
from pathlib import Path

from learned_mean_cartpole import run_sim


def offline_path(offline_seed):
    return Path(__file__).parent.resolve() / f'offline_{offline_seed}'


def mean_number_iter(offline_seed):
    modelpath = offline_path(offline_seed) / 'models' / 'safety_model'
    for mpath in modelpath.iterdir():
        modelname = mpath.stem
        modelnum = int(modelname.split('_')[-1])
        yield modelnum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('offline_seed', type=int)

    args = parser.parse_args()

    for mean_number in mean_number_iter(args.offline_seed):
        run_sim(
            offline_seed=args.offline_seed,
            mean_number=mean_number,
            output_dir=offline_path(args.offline_seed)/'models'/'learned_mean'
        )

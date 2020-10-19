import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from edge.dataset import Dataset

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def exp_has_safety(exppath):
    safetypath = exppath / 'models' / 'safety_model'
    return safetypath.exists()


def compute_test_series(test):
    meas = 'measurement'
    groups = test.df.groupby([meas, test.EPISODE])
    reward = groups.sum().groupby([meas]).mean()[test.REWARD]
    failures = (groups[test.FAILED].any().astype({test.FAILED: float})
                    .groupby([meas]).mean())
    return reward, failures

def compute_train_series(train, window=10):
    groups = train.df.groupby([train.EPISODE])
    reward = groups.sum()[train.REWARD].rolling(window=window)
    failures = (groups[test.FAILED].any().astype({train.FAILED: float})
                .rolling(window=window))
    reward = reward.mean()
    failures = failures.mean()
    return reward, failures

def plot_series(train_r, train_f, test_r, test_f, has_safety):
    title = f"{'with' if has_safety else 'without'} safety model"
    figure = plt.figure(figsize=(4.8, 5.5))
    gs = figure.add_gridspec(2, 1)

    ax_r = figure.add_subplot(gs[0, 0])
    ax_f = figure.add_subplot(gs[1, 0], sharex=ax_r)

    train_r.plot(label=f'Train reward', ax=ax_r, marker='+')
    test_r.plot(label=f'Test reward', ax=ax_r, marker='x')
    train_f.plot(label=f'Train failures', ax=ax_f, marker='+')
    test_f.plot(label=f'Test failures', ax=ax_f, marker='x')

    def setup_ax(ax, name):
        ax.grid(True)
        ax.legend(loc='best')
        ax.set_xlabel('Episodes')
        ax.set_ylabel(f'{name} ({title})')
        ax.set_xlim(0, train_r.index[-1])
    setup_ax(ax_r, 'Reward')
    setup_ax(ax_f, 'Failures')
    return figure

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expnum', help='Number of the full_test experiment',
                        type=int)
    parser.add_argument('--full', help='Full name of the experiment, if not a '
                                       'full_test experiment')
    args = parser.parse_args()
    expnum = args.expnum
    full = args.full
    if expnum is None and full is None:
        raise ValueError('Please specify either `expnum` or `full`')

    expname = full if expnum is None else f'lander_{expnum}'
    exppath = Path(__file__).absolute().parent / expname

    has_safety = exp_has_safety(exppath)

    datapath = exppath / 'data'
    figpath = exppath / 'figs'
    trainpath = datapath / 'training_samples.csv'
    testpath = datapath / 'testing_samples.csv'

    train = Dataset.load(trainpath)
    test = Dataset.load(testpath)

    test_reward, test_failures = compute_test_series(test)
    train_reward, train_failures = compute_train_series(train, window=2)

    fig = plot_series(train_reward, train_failures, test_reward, test_failures,
                      has_safety)

    savepath = figpath / 'metrics.pdf'

    fig.savefig(str(savepath), format='pdf')



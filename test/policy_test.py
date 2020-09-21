import unittest
import numpy as np
import matplotlib.pyplot as plt

from edge.model.policy_models.bayesian import ExpectedImprovementPolicy
from edge.model.inference import MaternGP
from edge.space import StateActionSpace, Segment


def get_spaces():
    state_space = Segment(0, 1, 101)
    action_space = Segment(-1, 2, 301)
    sa_space = StateActionSpace(
        state_space=state_space,
        action_space=action_space
    )
    return state_space, action_space, sa_space

def plot_prediction(x, truth, mean, acquisition, lower, upper,
                    x_samples, y_samples, sample, constraints=None):
    figure = plt.figure(constrained_layout=True, figsize=(5.5, 4.8))
    gs = figure.add_gridspec(2, 1)

    ax_f = figure.add_subplot(gs[0, 0])
    ax_a = figure.add_subplot(gs[1, 0], sharex=ax_f)

    ax_f.plot(x, truth, label='truth', color='g')
    ax_f.plot(x, mean.reshape(-1), label='mean', color='b')
    ax_f.fill_between(x, lower, upper, alpha=0.5)
    ax_f.legend(loc='best')
    ax_f.scatter(x_samples, y_samples, marker='x', color='black')
    ax_f.axvline(sample, linestyle='--', color='black')
    ax_f.grid(True)

    ax_a.plot(x, acquisition, label='acquisition', color='r')
    ax_a.axvline(sample, linestyle='--', color='black')
    ax_a.grid(True)
    ax_a.legend(loc='best')

    if constraints is not None:
        changes = np.diff(constraints, prepend=False, append=False)
        idx_changes = np.argwhere(changes) - 1
        x_fill = [(idx_changes[i], idx_changes[i+1])
                     for i in range(0, len(idx_changes) - 1, 2)]
        for start_fill, end_fill in x_fill:
            ax_f.axvspan(x[start_fill], x[end_fill], color='g', alpha=0.2)
            ax_a.axvspan(x[start_fill], x[end_fill], color='g', alpha=0.2)

    plt.show()

def get_gp(x_train, y_train, noise):
    return MaternGP(
        x_train.reshape((-1, 1)),
        y_train,
        noise_prior=(noise**2, 1e-3),
        lengthscale_prior=(1, 1e-3),
        outputscale_prior=(1, 1e-3)
    )

class ExpectedImprovementTest(unittest.TestCase):
    def test_1d(self):
        state_space, action_space, sa_space = get_spaces()

        noise = 0.2

        def f(x, noise=noise):
            return -np.sin(3 * x) - x ** 2 + 0.7 * x + \
                   noise * np.random.randn(*x.shape)

        n_train = 2
        x_train = np.array([-0.9, 1.1])
        y_train = f(x_train)
        best_sample = y_train.max()

        gp = get_gp(x_train, y_train, noise)
        xi = 0.01
        ei_policy = ExpectedImprovementPolicy(sa_space, xi)

        n_iter = 10
        # This reshape normally happens in GPModel
        x = action_space[:].squeeze()
        actions = x.reshape((-1, 1))
        truth = f(x, noise=0)
        for i in range(n_iter):
            prediction = gp.predict(actions)
            mean = prediction.mean.numpy()
            covar = prediction.variance.detach().numpy()
            action = ei_policy.get_action(mean, covar, best_sample)

            # Plotting
            acquisition = ei_policy.acquisition_function(mean, covar).reshape(
                -1)
            lower, upper = prediction.confidence_region()
            plot_prediction(
                x=x,
                truth=truth,
                mean=mean,
                acquisition=acquisition,
                lower=lower.numpy(),
                upper=upper.numpy(),
                x_samples=gp.train_x.numpy(),
                y_samples=gp.train_y.numpy(),
                sample=action,
                constraints=None,
            )

            value = f(action)
            gp.append_data(x=np.atleast_2d(action), y=np.atleast_1d(value))
            best_sample = max(best_sample, value)
        self.assertTrue(True)

    def test_constrained_1d(self):
        state_space, action_space, sa_space = get_spaces()

        noise = 0.2
        def f(x, noise=noise):
            return -np.sin(3 * x) - x ** 2 + 0.7 * x + \
                   noise * np.random.randn(*x.shape)
        n_train = 2
        x_train = np.array([-0.9, 1.1])
        y_train = f(x_train)
        best_sample = y_train.max()

        gp = get_gp(x_train, y_train, noise)
        xi = 0.01
        ei_policy = ExpectedImprovementPolicy(sa_space, xi)

        n_iter = 10
        # This reshape normally happens in GPModel
        x = action_space[:].squeeze()
        actions = x.reshape((-1, 1))
        truth = f(x, noise=0)
        constraints = np.logical_and(
            actions > -0.1,
            actions < 1.2
        ).squeeze()
        for i in range(n_iter):
            prediction = gp.predict(actions)
            mean = prediction.mean.numpy()
            covar = prediction.variance.detach().numpy()
            action = ei_policy.get_action(mean, covar, best_sample,
                                          constraints=constraints)

            # Plotting
            acquisition = ei_policy.acquisition_function(mean, covar).reshape(
                -1)
            lower, upper = prediction.confidence_region()
            plot_prediction(
                x=x,
                truth=truth,
                mean=mean,
                acquisition=acquisition,
                lower=lower.numpy(),
                upper=upper.numpy(),
                x_samples=gp.train_x.numpy(),
                y_samples=gp.train_y.numpy(),
                sample=action,
                constraints=constraints,
            )

            value = f(action)
            gp.append_data(x=np.atleast_2d(action), y= np.atleast_1d(value))
            best_sample = max(best_sample, value)
        self.assertTrue(True)

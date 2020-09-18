"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Continuous version by Ian Danforth
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from edge.gym_wrappers import GymEnvironmentWrapper


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = self.is_failure_state(x, theta)
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width /world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen-polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def is_failure_state(self, x, theta):
        return x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians

    def close(self):
        if self.viewer:
            self.viewer.close()


class ContinuousCartPole(GymEnvironmentWrapper):
    def __init__(self, discretization_shape, control_frequency=4):
        gym_env = ContinuousCartPoleEnv()
        super(ContinuousCartPole, self).__init__(
            gym_env, discretization_shape, control_frequency=control_frequency
        )

    def is_failure_state(self, state):
        x = state[0]
        theta = state[2]
        return self.gym_env.is_failure_state(x, theta)

    @property
    def in_failure_state(self):
        return self.is_failure_state(self.s)

    @property
    def done(self):
        # Gym gives 195 for the cartpole's standard reward threshold
        return self._done or self.reward_accumulator >= 195

    def reset(self, s=None):
        state = super(ContinuousCartPole, self).reset(s)
        self.reward_accumulator = 0
        return state

    def step(self, action):
        # ContinuousCartPoleEnv does not stop automatically when reaching the
        # reward threshold, so we enforce it
        new_state, reward, failed = super(ContinuousCartPole, self).step(action)
        self.reward_accumulator += reward
        return new_state, reward, failed

    def linearization(self, discrete_time=True):
        # The full equations are taken from
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # which is the source given by OpenAI Gym for the dynamics
        g = self.gym_env.gravity
        m = self.gym_env.total_mass
        eta = self.gym_env.masspole / m
        l = self.gym_env.length * 2  # Gym stores half the pole's length
        a12 = g * eta / (eta - 4 / 3)
        a32 = (g / l) / (4 / 3 - eta)
        A = np.array([[0, 1, 0, 0],
                      [0, 0, a12, 0],
                      [0, 0, 0, 1],
                      [0, 0, a32, 0]])
        B = (1 / m) * np.array([0,
                                (4 / 3) / (4 / 3 - eta),
                                0,
                                -1 / l / (4 / 3 - eta)]).reshape((-1, 1))
        # The action we give to Gym is in [-1, 1]: we rescale B accordingly
        B = self.gym_env.force_mag * B

        if discrete_time:
            # A and B are the dynamics matrices of the continuous dynamics.
            # We need to apply the integration scheme to get the matrices of the
            # time-discrete one.
            tau = self.control_frequency if self.control_frequency is not None \
                else self.gym_env.tau
            A = np.eye(A.shape[0], dtype=float) + tau * A
            B = tau * B

        return A, B
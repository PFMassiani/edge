import unittest
from inspect import signature
import numpy as np

from edge.dynamics import event, EventBased
from edge.dynamics import HovershipDynamics
from edge.space import Segment, StateActionSpace


def number_of_parameters(func):
    sig = signature(func)
    params = sig.parameters
    return len(params)


class TestEvent(unittest.TestCase):
    def test_event_decoration(self):
        class Foo(EventBased):
            def __init__(self):
                self.foo = 1

            @event(True, 1)
            def event_0(self, t, y):
                """A terminal event with direction 1"""
                return self.foo

            @event(-1)
            def event_1(self, t, y):
                """A nonterminal event with direction -1"""
                return self.foo

            @event
            def event_2(self, t, y):
                """An event with default parameterization"""
                return self.foo

            @event()
            def event_3(self, t, y):
                """A simple event"""
                return self.foo

        doc = ['', '', '', '']
        name = ['', '', '', '']
        params = [None, None, None, None]

        doc[0] = "A terminal event with direction 1"
        name[0] = "event_0"
        params[0] = (True, 1)
        doc[1] = "A nonterminal event with direction -1"
        name[1] = "event_1"
        params[1] = (False, -1)
        doc[2] = "An event with default parameterization"
        name[2] = "event_2"
        params[2] = (False, 0)
        doc[3] = """A simple event"""
        name[3] = 'event_3'
        params[3] = (False, 0)

        module = self.__module__
        events = [
            Foo.event_0,
            Foo.event_1,
            Foo.event_2,
            Foo.event_3
        ]

        bar = Foo()
        bar_events = [bar.event_0, bar.event_1, bar.event_2, bar.event_3]
        for foovalue in [1, 2]:
            bar.foo = foovalue
            for n in range(len(events)):
                for k in range(2):
                    if k == 0:
                        evtfunc = bar.get_events()[n]
                    elif k == 1:
                        evtfunc = bar_events[n]
                    self.assertEqual(evtfunc.__name__, name[n])
                    self.assertEqual(evtfunc.__doc__, doc[n])
                    self.assertEqual(evtfunc.__module__, module)
                    self.assertEqual(number_of_parameters(evtfunc), 2)

                    self.assertTrue(evtfunc.is_event)
                    self.assertEqual(evtfunc.terminal, params[n][0])
                    self.assertEqual(evtfunc.direction, params[n][1])

                    self.assertEqual(evtfunc(0, 0), foovalue)


class HovershipTests(unittest.TestCase):

    def test_still_hovership(self):
        hovership_dynamics = HovershipDynamics(
            ground_gravity=0,
            gravity_gradient=0,
            control_frequency=1,
            max_thrust=1,
            max_altitude=1
        )
        state_space = hovership_dynamics.stateaction_space.state_space

        initial_state = state_space.sample()
        action = np.atleast_1d(0)
        new_state = hovership_dynamics.step(initial_state, action)

        self.assertTrue(hovership_dynamics.is_feasible_state(initial_state))
        self.assertTrue(hovership_dynamics.is_feasible_state(new_state))
        self.assertEqual(initial_state, new_state)

    def test_hovership_fall(self):
        hovership_dynamics = HovershipDynamics(
            ground_gravity=10,
            gravity_gradient=0,
            control_frequency=1,
            max_thrust=1,
            max_altitude=1
        )
        state_space = hovership_dynamics.stateaction_space.state_space

        initial_state = state_space.sample()
        action = np.atleast_1d(0)
        new_state = hovership_dynamics.step(initial_state, action)

        self.assertTrue(hovership_dynamics.is_feasible_state(initial_state))
        self.assertTrue(hovership_dynamics.is_feasible_state(new_state))
        self.assertEqual(new_state, np.atleast_1d(0))

    def test_rocket_hovership(self):
        hovership_dynamics = HovershipDynamics(
            ground_gravity=0,
            gravity_gradient=0,
            control_frequency=1,
            max_thrust=1,
            max_altitude=1
        )
        state_space = hovership_dynamics.stateaction_space.state_space

        initial_state = state_space.sample()
        action = np.atleast_1d(1)
        new_state = initial_state
        for t in range(10):
            new_state = hovership_dynamics.step(new_state, action)

        self.assertTrue(hovership_dynamics.is_feasible_state(initial_state))
        self.assertTrue(hovership_dynamics.is_feasible_state(new_state))
        self.assertEqual(new_state, np.atleast_1d(1))

    def test_oscillating_hovership(self):
        TOL = 1e-7

        hovership_dynamics = HovershipDynamics(
            ground_gravity=0,
            gravity_gradient=0,
            control_frequency=1,
            max_thrust=1,
            max_altitude=1
        )
        state_space = Segment(0, 1, 100)
        action_space = Segment(-1, 1, 100)
        stateaction_space = StateActionSpace(
            state_space,
            action_space
        )
        hovership_dynamics.stateaction_space = stateaction_space

        initial_state = np.atleast_1d(0.5)
        actions = [np.atleast_1d(0.1), np.atleast_1d(-0.1)]
        previous_state = None
        state = initial_state
        for t in range(10):
            new_state = hovership_dynamics.step(state, actions[t % 2])
            if previous_state is not None:
                self.assertTrue(abs(previous_state[0] - new_state[0]) < TOL)
            previous_state, state = state, new_state

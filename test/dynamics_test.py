import unittest
from inspect import signature

from edge.dynamics import event, EventBased


def number_of_parameters(func):
    sig = signature(func)
    params = sig.parameters
    return len(params)


class TestEvent(unittest.TestCase):
    def test_event_based_class(self):
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

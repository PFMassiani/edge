import unittest

from edge.dynamics import event, event_based


class TestEvent(unittest.TestCase):
    def test_event_based_class(self):
        @event_based
        class Foo:
            def __init__(self):
                pass

            @event(True, 1)
            def event_0(t, y):
                """A terminal event with direction 1"""
                return y

            @event(-1)
            def event_1(t, y):
                """A nonterminal event with direction -1"""
                return y

            @event
            def event_2(t, y):
                """An event with default parameterization"""
                return y

            @event()
            def event_3(t, y):
                """A simple event"""
                return t + y

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

        for n in range(len(events)):
            evtfunc = Foo.get_events()[n]
            self.assertEqual(evtfunc.__name__, name[n])
            self.assertEqual(evtfunc.__doc__, doc[n])
            self.assertEqual(evtfunc.__module__, module)

            self.assertTrue(evtfunc.is_event)
            self.assertEqual(evtfunc.terminal, params[n][0])
            self.assertEqual(evtfunc.direction, params[n][1])

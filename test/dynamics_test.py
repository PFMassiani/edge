import unittest

from edge.dynamics import event, event_based


class TestEvent(unittest.TestCase):
    # def test_event_decorator(self):
    #     doc = ['', '', '']
    #     name = ['', '', '']
    #     params = [None, None, None]
    #
    #     @event(True, 1)
    #     def terminal_positive_event(t, y):
    #         """A terminal event with direction 1"""
    #         return y
    #     doc[0] = "A terminal event with direction 1"
    #     name[0] = "terminal_positive_event"
    #     params[0] = (True, 1)
    #
    #     @event(-1)
    #     def negative_event(t, y):
    #         """A nonterminal event with direction -1"""
    #         return y
    #     doc[1] = "A nonterminal event with direction -1"
    #     name[1] = "negative_event"
    #     params[1] = (False, -1)
    #
    #     @event
    #     def default_event(t, y):
    #         """An event with default parameterization"""
    #         return y
    #     doc[2] = "An event with default parameterization"
    #     name[2] = "default_event"
    #     params[2] = (False, 0)
    #
    #     module = self.__module__
    #     events = [terminal_positive_event, negative_event, default_event]
    #
    #     for n in range(len(events)):
    #         evtfunc = events[n]
    #         evtfunc(0)
    #         self.assertEqual(evtfunc.__name__, name[n])
    #         self.assertEqual(evtfunc.__doc__, doc[n])
    #         self.assertEqual(evtfunc.__module__, module)

    def test_event_based_class(self):
        @event_based
        class Foo:
            def __init__(self):
                pass

            @event(True, 1)
            def event_method(t, y):
                """A simple event"""
                return t + y

        doc = """A simple event"""
        name = 'event_method'
        module = self.__module__

        self.assertEqual(Foo.event_method.__name__, name)
        self.assertEqual(Foo.event_method.__doc__, doc)
        self.assertEqual(Foo.event_method.__module__, module)

        self.assertTrue(Foo.event_method.is_event)
        self.assertTrue(Foo.event_method.terminal)
        self.assertEqual(Foo.event_method.direction, 1)

        o = Foo()
        self.assertEqual(o.get_events(), [o.event_method])

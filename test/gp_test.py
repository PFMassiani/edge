import unittest
import torch
import numpy as np

from edge.model.inference.tensorwrap import tensorwrap


class TestTensorwrap(unittest.TestCase):
    def check_tensor(self, *args, **kwargs):
        for a in args:
            self.assertTrue(torch.is_tensor(a))
        for k, a in kwargs.items():
            self.assertTrue(torch.is_tensor(a))

    def test_for_method(self):
        @tensorwrap
        def m1(x, y):
            self.check_tensor(x, y)

        @tensorwrap(1)
        def m2(x, y):
            self.check_tensor(x)
            self.assertFalse(torch.is_tensor(y))

        @tensorwrap('y')
        def m3(x, y):
            self.check_tensor(y)
            self.assertFalse(torch.is_tensor(x))

        t = np.array([0])
        m1(t, y=t)
        m2(t, t)
        m3(t, t)

    def test_for_class(me):
        class Foo:
            @tensorwrap
            def m1(self, x, y):
                me.check_tensor(x, y)

            @tensorwrap(1)
            def m2(self, x, y):
                me.check_tensor(x)
                me.assertFalse(torch.is_tensor(y))

            @tensorwrap('y')
            def m3(self, x, y):
                me.check_tensor(y)
                me.assertFalse(torch.is_tensor(x))

            @staticmethod
            @tensorwrap(1)
            def m4(x, y):
                me.check_tensor(x)
                me.assertFalse(torch.is_tensor(y))

        bar = Foo()
        t = np.array([0])

        bar.m1(t, y=t)
        bar.m2(t, t)
        bar.m3(t, t)
        bar.m4(t, t)
        Foo.m4(t, t)

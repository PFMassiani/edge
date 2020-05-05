import unittest
import edge
from edge.space import Segment, Box, DiscreteProductSpace


class TestSpaces(unittest.TestCase):
    def test_segment(self):
        segment = Segment(0, 1, 3)
        values = [0, 0.5, 1]
        x = segment.sample()
        self.assertTrue(x in values)
        self.assertTrue(segment.contains(x))
        self.assertTrue(segment.contains(0.5))
        self.assertTrue(segment.contains(0))
        self.assertTrue(segment.contains(1))
        self.assertTrue(not segment.contains(0.25))
        self.assertTrue(not segment.contains(-0.5))
        self.assertTrue(not segment.contains(1.5))

        idx = segment.sample_idx()
        self.assertTrue(isinstance(idx, int))
        self.assertTrue(0 <= idx)
        self.assertTrue(2 >= idx)

        idx_iterator = segment.get_index_iterator()
        for i in idx_iterator:
            self.assertTrue(isinstance(i, int))
            self.assertTrue(segment[i] in values)
            self.assertTrue(segment[i] in segment)
            self.assertEqual(i, segment.indexof(segment[i]))

        try:
            Segment(-1, 0, 2)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)
        try:
            Segment(0, 0, 2)
            self.assertTrue(False)
        except Exception:
            self.assertTrue(True)

    def test_box_1(self):
        box = Box(0, 1, shape=(3, 3))
        values_1d = [0, 0.5, 1]
        values = [[x, y] for x in values_1d for y in values_1d]
        x = box.sample()
        self.assertTrue(x in values)
        self.assertTrue(x in box)
        self.assertTrue([0, 0] in box)
        self.assertTrue((-1, 0) not in box)
        self.assertTrue((0.5, 0.25) not in box)

        idx = box.sample_idx()
        indexes_1d = (0, 1, 2)
        indexes = ((i, j) for i in indexes_1d for j in indexes_1d)
        self.assertTrue(isinstance(idx, tuple))
        self.assertEqual(len(idx), 2)
        self.assertTrue(idx in indexes)

        for i in box.get_index_iterator():
            self.assertTrue(box[i] in values)
            self.assertTrue(box[i] in box)
            self.assertEqual(i, box.indexof(box[i]))

    def test_box_2(self):
        box = Box([0, 0], [1, 1], shape=(3, 3))
        values_1d = [0, 0.5, 1]
        values = [[x, y] for x in values_1d for y in values_1d]
        x = box.sample()
        self.assertTrue(x in values)
        self.assertTrue(x in box)
        self.assertTrue([0, 0] in box)
        self.assertTrue((-1, 0) not in box)
        self.assertTrue((0.5, 0.25) not in box)

        idx = box.sample_idx()
        indexes_1d = (0, 1, 2)
        indexes = ((i, j) for i in indexes_1d for j in indexes_1d)
        self.assertTrue(isinstance(idx, tuple))
        self.assertEqual(len(idx), 2)
        self.assertTrue(idx in indexes)

        for i in box.get_index_iterator():
            self.assertTrue(box[i] in values)
            self.assertTrue(box[i] in box)
            self.assertEqual(i, box.indexof(box[i]))

    def test_product_of_boxes(self):
        b1 = Box([0, 0], [1, 1], shape=(3, 3))
        b2 = Box(4, 5, shape=(11, 11))
        p = DiscreteProductSpace(b1, b2)

        x = p.sample()
        self.assertTrue(x in p)

        for i in iter(p):
            self.assertTrue(p[i] in p)
            self.assertEqual(i, p.indexof(p[i]))


if __name__ == '__main__':
    unittest.main()

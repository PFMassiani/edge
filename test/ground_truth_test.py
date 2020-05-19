import unittest

from edge.model.safety_models import SafetyTruth
from edge.envs import Hovership
from edge.space import StateActionSpace


class SafetyTruthTest(unittest.TestCase):
    def test_from_vibly(self):
        env = Hovership()
        truth = SafetyTruth(env)

        vibly_file_path = '../vibly/data/dynamics/hover_map.pickle'
        truth.from_vibly_file(vibly_file_path)

        self.assertTrue(isinstance(truth.stateaction_space, StateActionSpace))
        self.assertEqual(truth.viable_set.shape, truth.measure.shape)
        self.assertEqual(truth.viable_set.shape, truth.unviable_set.shape)
        self.assertEqual(truth.viable_set.shape, truth.failure_set.shape)

    def test_get_training_examples(self):
        env = Hovership()
        truth = SafetyTruth(env)

        vibly_file_path = '../vibly/data/dynamics/hover_map.pickle'
        truth.from_vibly_file(vibly_file_path)

        train_x, train_y = truth.get_training_examples(n_examples=2000)
        self.assertEqual(train_x.shape[0], train_y.shape[0])
        self.assertEqual(train_x.shape[0], 2000)
        self.assertEqual(train_x.shape[1],
                         truth.stateaction_space.index_dim)
        train_x, train_y = truth.get_training_examples(
            n_examples=2000, from_failure=True, viable_proportion=0.6
        )
        self.assertEqual(train_x.shape[0], train_y.shape[0])
        self.assertEqual(train_x.shape[0], 2000)
        self.assertEqual(train_x.shape[1],
                         truth.stateaction_space.index_dim)
        self.assertTrue((train_y[:1200] > 0).all())
        self.assertTrue((train_y[1200:] == 0).all())

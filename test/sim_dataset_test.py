import unittest
import numpy as np
from pathlib import Path

from edge.dataset import Dataset
from edge.utils import average_performances


class DatasetTest(unittest.TestCase):
    def _test_columns_equal(self, ds, columns, wo_group):
        self.assertEqual(set(ds.columns), set(columns))
        self.assertCountEqual(ds.columns, columns)
        self.assertEqual(set(ds.columns_wo_group), set(wo_group))
        self.assertCountEqual(ds.columns_wo_group, wo_group)

    def _test_data_equal(self, ds, data):
        dsdata = ds.to_numpy()
        self.assertTrue(
            (dsdata == data).all(),
            f'Arrays are not equal.\nds:\n{dsdata}\ndata:\n{data}'
        )

    def test_creation(self):
        ds = Dataset()
        self._test_columns_equal(ds, Dataset.DEFAULT_COLUMNS, Dataset.DEFAULT_COLUMNS_WO_EPISODE)
        wo_group = ['hello', 'world']
        group = 'my_groupname'
        ds = Dataset(*wo_group, group_name=group)
        self._test_columns_equal(ds, list((*wo_group, group)), wo_group)

    def test_add_data(self):
        ds = Dataset()
        s, a, s_ = np.array([0., 1., 2.])[:, np.newaxis]
        entry_args = [10, 1., s, a, s_]
        entry_kwargs = {'done': False, 'failed': True}
        ds.add_entry(*entry_args, **entry_kwargs)
        data = np.array([[10, 1., s, a, s_, True, False]])
        self._test_data_equal(ds, data)

        group = {ds.REWARD: [1.], ds.STATE: [s], ds.ACTION: [a], ds.NEW: [s_],
                 ds.FAILED: [True], ds.DONE: [False]}
        ds.add_group(group, group_number=100)
        new_data = data.copy()
        new_data[0, 0] = 100
        data = np.vstack((data, new_data))
        self._test_data_equal(ds, data)

        ds = Dataset('a', 'b', group_name='group')
        data = [[1., 10], ['b value', 'other b value']]
        group = dict(zip(['a', 'b'], data))
        ds.add_group(group)
        data = np.array([[0, 0]] + data, dtype=object).T
        self._test_data_equal(ds, data)

    def test_load(self):
        def test_load_preserves_ds(ds):
            sdir = Path('/tmp')
            spath = sdir / (ds.name + '.csv')
            ds = Dataset(name='my_dataset')
            ds.save(sdir)
            load = Dataset.load(spath)
            self._test_columns_equal(ds, load.columns, load.columns_wo_group)
            self._test_data_equal(ds, load.to_numpy())

        ds = Dataset('a', 'b', group_name='group', name='my_dataset.csv')

        entry_args = [10, 1.]
        entry_kwargs = {'b': False}
        ds.add_entry(*entry_args, **entry_kwargs)
        group = {'a': [2., 3.], 'b': [False, True]}
        ds.add_group(group, group_number=100)

        test_load_preserves_ds(ds)

    def test_average_performances(self):
        ds = Dataset(group_name='training')
        s, a, s_ = np.array([0., 1., 2.])[:, np.newaxis]
        entry_args = [10, 0, 1., s, a, s_]
        entry_kwargs = {'done': False, 'failed': True}
        ds.add_entry(*entry_args, **entry_kwargs)
        entry_args[1] = 1
        entry_kwargs = {'done': False, 'failed': False}
        ds.add_entry(*entry_args, **entry_kwargs)
        ep = {ds.REWARD: [1., 2.], ds.STATE: [s, s_], ds.ACTION: [a, a],
              ds.NEW: [s_, s], ds.FAILED: [True, False], ds.DONE: [False, True]}
        for n_ep in range(6):
            ep[ds.EPISODE] = [n_ep, n_ep]
            ep[ds.REWARD] = [n_ep, n_ep + 1.]
            ep[ds.FAILED] = [False, (n_ep % 2) == 0]
            ds.add_group(ep, group_number=11)

        perfs = average_performances(ds.df, ds.group_name, ds.EPISODE)
        truth = (np.mean([1, 1, 1, 3, 5, 7, 9, 11]),
                 np.mean([True, False] * 4))
        self.assertTupleEqual(perfs, truth)

        df = ds.loc[ds.df[ds.group_name] == 11]
        perfs = average_performances(df, ds.group_name, ds.EPISODE,
                                     last_n_episodes=3)
        truth = (9, 1/3)
        self.assertTupleEqual(perfs, truth)
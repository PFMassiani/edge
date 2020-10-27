import ast
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import zip_longest


def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))


class Dataset:
    EPISODE = 'episode'
    REWARD = 'reward'
    STATE = 'state'
    ACTION = 'action'
    NEW = 'new_state'
    FAILED = 'failed'
    DONE = 'done'
    _INDEX = 'index'

    DEFAULT_COLUMNS = [EPISODE, REWARD, STATE, ACTION, NEW, FAILED, DONE]
    DEFAULT_COLUMNS_WO_EPISODE = [REWARD, STATE, ACTION, NEW, FAILED, DONE]
    DEFAULT_ARRAY_CAST = [STATE, ACTION, NEW]

    def __init__(self, *columns, group_name=None, name=None):
        self.group_name = group_name if group_name is not None\
            else self.EPISODE
        self.columns = self.DEFAULT_COLUMNS if len(columns) == 0\
            else list(columns)
        self.columns_wo_group = [cname for cname in self.columns
                                 if cname != self.group_name]
        self.columns = [self.group_name] + self.columns_wo_group
        self.df = pd.DataFrame(columns=self.columns)
        self.df.index.name = Dataset._INDEX
        self.name = name

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)
        return getattr(self.df, item)

    def _complete_args(self, args):
        return [[arg] for _, arg in zip_longest(self.columns, args)]

    def _list_wrap(self, args):
        if isinstance(args, dict):
            return {argname: [arg] for argname, arg in args.items()}
        else:
            return [[arg] for arg in args]

    def add_entry(self, *args, **kwargs):
        entry = dict(zip(self.columns, self._list_wrap(
            self._complete_args(args)
        )))
        entry.update(self._list_wrap(kwargs))
        self.df = self.df.append(pd.DataFrame(entry), ignore_index=True)

    def add_group(self, group, group_number=None):
        if group_number is None:
            group_number = self.df[self.group_name].max() + 1
            if pd.isna(group_number):
                group_number = 0
        group_length = len(group[list(group.keys())[0]])
        group = group.copy()
        group.update({
            self.group_name: [group_number]*group_length
        })
        self.df = self.df.append(pd.DataFrame(group), ignore_index=True)

    def save(self, filepath):
        filepath = Path(filepath)
        if filepath.is_dir():
            filepath = filepath / (self.name + '.csv')
        with filepath.open('w') as f:
            self.df.to_csv(f)

    @staticmethod
    def _get_index_key(df):
        dflt_idx = 'Unnamed: 0'
        return dflt_idx if dflt_idx in list(df.columns) else Dataset._INDEX

    @staticmethod
    def load(filepath, *array_cast, group_name=None):
        if len(array_cast) == 0:
            array_cast = Dataset.DEFAULT_ARRAY_CAST
        filepath = Path(filepath)
        converters = {cname: from_np_array for cname in array_cast}
        with filepath.open('r') as f:
            df = pd.read_csv(f, converters=converters)
        idxkey = Dataset._get_index_key(df)
        df = df.set_index(idxkey)
        df.index.name = Dataset._INDEX

        name = filepath.stem
        cols = [cname for cname in df.columns if (group_name is None) or
                                                 (cname != group_name)]
        ds = Dataset(*cols, group_name=group_name, name=name)
        ds.df = df
        return ds

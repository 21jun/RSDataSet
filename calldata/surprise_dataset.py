import os

import numpy as np
import pandas as pd

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

class SurpriseDataset:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.DataFrame()

    def __getitem__(self, i):
        return self.df.iloc[i]



class SupriseMovielensDataset(SurpriseDataset):

    def __init__(self, file_path='./dataset/ml-100k/u.data'):
        super().__init__(file_path)
        reader = Reader(line_format='user item rating timestamp', sep='\t')
        self.data = Dataset.load_from_file(file_path, reader=reader)
        self.df = pd.DataFrame(self.data.raw_ratings, columns=['uid', 'iid', 'rate', 'timestamp'])


class SupriseEachMovieDataset(SurpriseDataset):     

    def __init__(self, file_path):
        super().__init__(file_path)
        reader = Reader(line_format='user item rating', sep=' ')
        self.data = Dataset.load_from_file(file_path, reader=reader)
        self.df = pd.DataFrame(np.array(self.data.raw_ratings)[:,:3], columns=['user', 'item', 'rating'])

import numpy as np
from pathlib import Path 


class RSDataSet:
    def __init__(self, filepath):
        self.filepath = filepath
        self.x = []
        self._load_dataset()

    def _load_dataset(self):
        pass
    
    def __getitem__(self, i):
        return self.x[i]

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        pass


class RSDataLoader():
    def __init__(self, ds: RSDataSet, bs=1):
        self.ds, self.bs = ds, bs

    def __iter__(self):
        for i in range(0, len(self.ds), self.bs):
            yield self.ds[i : i+self.bs]


class MovieLensDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def _load_dataset(self):

        user_max=0
        item_max=0
        with open(self.filepath) as f:
            for line in f:
                user_id, item_id, _, _ = line.split('\t')
                user_max = max([user_max, int(user_id)])
                item_max = max([item_max, int(item_id)])

        matrix = [[0 for col in range(item_max)] for row in range(user_max)]

        with open(self.filepath) as f:
            for line in f:
                user_id, item_id, rating, _ = line.split('\t')
                matrix[int(user_id)-1][int(item_id)-1] = int(rating)

        self.x = matrix
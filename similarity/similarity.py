import numpy as np
from abc import abstractmethod


class Similarity:

    def __init__(self):
        self.name = self.__class__.__name__
        self.m = None
        self.sim_ = None
        # 테스트용
        self.test_matrix = np.array([[4, 1, 1, 4],
                                     [0, 4, 2, 0],
                                     [2, 0, 4, 5],
                                     [1, 4, 0, 1]])

    def set_base(self, base):
        options = ['user', 'item', 'test']
        assert base.lower() in options
        if base is 'test':
            self.m = self.test_matrix
            return

        self.m = self.m if base is 'user' else self.m.T

    # 클래스마다 별도의 구현 필요
    @abstractmethod
    def fit(self, dataset, base='user'):
        pass


class CosineSimilarity(Similarity):

    def __init__(self):
        super().__init__()

    def fit(self, dataset, base='user'):
        self.m = np.array(dataset[:])
        self.set_base(base)

        d = self.m @ self.m.T
        norm = np.linalg.norm(self.m, axis=1)
        norm = norm.reshape(norm.shape[0], -1)
        self.sim_ = d / (norm * norm.T)


class PearsonCorrelationCoefficient(Similarity):

    def __init__(self):
        super().__init__()

    def fit(self, dataset, base='user'):
        self.m = np.array(dataset[:])
        self.set_base(base)

        x = self.m
        mean_x = x.mean(axis=1)
        xm = np.subtract(x, mean_x.reshape(-1, 1))

        d = xm @ xm.T
        norm = np.linalg.norm(xm, axis=1)
        norm = norm.reshape(norm.shape[0], -1)
        self.sim_ = d / (norm * norm.T)


class JaccardSimilarity(Similarity):

    def __init__(self):
        super().__init__()

    def fit(self, dataset, base='user'):
        self.m = np.array(dataset[:])
        self.set_base(base)
        x = np.where(np.isnan(self.m), 0, 1)

        sim = [0 for x in range(x.shape[0])]
        for i in range(x.shape[0]):
            if i % 1000 == 0: print(i, " / ", x.shape[0])
            intersection = np.logical_and(x[i], x)
            union = np.logical_or(x[i], x)
            jac = intersection.sum(1) / union.sum(1)
            sim[i] = jac

        self.sim_ = np.array(sim)


class PIPSimilarity(Similarity):

    def __init__(self, Rmax, Rmin):
        super().__init__()
        self.Rmax = Rmax
        self.Rmin = Rmin
        self.Rmed = (self.Rmax + self.Rmax) / 2.0

    def agreement(self, r1, r2):
        Rmed = self.Rmed
        if (r1 > Rmed and r2 < Rmed) or (r1 < Rmed and r2 > Rmed):
            return False
        return True

    def proximity(self, r1, r2):
        
        if self.agreement(r1, r2):
            dist = abs(r1 - r2)
        else:
            dist = 2*abs(r1 - r2)
            
        return ((2*(self.Rmax-self.Rmin)+1)-dist)**2

    def impact(self, r1, r2):
        if self.agreement(r1,r2):
            return ((abs(r1-self.Rmed)+1)*(abs(r2-self.Rmed)+1))
        else:
            return (1.0/((abs(r1-self.Rmed)+1)*(abs(r2-self.Rmed)+1)))

    def popularity(self, r1, r2, item_no):
        arr = self.m
        arr = arr.astype('float')
        arr[arr == 0.0] = np.nan

        mu = np.nanmean(arr, axis=0)[item_no]
        if (r1 > mu and r2 > mu) or (r1 < mu and r2 < mu):
            return (1 + (((r1+r2)/2.0 - mu)**2))
        else:
            return 1
        
    def PIP(self, u1, u2):
        
        common = np.logical_and(self.m[u1], self.m[u2])
        result = 0.0
        for r1, r2, idx in zip(self.m[u1][common], self.m[u2][common], np.argwhere(common==True)):
            result += self.proximity(r1, r2) * self.impact(r1, r2) * self.popularity(r1, r2, idx[0])

        return result
            

    def fit(self, dataset, u1, u2, base='user'):
        self.m = np.array(dataset[:])
        self.set_base(base)
        self.sim_ = self.PIP(u1, u2)

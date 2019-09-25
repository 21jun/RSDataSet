import numpy as np
from abc import abstractmethod

class Similarity():

    def __init__(self):
        
        self.name = self.__class__.__name__      
        self.m = None
        self.sim_ = None
        # 테스트용
        self.test_matrix = np.array([[4,1,1,4],
                                    [0,4,2,0],
                                    [2,0,4,5],
                                    [1,4,0,1]])

    def set_base(self, base):
        
        options = ['user', 'item', 'test']
        assert base.lower() in options
        if base is 'test':  
            self.m = self.test_matrix
            return

        self.m = self.m if base is 'user' else self.m.T

    # 클래스마다 별도의 구현 필요
    @abstractmethod
    def fit(self):
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
        # self.sim_ = np.corrcoef(self.m)
        x = self.m
        mean_x = x.mean(axis=1)
        xm = np.subtract(x, mean_x.reshape(-1,1))

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
        x = np.where(np.isnan(self.m),0, 1)
        
        sim = [0 for x in range(x.shape[0])]
        for i in range(x.shape[0]):
            if i%1000 == 0: print(i," / ",x.shape[0])
            intersection = np.logical_and(x[i], x)
            union = np.logical_or(x[i], x)
            jac = intersection.sum(1) / union.sum(1)           
            sim[i]=jac

        self.sim_ = np.array(sim)
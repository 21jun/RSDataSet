import numpy as np

class Similarity():

    def __init__(self, dataset):
        self.name = self.__class__.__name__
        self.ds = dataset
        self.matrix = np.array(dataset[:])  # numpy로 관리
        self.options = ['user', 'item']
        self.sim_ = None

    def set_base(self, base):
        _base = base.lower()
        assert _base in self.options, "Choose btw [user][item]"
        self.base = _base
        self.m = self.matrix if self.base is 'user' else self.matrix.T

    # 클래스마다 별도의 구현 필요
    def fit(self):
        pass

    def fit_one(self):
        pass

    def __repr__(self):
        return f"{self.name}\ndataset : {self.ds.name}\nbase : {self.base}\n"

class COS(Similarity):

    def __init__(self, dataset):
        super().__init__(dataset)
        # nan 값을 모두 0.0 으로 치환
        self.matrix = np.nan_to_num(self.matrix)

    def fit(self, base='user'):
        self.set_base(base)
        # item base 일때 transpose로 처리
        d = self.m @ self.m.T
        norm = np.linalg.norm(self.m, axis=1)
        norm = norm.reshape(norm.shape[0], -1)
        self.sim_ = d / (norm * norm.T)

class PCC(Similarity):

    def __init__(self, dataset):
        super().__init__(dataset)

    def fit(self, base='user'):
        self.set_base(base)
        mean = np.nanmean(self.m, axis=1)
        mean = mean.reshape(mean.shape[0], -1)
        B = self.m - mean
        C = np.nan_to_num(B)
        dot = C @ C.T
        for i in range(B.shape[0]):
            u = B[i]
            for j in range(B.shape[1]):
                v = B[j]
                not_nan = np.invert(np.isnan(u) | np.isnan(v))

                u_ = u[not_nan]
                v_ = v[not_nan]
                
                norm = np.linalg.norm(u_) * np.linalg.norm(v_) 
                dot[i][j] = dot[i][j] / norm 

        return dot
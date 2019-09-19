import numpy as np

class Similarity():

    def __init__(self, dataset):
        self.name = self.__class__.__name__
        self.ds = dataset
        self.matrix = np.array(dataset[:])  # numpy로 관리
        self.options = ['user', 'item', 'test']
        self.sim_ = None

        self.test_matrix =\
            np.array([[4,1,1,4],
                     [np.nan,4,2,np.nan],
                     [2,np.nan,4,5],
                     [1,4,np.nan,1]])

    def set_base(self, base):
        _base = base.lower()
        assert _base in self.options, "Choose btw [user][item]"
        self.base = _base
        self.m = self.matrix if self.base is 'user' else self.matrix.T
        if base is 'test':
            self.m = self.test_matrix
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

    def fit(self, base='user'):
        self.set_base(base)
        # nan 값을 모두 0.0 으로 치환
        self.m = np.nan_to_num(self.m)
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
        # PCC는 데이터셋에 따라 0을 nan으로 바꿔줘야함.!!!
        # dataset에서 0으로 나오는 애들 모두 nan으로 바꿔두자. (더 의미 있음)
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
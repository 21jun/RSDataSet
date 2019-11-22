import numpy as np
from similarity.similarity import Similarity
from calldata.call_dataset import RSDataSet
import time

# TODO: 아래 TODO 코드들을 주석제거하면 PPT와 같은 결과가 나옵니다.
# 제가 생각하기에 0을 null으로 처리하면 몇몇 데이터셋(Jester)에서 
# 평점 0점을 처리하지 못하여 0을 null으로 처리하지 않도록 수정하였습니다.

# TODO
# def COS(mat):
#     NumUsers = np.size(mat, axis=0)
#     Sim = np.full((NumUsers, NumUsers), 0.0)
#     for u in range(0, NumUsers):
#         arridx_u = np.where(mat[u] == 0)
#         for v in range(u + 1, NumUsers):
#             arridx_v = np.where(mat[v] == 0)
#             arridx = np.unique(np.concatenate((arridx_u, arridx_v), axis=None))
#
#             U = np.delete(mat[u], arridx)
#             V = np.delete(mat[v], arridx)
#
#             if np.linalg.norm(U) == 0 or np.linalg.norm(V) == 0:
#                 Sim[u, v] = 0
#             else:
#                 Sim[u, v] = np.dot(U, V) / (np.linalg.norm(U) * np.linalg.norm(V))
#             Sim[v, u] = Sim[u, v]
#     return Sim


class CollaborativeFiltering:

    def __init__(self, with_='none'):
        self.m = np.array([])
        self.sim_ = None
        self.predicted_rating = None
        self.dataset = None

        with_options = ['none', 'mean', 'baseline']
        assert with_.lower() in with_options
        self.with_ = with_
        # 테스트용
        self.test_mat = np.array([[4, 1, 1, 4, 2, 3, 5, 0, 4, 0],
                                  [0, 4, 2, 0, 3, 2, 5, 0, 4, 3],
                                  [2, 0, 4, 5, 2, 0, 1, 3, 4, 2],
                                  [1, 4, 0, 1, 4, 5, 0, 3, 1, 2],
                                  [1, 2, 3, 4, 0, 3, 4, 4, 2, 5]])

    def predict_rating(self, sim_list, rating_list, knn):
        if self.with_ == 'none':
            numerator = np.dot(sim_list, rating_list)
            numerator = np.array([numerator[i][i] for i in range(self.m.shape[0])])
            denominator = np.sum(sim_list, axis=1)
            self.predicted_rating = numerator / denominator.reshape(-1, 1)

        if self.with_ == 'mean':
            mean = np.nanmean(self.m, axis=1)

            # TODO
            # mean = np.nanmean(np.where(self.m != 0, self.m, np.nan), axis=1)

            mean_list = mean[knn]
            rating_list_sub_mean = np.array(
                [rating_list[i] - mean_list[i].reshape(-1, 1) for i in range(mean_list.shape[0])])
            rating_list_sub_mean.reshape(-1, rating_list_sub_mean.shape[2])

            numerator = np.dot(sim_list, rating_list_sub_mean)
            numerator = np.array([numerator[i][i] for i in range(self.m.shape[0])])
            denominator = np.sum(sim_list, axis=1)

            self.predicted_rating = mean.reshape(-1, 1) + (numerator / denominator.reshape(-1, 1))

        if self.with_ == 'baseline':
            all_rating_mean = self.m.mean()  # 전체평균
            mean_u = np.nanmean(self.m, axis=1)
            mean_i = np.nanmean(self.m, axis=0)
            # TODO:
            # mean_u = np.nanmean(np.where(self.m != 0, self.m, np.nan), axis=1)
            # mean_i = np.nanmean(np.where(self.m != 0, self.m, np.nan), axis=0)

            bu = mean_u - all_rating_mean
            bi = mean_i - all_rating_mean
            baseline = np.add(bu.reshape(-1, 1), bi) + all_rating_mean
            baseline_list = baseline[knn]
            rating_list_sub_baseline = rating_list - baseline_list

            numerator = np.dot(sim_list, rating_list_sub_baseline)
            numerator = np.array([numerator[i][i] for i in range(self.m.shape[0])])
            denominator = np.sum(sim_list, axis=1)

            self.predicted_rating = baseline + numerator / denominator.reshape(-1, 1)

    def set_base(self, base):
        options = ['user', 'item', 'test']
        assert base.lower() in options
        if base is 'test':
            self.m = self.test_mat
            return

        self.m = self.m if base is 'user' else self.m.T

    def fit(self, dataset: RSDataSet, sim: Similarity, k, base='user'):
        start_time = time.time()
        self.dataset = dataset
        self.m = np.array(dataset[:])

        self.set_base(base)

        assert self.m.shape[0] > k, f"자기 자신까지 이웃으로 지정할 수 없습니다. self.m.shape[0]:{self.m.shape[0]} <= {k}"

        # Similarity measure
        sim.fit(self.m)
        self.sim_ = sim.sim_

        # TODO (코사인만)
        # self.sim_ = COS(self.m)
        # self.sim_ = np.array(self.sim_)

        knn = np.argsort(-self.sim_)  # 오름 차순 정렬후의 인덱스를 반환
        knn = knn[:, 1:k + 1]  # 자기 자신과는 이웃으로 처리하지 않기 위해서 맨 앞 제거
        # PCC, COS 유사도는 자기 자신과의 유사도가 1 이므로 정렬하면 맨 앞이 자기 자신

        # TODO
        # knn = np.delete(knn, np.s_[k:], 1)

        sim_list = self.sim_[:, knn]
        sim_list = np.array([sim_list[i][i] for i in range(self.m.shape[0])])
        rating_list = self.m[knn]

        self.predict_rating(sim_list, rating_list, knn)

        print(f'---Collaborative Filtering with {self.with_}---')
        print(f'{self.dataset}')
        print(f'neighbors :{k}')
        print(f'execute time :{time.time() - start_time}')

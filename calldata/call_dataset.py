import numpy as np
import pandas as pd

class RSDataSet:
    def __init__(self, filepath):
        self.name = self.__class__.__name__
        self.filepath = filepath
        self.user_num, self.item_num, self.x = 0, 0, []
        self.mapping_table = {} # user_ids, item_ids

        self._load_dataset()
        self._verify_dataset()
    
    # 데이터셋 마다 다른 로드 구현이 필요
    def _load_dataset(self):
        pass
    
    # 인덱스로 해당 유저의 id 를 검색
    def get_user_id(self, i):
        assert i < self.user_num
        return self.mapping_table['user_ids'][i]
    
    # 인덱스로 해당 아이템의 id 를 검색
    def get_item_id(self, i):
        assert i < self.item_num
        return self.mapping_table['item_ids'][i]
        
    # 생성한 데이터셋이 정상적으로 로드되었는지 검증
    def _verify_dataset(self):
        assert self.user_num > 0
        assert self.item_num > 0
        assert self.user_num == self.__len__()
    
    # 생성한 인스턴스에서 바로 User-Item Rating Matrix 로 접근 가능
    def __getitem__(self, i):
        return self.x[i]

    def __len__(self):
        return len(self.x)

    # 인스턴스의 정보 확인용
    def __repr__(self):
        return \
        f'\n{self.name}\n' +\
        f'number of users : {self.user_num}\n' +\
        f'number of items : {self.item_num}'

class MovieLensDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def _load_dataset(self):

        user_max, item_max = 0, 0
        with open(self.filepath) as f:
            for line in f:
                user_id, item_id, _, _ = line.split('\t')
                user_max = max([user_max, int(user_id)])
                item_max = max([item_max, int(item_id)])

        self.user_num = user_max
        self.item_num = item_max

        matrix = [[0 for col in range(item_max)] for row in range(user_max)]

        with open(self.filepath) as f:
            for line in f:
                user_id, item_id, rating, _ = line.split('\t')
                matrix[int(user_id)-1][int(item_id)-1] = int(rating)

        self.x = matrix

        user_ids = [id for id in range(1, self.user_num)]
        item_ids = [id for id in range(1, self.item_num)]

        self.mapping_table = {'user_ids': user_ids, 'item_ids': item_ids}

class BookCrossingDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def _load_dataset(self):

        df = pd.read_csv(self.filepath, sep=';', encoding='utf-8')
        df = df.iloc[:1000, :]  # 앞쪽 1000개의 row 만 사용
        # 전체 데이터로 메트릭스 만들면 메모리 초과 (over 16gb)

        matrix = df.pivot(index='User-ID', columns='ISBN', values='Book-Rating')
        self.user_num, self.item_num = matrix.shape
        self.x = matrix.values.tolist()

        user_ids = matrix.index.values.tolist()
        item_ids = matrix.columns.values.tolist()

        self.mapping_table = {'user_ids': user_ids, 'item_ids': item_ids}

class JesterDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)

    def _load_dataset(self):
    
        df = pd.read_excel(self.filepath, header=None)
        df= df.iloc[:, 1:]  # 첫번째 컬럼은 필요없기에 제거
        self.user_num , self.item_num = df.shape
        matrix = df
        self.x = matrix.values.tolist()

        user_ids = matrix.index.values.tolist()
        item_ids = matrix.columns.values.tolist()

        self.mapping_table = {'user_ids': user_ids, 'item_ids': item_ids}

class EachMovieDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)
    
    def _load_dataset(self):
        df = pd.read_csv(self.filepath, sep=' ', header=None)
        df.columns = ['item_id', 'user_id', 'rating']
        matrix = df.pivot(index='user_id', columns='item_id', values='rating')

        self. user_num, self.item_num = matrix.shape
        self.x = matrix.values.tolist()

        user_ids = matrix.index.values.tolist()
        item_ids = matrix.columns.values.tolist()

        self.mapping_table = {'user_ids': user_ids, 'item_ids': item_ids}


class NetflixDataSet(RSDataSet):
    def __init__(self, filepath):
        super().__init__(filepath)

    def _load_dataset(self):
        pass


# 데이터로더에 데이터셋을 등록하면 원하는 배치사이즈 만큼 씩 iteration 가능
class RSDataLoader():
    def __init__(self, ds: RSDataSet, bs=1):
        self.ds, self.bs = ds, bs

    def __iter__(self):
        for i in range(0, len(self.ds), self.bs):
            yield self.ds[i : i+self.bs]

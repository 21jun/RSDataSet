from calldata.RS_dataset import JesterDataSet, MovieLensDataSet, EachMovieDataSet
from similarity.similarity import CosineSimilarity
from similarity.similarity import PearsonCorrelationCoefficient
from similarity.similarity import JaccardSimilarity
from predict.collaborative_filtering import CollaborativeFiltering
from pandas import DataFrame

if __name__ == '__main__':
    # 데이터셋 불러오기
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')

    


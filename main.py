from calldata.RS_dataset import JesterDataSet, MovieLensDataSet, EachMovieDataSet
from similarity.similarity import CosineSimilarity
from similarity.similarity import PearsonCorrelationCoefficient
from similarity.similarity import JaccardSimilarity
from similarity.similarity import PIPSimilarity
from predict.collaborative_filtering import CollaborativeFiltering
from pandas import DataFrame

if __name__ == '__main__':
    # 데이터셋 불러오기
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')
    # EachMovie = EachMovieDataSet('./dataset/rec-eachmovie/rec-eachmovie.edges')
    
    u1 = 0
    u2 = 9

    pip = PIPSimilarity(5, 0)
    pip.fit(MovieLens, u1, u2)
    print(pip.sim_)
from calldata.RS_dataset import JesterDataSet, MovieLensDataSet, EachMovieDataSet
from similarity.similarity import CosineSimilarity
from similarity.similarity import PearsonCorrelationCoefficient
from similarity.similarity import JaccardSimilarity
from predict.collaborative_filtering import CollaborativeFiltering
from pandas import DataFrame

if __name__ == '__main__':
    # 데이터셋 불러오기
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')

    # 유사도 모듈 불러오기
    cos = CosineSimilarity()
    pcc = PearsonCorrelationCoefficient()

    # basic Collaborative Filtering
    basic = CollaborativeFiltering(with_='none')
    basic.fit(MovieLens, pcc, k=3, base='test')
    print(DataFrame(basic.predicted_rating))

    # Collaborative Filtering with baseline
    baseline = CollaborativeFiltering(with_='baseline')
    baseline.fit(MovieLens, cos, k=2, base='user')
    print(DataFrame(baseline.predicted_rating))
    
    # Collaborative Filtering with mean
    mean = CollaborativeFiltering(with_='mean')
    mean.fit(MovieLens, cos, k=2, base='test')
    print(DataFrame(mean.predicted_rating))


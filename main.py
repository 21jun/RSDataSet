from calldata.call_dataset import JesterDataSet, MovieLensDataSet
from similarity.similarity import CosineSimilarity 
from similarity.similarity import PearsonCorrelationCoefficient
from similarity.similarity import JaccardSimilarity


if __name__ == '__main__':
    

    Jester = JesterDataSet('./dataset/jester-data-1/jester-data-1.xls')
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')
    
    cos = CosineSimilarity()
    cos.fit(MovieLens)
    print(cos.sim_)

    # pcc = PearsonCorrelationCoefficient()
    # pcc.fit(MovieLens)
    # print(pcc.sim_)

    # jac = JaccardSimilarity()
    # jac.fit(s, 'user')
    # print(jac.sim_)
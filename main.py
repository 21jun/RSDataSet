from calldata.call_dataset import *
from similarity.similarity import COS, PCC
import numpy as np
if __name__ == '__main__':
    

    # Jester = JesterDataSet('./dataset/jester-data-1/jester-data-1.xls')
    # MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')
    # EachMovie = EachMovieDataSet('./dataset/rec-eachmovie/rec-eachmovie.edges')
    BXBook = BookCrossingDataSet('./dataset/BX-CSV-Dump/BX-Book-Ratings.csv')
    # print(EachMovie)# 
    
    cos = COS(BXBook)
    cos.fit(base='user')
    print(cos.sim_)
    
    # pcc = PCC(MovieLens)
    # print(pcc.fit())
    # print(cos.sim_)
    # print(MovieLens)
    # print(pcc.fit())
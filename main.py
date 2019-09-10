from calldata.call_dataset import *

if __name__ == '__main__':
    
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')
    print(MovieLens)
    print(f'MovieLens[195][241] = {MovieLens[195][241]}')

    Jester = JesterDataSet('./dataset/jester-data-1/jester-data-1.xls')
    print(Jester)
    print(f'Jester[0][0] = {Jester[0][0]}')
    Jester.get_user_id(1343)
    Jester.get_item_id(5)

    BookCrossing = BookCrossingDataSet('./dataset/BX-CSV-Dump/BX-Book-Ratings.csv')
    print(BookCrossing)
    print(f'BookCrossing[0][1] = {BookCrossing[0][1]}')
    BookCrossing.get_user_id(140)
    BookCrossing.get_item_id(750)

    EachMovie = EachMovieDataSet('./dataset/rec-eachmovie/rec-eachmovie.edges')
    print(EachMovie)
    print(f'EachMovie[0][0] = {EachMovie[0][0]}')
    EachMovie.get_user_id(14553)
    EachMovie.get_item_id(120)

    # DataLoader = RSDataLoader(MovieLens)

    # for i in DataLoader:
    #     print(i[0][0])
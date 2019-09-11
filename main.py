from calldata.call_dataset import *

if __name__ == '__main__':
    
    MovieLens = MovieLensDataSet('./dataset/ml-100k/u.data')
    print(MovieLens)
    print(f'MovieLens[195][241] = {MovieLens[195][241]}')

    Jester = JesterDataSet('./dataset/jester-data-1/jester-data-1.xls')
    print(Jester)
    print(f'Jester[0][0] = {Jester[0][0]}')
    print(f'user index = {333}, actual user id = {Jester.get_user_id(333)}')
    print(f'item index = {45}, actual item id = {Jester.get_item_id(45)}')

    BookCrossing = BookCrossingDataSet('./dataset/BX-CSV-Dump/BX-Book-Ratings.csv')
    print(BookCrossing)
    print(f'BookCrossing[0][1] = {BookCrossing[0][1]}')
    print(f'user index = {140}, actual user id = {BookCrossing.get_user_id(140)}')
    print(f'item index = {985}, actual item id = {BookCrossing.get_item_id(985)}')

    EachMovie = EachMovieDataSet('./dataset/rec-eachmovie/rec-eachmovie.edges')
    print(EachMovie)
    print(f'EachMovie[0][0] = {EachMovie[0][0]}')
    print(f'user index = {1234}, actual user id = {EachMovie.get_user_id(1234)}')
    print(f'item index = {123}, actual item id = {EachMovie.get_item_id(123)}')


    dl = RSDataLoader(MovieLens)

    for data in dl:
        pass
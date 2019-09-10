from calldata import call_dataset

if __name__ == '__main__':
    
    MovieLens = call_dataset.MovieLensDataSet('./dataset/ml-100k/u.data')
    print(MovieLens[307][0])
   
    a = call_dataset.RSDataLoader(MovieLens, 3)


    for i in a:
        print(i)
import pandas as pd
import numpy as np
import pandas as pd



for i in range (1,2):
    print(f"currently in fold {i}")
    train = 'http://files.grouplens.org/datasets/movielens/ml-100k/u{}.base'.format(i)
    fold_data = pd.read_csv(train, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

    test = 'http://files.grouplens.org/datasets/movielens/ml-100k/u{}.test'.format(i)
    test_data = pd.read_csv(test, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    fn1(fold_data, test_data)

    
    def fn1(fold_data, test_data):
    user_item_matrix = np.zeros((1000, 1700))

    for index, row in fold_data.iterrows():
        m = row['user_id'] - 1 # subtract 1 to adjust for 0-based indexing
        n = row['item_id'] - 1 # subtract 1 to adjust for 0-based indexing
        user_item_matrix[m][n] = row['rating'] # assign the rating to the corresponding matrix element

    #make R matrix 
    R = np.where(user_item_matrix != 0, 1, 0) # create R matrix where 1 corresponds to non-zero elements in user_item_matrix and 0 corresponds to zero elements

    R.shape
    fn(R,user_item_matrix,test_data)

# make U and V
def fn(R, user_item_matrix,test_data):
    import matplotlib.pyplot as plt


    nmae_arr=[]
    arr= [a for a in range(75,101)]

    for z in arr:
        m=1000
        n=1700
        f =5
        lmbda=0.5
        U = np.random.rand(m, f)*lmbda
        U
        V=np.random.rand(f,n)*lmbda

        X=U@V
        print(f"current iteration is {z}")
        curr=z
        for i in range (curr):

          B=X+user_item_matrix-R*X
          for l in range(15):
            # Fix U and solve for Vfor i in range(num_iterations):
            # Update U using current V
                # Update U using current V
            U = U * np.divide(np.dot(B, V.T), np.dot(np.dot(U, V), V.T))

            # Update V using current U
            V = V * np.divide(np.dot(U.T, B), np.dot(np.dot(U.T, U), V))
          X=U@V

        # print(X)

        nmae=0
        for r in range (20000):
          nmae += abs(X[test_data.iloc[r,0]-1][test_data.iloc[r, 1]-1] - test_data.iloc[r,2])

        nmae = nmae/80000
        print(f"value of nmae is {nmae}")
        nmae_arr.append(nmae)
    x = np.arange(len(nmae_arr))
    plt.plot(x, nmae_arr)

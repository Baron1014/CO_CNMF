import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model
import cv2
import datetime 

def main(kronecker_product):
    data = scipy.io.loadmat("data.mat")
    print(data.keys())
    print(f"Ym : {data['Ym'].shape}")
    print(f"Yh : {data['Yh'].shape}")
    print(f"N : {data['N'].shape}")
    print(f"I_REF : {data['I_REF'].shape}")
    print(f"D : {data['D'].shape}")
    print(f"symvar : {data['symvar'].shape}")
    print(data["N"])
    print(data["symvar"])

    # plt raw data
    plot(data["I_REF"], "Target")
    plot(data["Yh"], "Yh")
    plot(data["Ym"], "Ym", [2, 1, 0])

    # reduce data size(init)
    small_ratio = 10
    Ym_small = data["Ym"][:21, :21, :]
    Yh_small = data["Yh"][:3, :3, :]
    print(f"Ym_small: {Ym_small.shape}, Yh_small: {Yh_small.shape}")
    
    # Get L and M of Ym and Yh
    Ym_L,Ym_M = (Ym_small.shape[0])**2, Ym_small.shape[2]
    Yh_L,Yh_M = (Yh_small.shape[0])**2, Yh_small.shape[2]

    # get original spatial of Ym 
    Origin_YmL = (data['Ym'].shape[0])**2

    # init A, S, B, N
    N = data['N'][0,0]
    #N = 3
    A = get_gaussian_array(Yh_M, N)
    S = get_gaussian_array(N, Ym_L)
    B = get_gaussian_array(Ym_L, Yh_L)
    print(A.shape, S.shape)

    # identity matrix
    I_L = np.identity(Ym_L)
    I_M = np.identity(Yh_M)

    # The width and height of the small figure
    small_yh_size, small_ym_size = int(data['Yh'].shape[0]/small_ratio), int(data['Ym'].shape[0]/small_ratio) 

    epochs = 10
    for epoch in range(epochs): 
        start = datetime.datetime.now()
        # init mean score
        lasso_mean = list()
        ridge_mean = list()
        # Generate original image using small image
        for i in range(small_ratio):
            print(f"Start scan {i} row parts")
            for j in range(small_ratio):
                print(f"Start scan {i} row {j} col")
                Ym_small = data["Ym"][i*small_ym_size:(i+1)*small_ym_size, j*small_ym_size:(j+1)*small_ym_size, :]
                Yh_small = data["Yh"][i*small_yh_size:(i+1)*small_yh_size, j*small_yh_size:(j+1)*small_yh_size, :]

                # vec(Array)
                vec_s = S.flatten('F').reshape(-1, 1)
                vec_a = A.flatten('F').reshape(-1, 1)
                print(f"vec_s: {vec_s.shape}, vec_a: {vec_a.shape}")

                # get C
                D = data['D']
                print("start compute C1 ...")
                C1 = list()
                C1.append(kronecker_product(B.T, A).T)
                C1.append(kronecker_product(I_L, np.dot(D,A)))
                print(f"C1 = [{C1[0].shape}, {C1[1].shape} ]")
                print("start compute C2 ...")
                C2 = list()
                C2.append(kronecker_product(np.dot(S,B).T, I_M))
                C2.append(kronecker_product(S.T, D))
                print(f"C2 = [{C2[0].shape}, {C2[1].shape} ]")

                # calculate lasso regression and ridge regression
                lasso = linear_model.Lasso(alpha=0.1, max_iter=1)
                train_X = np.concatenate((np.dot(C1[0].T, vec_s), np.dot(C1[1], vec_s)), axis=0)
                train_Y = np.concatenate((Yh_small.reshape(-1,1), Ym_small.reshape(-1,1)), axis=0)
                lasso.fit(train_X, train_Y)
                update_S = lasso.predict(S.reshape(-1, 1)).reshape(N, Ym_L)

                ridge = linear_model.Ridge(alpha=0.1, max_iter=1)
                train_X = np.concatenate((np.dot(C2[0], vec_a), np.dot(C2[1], vec_a)), axis=0)
                train_Y = np.concatenate((Yh_small.reshape(-1,1), Ym_small.reshape(-1,1)), axis=0)
                ridge.fit(train_X, train_Y)
                update_A = ridge.predict(A.reshape(-1, 1)).reshape(Yh_M, N)

                # evaluation
                lasso_score = lasso.score(train_X, train_Y)
                ridge_score = ridge.score(train_X, train_Y)
                print(f"Lasso score:{lasso_score}, Ridge score:{ridge_score}")
                lasso_mean.append(lasso_score)
                ridge_mean.append(ridge_score)

                # generate small_z image
                new_Z = np.dot(A, S).T
                if j==0:
                    row_z = new_Z
                else:
                    row_z = np.concatenate((row_z, new_Z), axis=0)
                
                # Accumulate S and A
                if j==0 and i==0:
                    sum_S = update_S
                    sum_A = update_A
                else:
                    sum_A += update_A
                    sum_S += update_S

                print(f"row_z: {row_z.shape}")
            # accumulate each row_z
            if i==0:
                img_Z = row_z
            else:
                img_Z = np.concatenate((img_Z, row_z), axis=0)
            print(f"img_z: {img_Z.shape}")

        # update A and Ｓ
        S = sum_S / (i+1)*(j+1)
        A = sum_A / (i+1)*(j+1)
        
        img_Z = img_Z.reshape(int(Origin_YmL**0.5), int(Origin_YmL**0.5), Yh_M)
        plot(img_Z, f"result_{epoch}_iter")

        end = datetime.datetime.now()
        print(f"Total Lasso score:{sum(lasso_mean)/len(lasso_mean)}, Total Ridge score:{sum(ridge_mean)/len(ridge_mean)}")
        print(f"{epoch} iter cost time: {end-start}")
        

def get_gaussian_array(x1, x2):
    return np.random.normal(loc=1, scale=2, size=(x1, x2))

def plot(data, file_name, RGB_channel=[61, 25, 13]):
    img = Image.fromarray(np.uint8(data[:, :, RGB_channel] * 255))
    img.save(f"reports/{file_name}.jpg")

def kronecker_product(arr1, arr2):
    arr1_w, arr1_col = arr1.shape[0], arr1.shape[1]
    arr2_w, arr2_col = arr2.shape[0], arr2.shape[1]

    for i in tqdm(range(arr1_w), desc="kronecker product"):
        # the matrix of multiplying column i of arr1 by arr2
        con_list = list()
        for c in range(arr1_col):
            # store the result
            subblock = np.zeros((arr2_w, arr2_col))
            for ii in range(arr2_w):
                for cc in range(arr2_col):
                    subblock[ii, cc] = arr1[i,c] * arr2[ii, cc]

            # concatenate each subblock into the set of row i of arr
            if c==0:
                arr1_row_block = subblock
            else:
                arr1_row_block = np.concatenate((arr1_row_block, subblock), axis=1)

        # concatenate the set of every i row into the final output
        if i==0:
            kron_array = arr1_row_block
        else:
            kron_array = np.concatenate((kron_array, arr1_row_block), axis=0)

    return kron_array


if __name__ == "__main__":
    # Comparing the speed of my own method and numpy
    #main(kronecker_product = lambda x, y: np.kron(x,y))
    main(kronecker_product = lambda x, y: kronecker_product(x,y))

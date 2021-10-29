import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import linear_model
import cv2
import datetime 
import wandb

def main(kronecker_product):
    # init wandb run
    run = wandb.init(project='CO_CNMF',
                        entity='Baron'
                        )
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

    # 畫raw data
    plot(data["I_REF"], "Target")
    plot(data["Yh"], "Yh")
    plot(data["Ym"], "Ym", [2, 1, 0])

    # 取得Ym及Yh的L及M
    Ym_L,Ym_M = (data["Ym"].shape[0])**2, data["Ym"].shape[2]
    Yh_L,Yh_M = (data["Yh"].shape[0])**2, data["Yh"].shape[2]

    # init A, S, B, N
    #N = data['N'][0,0]
    N = 3
    A = get_gaussian_array(Yh_M, N)
    S = get_gaussian_array(N, Ym_L)
    B = get_gaussian_array(Ym_L, Yh_L)
    print(A.shape, S.shape)

    # identity matrix
    I_L = np.identity(Ym_L)
    I_M = np.identity(Yh_M)

    # epochs = 10
    epochs = 10
    for epoch in range(epochs): 
        start = datetime.datetime.now()
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

        # 計算lasso及ridge regression
        lasso = linear_model.Lasso(alpha=0.1, max_iter=1)
        train_X = np.concatenate((np.dot(C1[0], vec_s), np.dot(C1[1], vec_s)), axis=0)
        train_Y = np.concatenate((data['Yh'].reshape(-1,1), data['Ym'].reshape(-1,1)), axis=0)
        lasso.fit(train_X, train_Y)
        S = lasso.predict(S.reshape(-1, 1)).reshape(data['N'][0,0], Ym_L)

        ridge = linear_model.Ridge(alpha=0.1, max_iter=1)
        train_X = np.concatenate((np.dot(C2[0], vec_a), np.dot(C2[1], vec_a)), axis=0)
        train_Y = np.concatenate((data['Yh'].reshape(-1,1), data['Ym'].reshape(-1,1)), axis=0)
        ridge.fit(train_X, train_Y)
        A = ridge.predict(S.reshape(-1, 1)).reshape(Yh_M, data['N'][0,0])

        new_Z = np.dot(A, S)
        plot(new_Z, f"result_{epoch}_iter")

        end = datetime.datetime.now()
        print(f"Lasso score:{lasso.score(train_X, train_Y)}, Ridge score:{ridge.score(train_X, train_Y)}")
        print(f"{epoch} iter cost time: {end-start}")
        wandb.summary['Lasso score'] = lasso.score(train_X, train_Y) 
        wandb.summary['Ridge score'] = ridge.score(train_X, train_Y) 


def get_gaussian_array(x1, x2):
    return np.random.normal(loc=1, scale=2, size=(x1, x2))

def plot(data, file_name, RGB_channel=[61, 25, 13]):
    img = Image.fromarray(np.uint8(data[:, :, RGB_channel] * 255))
    img.save(f"reports/{file_name}.jpg")

def kronecker_product(arr1, arr2):
    arr1_w, arr1_col = arr1.shape[0], arr1.shape[1]
    arr2_w, arr2_col = arr2.shape[0], arr2.shape[1]

    for i in tqdm(range(arr1_w), desc="kronecker product"):
        # arr1第i列乘完arr2的矩陣
        con_list = list()
        for c in range(arr1_col):
            # 儲存跟arr1做完運算的結果
            subblock = np.zeros((arr2_w, arr2_col))
            for ii in range(arr2_w):
                for cc in range(arr2_col):
                    subblock[ii, cc] = arr1[i,c] * arr2[ii, cc]

            # 連接每一個子區塊, 成為arr第i行的集合
            if c==0:
                arr1_row_block = subblock
            else:
                arr1_row_block = np.concatenate((arr1_row_block, subblock), axis=1)

        # 連接每i行的集合, 成為最後輸出
        if i==0:
            kron_array = arr1_row_block
        else:
            kron_array = np.concatenate((kron_array, arr1_row_block), axis=0)

    return kron_array


if __name__ == "__main__":
    # 比較自己kron跟numpy的速度
    #main(kronecker_product = lambda x, y: np.kron(x,y))
    main(kronecker_product = lambda x, y: kronecker_product(x,y))

    #arr1 = np.array([
        #[1,2,3,4],
        #[2,3,4,5],
        #[3,4,5,6]
    #])
    #arr2 = np.array([
        #[1,2,3,4],
        #[1,2,3,4],
        #[1,2,3,4]
    #])
    #arr3 = arr1 - arr2

    #lasso = linear_model.Lasso(alpha=0.1, max_iter=1)
    #lasso.fit(arr1.reshape(-1 ,1), arr2.reshape(-1, 1))
    #score = lasso.score(arr1.reshape(-1 ,1), arr2.reshape(-1, 1))
    #predict = lasso.predict(arr1.reshape(-1 ,1))
    #print(arr1.reshape(1,-1))
    #print(arr2.reshape(1,-1))
    #print(predict)
    #print(lasso.n_iter_)





        
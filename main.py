import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

def main():
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

    # init A, S
    A = get_gaussian_array(Yh_M, data['N'][0,0])
    S = get_gaussian_array(data['N'][0,0], Ym_L)

    print(A.shape, S.shape)

    # get C
    D = data['D']
    C1 = np.dot(D,A)
    C2 = kronecker_product(S.T, D)
    print(C2.shape)

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
    arr1 = np.array([
        [1,2,3], 
        [4,5,6] 
    ])

    arr2 = np.array([
        [2,3,4,5], 
        [2,3,4,5] 
    ])
    output = kronecker_product(arr1, arr2)
    print(output.shape)
    print(output)

    main()


        
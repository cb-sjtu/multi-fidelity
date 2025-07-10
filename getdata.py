import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def get_data():
    kzs = np.load('data_gen/data_kzs87.npy').astype(np.float32).reshape((5, 1, 87))
    uum = np.load('data_gen/data_uvm.npy').astype(np.float32)[:,:,::2]
    uum_1d= np.load('data_gen/uv_1d.npy').astype(np.float32)#低精度
    # 高精度数据：只取 [0,1,3,4] 为训练集，reshape 成 [4, 200*87]
    uum = kzs * uum
    uum_new = uum[[0, 1, 3, 4], 0:200].reshape((4, -1))
    uum_new_test = uum[[2], 0:200].reshape((1, -1))
    # 低精度数据：只取 [0,1,3,4] 为训练集，reshape 成 [4, 200]
    uum_1d_new = uum_1d[[0, 1, 3, 4], :200].astype(np.float32)
    uum_1d_new_test = uum_1d[[2], :200].astype(np.float32)
    trunk_out = np.load('data_gen/data_motai_3d_mix.npy')[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((8, 200, -1))
    trunk_out_test = np.load('data_gen/data_motai_3d_mix.npy')[[2, 7]].reshape((2, 200, -1))
    motai = trunk_out.transpose(0, 2, 1)
    motai_test = trunk_out_test.transpose(0, 2, 1)
    branch_in = np.concatenate((trunk_out.real[..., np.newaxis], trunk_out.imag[..., np.newaxis]), axis=-1).astype(np.float32).reshape((8, 200, -1))
    branch_in_test = np.concatenate((trunk_out_test.real[..., np.newaxis], trunk_out_test.imag[..., np.newaxis]), axis=-1).astype(np.float32).reshape((2, 200, -1))
    
    for i in range(8):
        scaler_Euuc = StandardScaler().fit(branch_in[i])
        std_Euuc = np.sqrt(scaler_Euuc.var_.astype(np.float32))
        branch_in[i] = (branch_in[i] - scaler_Euuc.mean_.astype(np.float32)) / std_Euuc

    for i in range(2):  
        scaler_Euuc_test = StandardScaler().fit(branch_in_test[i])
        std_Euuc_test = np.sqrt(scaler_Euuc_test.var_.astype(np.float32))
        branch_in_test[i] = (branch_in_test[i] - scaler_Euuc_test.mean_.astype(np.float32)) / std_Euuc_test

    kzs_s = kzs[[0, 1, 3, 4]]
    kzs_s[0] = 2 * math.pi / kzs_s[0] * 182.088
    kzs_s[1] = 2 * math.pi / kzs_s[1] * 543.496
    kzs_s[2] = 2 * math.pi / kzs_s[2] * 1000.512
    kzs_s[3] = 2 * math.pi / kzs_s[3] * 1994.756
    kzs_s_test = kzs[[2]]
    kzs_s_test = 2 * math.pi / kzs_s_test * 1000.512

    dcPs_s = np.load('data_gen/data_dcp23_mix.npy').astype(np.float32)[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((-1,  23))
    dcPs_s_test = np.load('data_gen/data_dcp23_mix.npy').astype(np.float32)[[2, 7]].reshape((-1,  23))
    dkxs_s = np.load('data_gen/data_dkx86_mix.npy').astype(np.float32)[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((-1, 86))
    dkxs_s_test = np.load('data_gen/data_dkx86_mix.npy').astype(np.float32)[[2,7]].reshape((-1, 86))
 
    y = np.load('data_gen/data_yy.npy')
    # y=1-y
    real_2d_y = y
    real_2d_x = np.zeros((5, 87))
    real_2d_x[[0, 1, 3, 4]] = kzs_s.reshape((4, 87))
    real_2d_x[2] = kzs_s_test.reshape((1, 87))
    
    y[0] = y[0] * 182.088
    y[1] = y[1] * 543.496
    y[2] = y[2] * 1000.512
    y[3] = y[3] * 1994.756
    y[4] = y[4] * 5185.897
    y = y[[0, 1, 3, 4]]
    y_test = y[[2]]

    real_2d = np.zeros((5, 100, 87, 2))

    for i in range(5):
        real_2d_y_i = real_2d_y[i]
        repeats = 100 // real_2d_y_i.shape[0] + 1
        real_2d_y_i = np.tile(real_2d_y_i, repeats)[:100]
        for j in range(100):
            real_2d[i, j, :, 0] = real_2d_x[i]
            real_2d[i, j, :, 1] = real_2d_y_i[j]

    coodinates = np.load('data_gen/3d_coordinates_mix.npy').reshape((10, -1, 3)).astype(np.float32)
    #正则化
    for i in range(10):
        scaler_coordinates = StandardScaler().fit(coodinates[i])
        std_coordinates = np.sqrt(scaler_coordinates.var_.astype(np.float32))
        coodinates[i] = (coodinates[i] - scaler_coordinates.mean_.astype(np.float32)) / std_coordinates
    print("coodinates max:", np.max(coodinates))
    print("coodinates min:", np.min(coodinates))
    coodinates_zz = coodinates[[0, 1, 3, 4, 5, 6, 8, 9]]
    coodinates_zz_test = coodinates[[2,7]]
    
    return uum_new, uum_new_test, branch_in, branch_in_test, \
        dcPs_s, dcPs_s_test, dkxs_s, dkxs_s_test, kzs_s, kzs_s_test, \
        y, y_test, real_2d, kzs, motai, motai_test, \
        coodinates_zz, coodinates_zz_test, uum_1d_new, uum_1d_new_test

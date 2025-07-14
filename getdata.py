import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def get_data():
    kzs = np.load('data_gen/data_kzs87.npy').astype(np.float32).reshape((5, 1, 87))
    uum = np.load('data_gen/data_uvm.npy').astype(np.float32)[:, :, ::2]
    uum_1d = np.load('data_gen/uv_1d.npy').astype(np.float32)  # 低精度

    # 高精度数据：只取 [0,1,3,4] 为训练集，reshape 成 [4, 200*87]
    uum = kzs * uum
    uum_new = uum[[0, 1, 3, 4], 0:200].reshape((4, -1))
    uum_new_test = uum[[2], 0:200].reshape((1, -1))

    # 低精度数据：只取 [0,1,3,4] 为训练集，reshape 成 [4, 200]
    uum_1d_new = uum_1d[[0, 1, 3, 4], :200].astype(np.float32)
    uum_1d_new_test = uum_1d[[2], :200].astype(np.float32)

    trunk_out = np.load('data_gen/data_motai_3d_mix.npy')[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((8, 200, -1))
    trunk_out_test = np.load('data_gen/data_motai_3d_mix.npy')[[2, 7]].reshape((2, 200, -1))
    trunk_out_v = np.load('data_gen/data_motai_3d_mix_v.npy')[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((8, 200, -1))
    trunk_out_v_test = np.load('data_gen/data_motai_3d_mix_v.npy')[[2, 7]].reshape((2, 200, -1))

    # 对 trunk_out 和 trunk_out_v 的实部和虚部分别进行标准化
    for arr in [trunk_out, trunk_out_v]:
        for i in range(arr.shape[0]):
            # 1. 实部标准化
            data_real = arr[i].real
            scaler_real = StandardScaler()
            scaler_real.fit(data_real)
            std_real = np.sqrt(scaler_real.var_.astype(np.float32))
            data_real -= scaler_real.mean_.astype(np.float32)
            data_real /= std_real
            arr[i].real[:] = data_real  # in-place 写回
            
            # 2. 虚部标准化
            data_imag = arr[i].imag
            scaler_imag = StandardScaler()
            scaler_imag.fit(data_imag)
            std_imag = np.sqrt(scaler_imag.var_.astype(np.float32))
            data_imag -= scaler_imag.mean_.astype(np.float32)
            data_imag /= std_imag
            arr[i].imag[:] = data_imag  # in-place 写回

    # 对 trunk_out_test 和 trunk_out_v_test 的实部和虚部分别进行标准化
    for arr in [trunk_out_test, trunk_out_v_test]:
        for i in range(arr.shape[0]):
            # 1. 实部标准化
            data_real = arr[i].real
            scaler_real = StandardScaler()
            scaler_real.fit(data_real)
            std_real = np.sqrt(scaler_real.var_.astype(np.float32))
            data_real -= scaler_real.mean_.astype(np.float32)
            data_real /= std_real
            arr[i].real[:] = data_real

            # 2. 虚部标准化
            data_imag = arr[i].imag
            scaler_imag = StandardScaler()
            scaler_imag.fit(data_imag)
            std_imag = np.sqrt(scaler_imag.var_.astype(np.float32))
            data_imag -= scaler_imag.mean_.astype(np.float32)
            data_imag /= std_imag
            arr[i].imag[:] = data_imag


    # 把两个数据集在一个新的 axis 上合并
    motai = np.stack((trunk_out, trunk_out_v), axis=-1)
    motai_test = np.stack((trunk_out_test, trunk_out_v_test), axis=-1)

    motai = motai.transpose(0, 2, 1, 3)# 8，-1，200， 2
    motai_test = motai_test.transpose(0, 2, 1, 3)# 2，-1，200， 2

    branch_in = np.real(trunk_out)
    branch_in_test = np.real(trunk_out_test)


    kzs_s = kzs[[0, 1, 3, 4]].reshape((-1, 87, 1))
    kzs_s_test = kzs[[2]].reshape((-1, 87, 1))

    kzs_s[0] = 2 * math.pi / kzs_s[0] * 182.088
    kzs_s[1] = 2 * math.pi / kzs_s[1] * 543.496
    kzs_s[2] = 2 * math.pi / kzs_s[2] * 1000.512
    kzs_s[3] = 2 * math.pi / kzs_s[3] * 1994.756
    kzs_s_test = 2 * math.pi / kzs_s_test * 1000.512

    dcPs_s = np.load('data_gen/data_dcp23_mix.npy').astype(np.float32)[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((-1, 23))
    dcPs_s_test = np.load('data_gen/data_dcp23_mix.npy').astype(np.float32)[[2, 7]].reshape((-1, 23))
    dkxs_s = np.load('data_gen/data_dkx86_mix.npy').astype(np.float32)[[0, 1, 3, 4, 5, 6, 8, 9]].reshape((-1, 86))
    dkxs_s_test = np.load('data_gen/data_dkx86_mix.npy').astype(np.float32)[[2, 7]].reshape((-1, 86))

    y = np.load('data_gen/data_yy.npy')
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
    for i in range(10):
        scaler_coordinates = StandardScaler().fit(coodinates[i])
        std_coordinates = np.sqrt(scaler_coordinates.var_.astype(np.float32))
        coodinates[i] = (coodinates[i] - scaler_coordinates.mean_.astype(np.float32)) / std_coordinates

    coodinates_zz = coodinates[[0, 1, 3, 4, 5, 6, 8, 9]]
    coodinates_zz_test = coodinates[[2, 7]]

    # 重新组织数据
    trunk_out_input = (branch_in.transpose(0, 2, 1).reshape((-1, 87, 87, 4800)), coodinates_zz, motai, dcPs_s, dkxs_s)
    trunk_out_input_test = (branch_in_test.transpose(0, 2, 1).reshape((-1, 87, 87, 4800)), coodinates_zz_test, motai_test, dcPs_s_test, dkxs_s_test)

    uum_new = np.reshape(uum_new, (-1, 200, 87)).transpose(0, 2, 1).reshape((-1, 200 * 87))
    uum_new_test = np.reshape(uum_new_test, (-1, 200, 87)).transpose(0, 2, 1).reshape((-1, 200 * 87))
    uum_new = (uum_new, uum_1d_new)
    uum_new_test = (uum_new_test, uum_1d_new_test)

    return uum_new, uum_new_test, trunk_out_input, trunk_out_input_test, \
       dkxs_s, dkxs_s_test, kzs_s, kzs_s_test, \
        y, y_test, real_2d

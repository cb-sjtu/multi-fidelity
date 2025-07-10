import numpy as np
from plot_results import plot_and_save  # 导入绘图函数

def save_results(dataframe, real_2d, label, uum, error, yy, dkxs_s,kzs_s, mode="train"):
    """
    保存训练集或测试集的结果，包括2D能谱和1D积分结果。

    Args:
        dataframe (str): 保存结果的文件夹路径。
        real_2d (np.ndarray): 2D坐标数据。
        label (np.ndarray): 模型预测结果。
        uum (np.ndarray): 真实值。
        error (np.ndarray): 误差值（可选）。
        yy (np.ndarray): y坐标数据。
        dkxs_s (np.ndarray): dkx积分权重。
        mode (str): 模式，"train" 或 "test"。
    """
    # 处理2D能谱结果
    label = np.reshape(label, (-1, 87 * 100)).T
    uum = np.reshape(uum, (-1, 87 * 100)).T
    if error is None:
        error = np.abs(label - uum)

    for i in range(label.shape[1]):
        result = np.hstack((
            real_2d[i].reshape((100 * 87, 2)),
            label[:100 * 87, i].reshape((100 * 87, 1)),
            uum[:100 * 87, i].reshape((100 * 87, 1)),
            error[:100 * 87, i].reshape((100 * 87, 1))
        ))
        filename = f"{dataframe}/real-Euukc_new_6_{mode}_{i}.dat"
        np.savetxt(filename, result)
        with open(filename, 'r') as file:
            filedata = file.read()
        newline = 'VARIABLES="x","y","u","u_p","error"\nZONE I=100,J=87\n'
        filedata = newline + filedata
        with open(filename, 'w') as file:
            file.write(filedata)

    # 处理1D积分结果
    label = np.reshape(label.T, (-1, 87, 100))
    uum = np.reshape(uum.T, (-1, 87, 100))
  
    dkxs_s = dkxs_s.reshape((-1, 1, 86))

    label = label / kzs_s
    uum = uum / kzs_s

    label_1 = label[:, :-1, :]
    label_2 = label[:, 1:, :]
    label_sum = (label_1 + label_2).transpose(0, 2, 1)
    label_profile = (label_sum * dkxs_s * 0.5).sum(axis=2).T

    uum_1 = uum[:, :-1, :]
    uum_2 = uum[:, 1:, :]
    uum_sum = (uum_1 + uum_2).transpose(0, 2, 1)
    uum_profile = (uum_sum * dkxs_s * 0.5).sum(axis=2).T

    yy = yy.T
    for i in range(label_profile.shape[1]):
        result = np.hstack((
            yy[:100, i].reshape((100, 1)),
            label_profile[:100, i].reshape((100, 1)),
            uum_profile[:100, i].reshape((100, 1))
        ))
        filename = f"{dataframe}/2loss1_7_{mode}{i}.dat"
        np.savetxt(filename, result)

    # 调用绘图函数
    plot_and_save(dataframe, mode=mode)

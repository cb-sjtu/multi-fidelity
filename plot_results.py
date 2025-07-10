import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter, LogLocator

def plot_and_save(data_folder, mode="train"):
    """
    读取2loss1_7_{mode}{i}.dat文件，绘制并保存图片。

    Args:
        data_folder (str): 数据文件夹路径。
        mode (str): 模式，"train" 或 "test"。
    """
    files = [f for f in os.listdir(data_folder) if f.startswith(f"2loss1_7_{mode}") and f.endswith(".dat")]
    for file in files:
        data = np.loadtxt(os.path.join(data_folder, file))
        y_plus = data[:, 0]  # 第一列为坐标 y^+
        predicted = data[:, 1]  # 第二列为预测值
        ground_truth = data[:, 2]  # 第三列为真实值

        plt.figure()
        plt.plot(np.log10(y_plus), predicted, label="Predicted", color="blue")  # 对y^+取log10
        plt.plot(np.log10(y_plus), ground_truth, label="Ground Truth", color="red", linestyle="--")  # 对y^+取log10
        plt.xlabel("log(y⁺)")
        plt.ylabel("E_uu")
        plt.title(file)
        plt.legend()
        plt.grid(True)


        # 保存图片
        output_file = os.path.join(data_folder, file.replace(".dat", ".png"))
        plt.savefig(output_file)
        plt.close()

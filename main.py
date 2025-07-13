import third_party.deepxde as dde
print(dde.__file__)
import numpy as np

from sklearn.preprocessing import StandardScaler
from third_party.deepxde.nn.tensorflow_compat_v1.resolvent_deeponet import DeepONet_resolvent_3d_mix
from getdata import get_data
from network import network
from postprocess import save_results
from config import CONFIG  # 导入配置文件
import os
# 尝试从系统继承代理设置

import wandb  # 导入wandb库
wandb.init(mode="disabled")
# 如果需要使用代理，可以在这里设置
from wandb_callback import WandbCallback  # 导入 WandbCallback

def main():
    # 初始化wandb
    wandb.init(project=CONFIG["project_name"], name=CONFIG["run_name"], config=CONFIG)
    config = wandb.config

    dataframe = "DeepONet_resolvent_3d"
    uum, uum_test, trunk_out_input, trunk_out_input_test, \
    dkxs_s, dkxs_s_test, \
    kzs_s, kzs_s_test, yy, yy_test, real_2d= get_data()

    problem = "flow"
    N_points = 87 * 24
    data = dde.data.Sixthple(trunk_out_input, uum, trunk_out_input_test, uum_test)
    m = 100 * 87 * 87 * 24 * 2
    activation = ["relu", "relu", "relu"]
    initializer = config.initializer

    branch_net, trunk_net_1, trunk_net_2, dot = network(problem, m, N_points)

    net = DeepONet_resolvent_3d_mix(
        branch_net,
        trunk_net_1,
        trunk_net_2,
        dot,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        kernel_initializer=initializer,
        regularization=None,
 
    )

    # scaler = StandardScaler().fit(uum)
    # std = np.sqrt(scaler.var_.astype(np.float32))

    # def output_transform(outputs):
    #     return outputs * std + scaler.mean_.astype(np.float32)

    # net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=config.learning_rate,
        loss=config.loss,
        metrics=["l2 relative error"],
        decay=(config.decay_method, config.decay_step, config.decay_rate),
    )
    checker = dde.callbacks.ModelCheckpoint(
        dataframe + "/model.ckpt", save_better_only=False, period=config.save_period
    )

    # 开始训练
    losshistory, train_state = model.train(
        epochs=config.epochs,
        batch_size=config.batch_size,
        display_every=5,
        callbacks=[checker, WandbCallback(display_every=5)],
        model_save_path=dataframe
    )
    dde.saveplot(losshistory, train_state, issave=True, isplot=True, loss_fname=dataframe + "/loss.dat")

    # 后处理
    model.restore(dataframe + "/model.ckpt-2000.ckpt")
    label_train = model.predict(trunk_out_input)
    save_results(dataframe, real_2d, label_train, uum, None, yy, dkxs_s, kzs_s, mode="train")

    label_test = model.predict(trunk_out_input_test)
    save_results(dataframe, real_2d, label_test, uum_test, None, yy_test, dkxs_s_test, kzs_s_test, mode="test")

if __name__ == "__main__":
    main()

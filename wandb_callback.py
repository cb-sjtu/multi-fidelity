import wandb
import deepxde as dde

class WandbCallback(dde.callbacks.Callback):
    def __init__(self, display_every):
        self.display_every = display_every
        self.model = None  # 初始化模型属性

    def set_model(self, model):
        """设置模型对象"""
        self.model = model

    def on_epoch_end(self):
        """在每个 epoch 结束时记录日志"""
        step = self.model.train_state.step
        epoch = self.model.train_state.epoch  # 计算当前 epoch

        # 从训练状态中提取损失、指标和学习率
        loss_train = self.model.train_state.loss_train  # 当前 epoch 的训练损失
        loss_test = self.model.train_state.loss_test  # 当前 epoch 的测试损失
        # l2_relative_error = self.model.train_state.metrics_test  # 当前 epoch 的指标

        # 记录到 wandb
        wandb.log({
            "epoch": epoch,
            "loss_train": loss_train,
            "loss_test": loss_test,
            # "l2_relative_error": l2_relative_error,
            # "learning_rate": learning_rate,  # 记录学习率
        })
from . import backend as bkd
from . import config
from .backend import tf
import numpy as np
from scipy import io

def custom_mixed_loss(y_true, y_pred):
    y_true_high, y_true_low = y_true  # unpack
    y_pred_high, y_pred_low = y_pred

    loss_high = tf.reduce_mean(tf.square(y_true_high - y_pred_high))
    loss_low = tf.reduce_mean(tf.square(y_true_low - y_pred_low))

    # 权重可以调
    return loss_high + 0.5 * loss_low



def mean_absolute_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)


def mean_squared_error(y_true, y_pred):
    # Warning:
    # - Do not use ``tf.losses.mean_squared_error``, which casts `y_true` and `y_pred` to ``float32``.
    # - Do not use ``tf.keras.losses.MSE``, which computes the mean value over the last dimension.
    # - Do not use ``tf.keras.losses.MeanSquaredError()``, which casts loss to ``float32``
    #     when calling ``compute_weighted_loss()`` calling ``scale_losses_by_sample_weight()``,
    #     although it finally casts loss back to the original type.
    if isinstance(y_true, tuple):
        loss_1=bkd.reduce_mean(bkd.square(y_true[0] - y_pred[0]))
        loss_2=bkd.reduce_mean(bkd.square(y_true[1]*y_true[1] - y_pred[1]))
        print(loss_1,loss_2)
        loss=loss_1+loss_2
    else:
        loss = bkd.reduce_mean(bkd.square(y_true - y_pred))
    return loss



def mean_l2_relative_error(y_true, y_pred):
    

    return bkd.square(bkd.norm(y_true - y_pred) / bkd.norm(y_true))
    #return bkd.reduce_mean(bkd.norm(y_true - y_pred, axis=1) / bkd.norm(y_true, axis=1))

def smooth_logl2(y_true, y_pred):
    N=200
    n=173
    y=np.zeros((5,200))
    y[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re180.mat')['y'][:200][::-1].reshape((200,)))*182.088

    y[1]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re550.mat')['y'][:200][::-1].reshape((200,)))*543.496 
    y[2]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*1000.512
    y[3]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re2000.mat')['y'][:200][::-1].reshape((200,)))*1994.756 
    y[4]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re5200.mat')['y'][:200][::-1].reshape((200,)))*5185.897

    y_test=np.zeros((1,200))
    y_test[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*1000.512


    if y_true.shape[1]==100:
        y=y_test
    y_true=tf.reshape(y_true,(-1,n,N))
    if y_true.shape[0]==1:
        y=y_test
    y=y.astype(np.float32)
    y=tf.reshape(y,(-1,1,N))
    y_true=y_true/y
    y_true=tf.reshape(y_true,(-1,n*N))
    y_pred=tf.reshape(y_pred,(-1,n,N))
    y_pred=y_pred/y
    y_pred=tf.reshape(y_pred,(-1,n*N))
    
    
    loss1=bkd.square(bkd.norm(y_true - y_pred))
    loss2=bkd.norm(y_pred)
    loss=loss1+loss2/2000
    
    
    return loss

def def_l2_relative_error(y_true, y_pred):
    N=100
    n=87
    # y=np.zeros((5,N))
    # y[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re180.mat')['y'][:N][::-1].reshape((N,)))*182.088
    # y[1]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re550.mat')['y'][:N][::-1].reshape((N,)))*543.496 
    # y[2]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:N][::-1].reshape((N,)))*1000.512
    # y[3]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re2000.mat')['y'][:N][::-1].reshape((N,)))*1994.756 
    # y[4]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re5200.mat')['y'][:N][::-1].reshape((N,)))*5185.897

    # y_test=np.zeros((1,N))
    # y_test[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:N][::-1].reshape((N,)))*1000.512
    # y=y[[0,1,2,3,4]]
    y = np.load('data_gen/data_y100.npy')
    y=1-y
    #检测y里面所有的数，如果有0,就改为20
    y[y == 0] = 20
    y[0] = y[0] * 182.088
    y[1] = y[1] * 543.496
    y[2] = y[2] * 1000.512
    y[3] = y[3] * 1994.756
    y[4] = y[4] * 5185.897
    y = y[[0, 1, 3, 4]]
    y_test = y[[2]]
    

    y_true=tf.reshape(y_true,(-1,n,N))
    if y_true.shape[0]==1:
        y=y_test
    y=y.astype(np.float32)
    y=tf.reshape(y,(-1,1,N))
    y_true=y_true/y
    y_true=tf.reshape(y_true,(-1,n*N))
    y_pred=tf.reshape(y_pred,(-1,n,N))
    y_pred=y_pred/y
    y_pred=tf.reshape(y_pred,(-1,n*N))
    
    loss=bkd.square(bkd.norm(y_true - y_pred))
    
    
    return loss

def def_l2_error_chatgpt(y_true, y_pred):
    N=200
    n=173
    y=np.zeros((5,200))
    y[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re180.mat')['y'][:200][::-1].reshape((200,)))*182.088
    y[1]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re550.mat')['y'][:200][::-1].reshape((200,)))*543.496 
    y[2]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*1000.512
    y[3]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re2000.mat')['y'][:200][::-1].reshape((200,)))*1994.756 
    y[4]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re5200.mat')['y'][:200][::-1].reshape((200,)))*5185.897

    y_test=np.zeros((1,200))
    y_test[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*1000.512
    y=y[[0,1,2,3,4]]
    
    #
    y_true=tf.reshape(y_true,(-1,n,N))
    if y_true.shape[0]==1:
        y=y_test
    y=y.astype(np.float32)
    # y=np.reshape(y,(-1,1,N))


    log_y = np.log(y)
    weights = np.zeros_like(log_y)
    for i in range(5):
        dy_log = np.diff(log_y[i])
        # 使用梯形积分法，对中间点做均值延拓
        weights[i,1:-1] = (dy_log[:-1] + dy_log[1:]) / 2
        weights[i,0] = dy_log[0] / 2
        weights[i,-1] = dy_log[-1] / 2
    weights=np.reshape(weights,(-1,1,200))

    weights_tf = tf.convert_to_tensor(weights, dtype=tf.float32)  # shape: (4, 200)

    # loss 函数本体
    
    # 假设 batch size 是 4（与 y 对应）
    y_true=tf.reshape(y_true,(-1,n,N))
    y_pred=tf.reshape(y_pred,(-1,n,N))
    squared_diff = tf.square(y_true - y_pred)  # shape: (-1,173, 200)
    squared_true = tf.square(y_true)           # shape: (-1,173, 200)

    # 积分（加权）在 log y+ 空间
    weighted_diff = tf.multiply(squared_diff, weights_tf)  # shape: (-1,173, 200)
    weighted_true = tf.multiply(squared_true, weights_tf)  # shape: (-1,173, 200)

    weighted_true=tf.reshape(weighted_true,(-1,173*200))
    weighted_diff=tf.reshape(weighted_diff,(-1,173*200))
    # 计算损失


    loss = bkd.reduce_mean(bkd.square(weighted_true - weighted_diff))
       

    return loss
    



def def_mse(y_true, y_pred):
    N=200
    n=173
    y=np.zeros((5,200))
    y[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re180.mat')['y'][:200][::-1].reshape((200,)))*182.088

    y[1]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re550.mat')['y'][:200][::-1].reshape((200,)))*182.088
    y[2]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*182.088
    y[3]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re2000.mat')['y'][:200][::-1].reshape((200,)))*182.088 
    y[4]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re5200.mat')['y'][:200][::-1].reshape((200,)))*182.088

    y_test=np.zeros((1,200))
    y_test[0]=(io.loadmat('data_gen/y_jiami_data/N400/stat_Re1000.mat')['y'][:200][::-1].reshape((200,)))*1000.512

    # y_4=np.zeros((5,800))
    # y_4[:,0:200]=y
    # y_4[:,200:400]=y
    # y_4[:,400:600]=y
    # y_4[:,600:800]=y

    

    # y=np.zeros((4,100))
    # y[0]=(io.loadmat('data_1207/Euukc_Re180.mat')['y'][0:100].reshape(100))*182.088

    # y[1]=(io.loadmat('data_1207/Euukc_Re550.mat')['y'][0:100].reshape(100))*543.496
    # # y[2]=(io.loadmat('data_0605/database/stat_Re1000.mat')['y'].reshape((200,)))*1000.512
    # y[2]=(io.loadmat('data_1207/Euukc_Re2000.mat')['y'][0:100].reshape(100))*1994.756
    # y[3]=(io.loadmat('data_1207/Euukc_Re5200.mat')['y'][0:100].reshape(100))*5185.897

    # y_test=np.zeros((1,100))
    # y_test[0]=(io.loadmat('data_1207/Euukc_Re1000.mat')['y']['y'][0:100].reshape(100))*1000.512

    # y=np.load('data_gen/data_yy.npy')
    # y[0]=y[0]*182.088
    # y[1]=y[1]*543.496
    # y[2]=y[2]*1000.512
    # y[3]=y[3]*1994.756
    # y[4]=y[4]*5185.897
    # y=y[[0,1,3,4]]
    # y_test=y[[2]]
    if y_true.shape[1]==100:
        y=y_test
    y_true=tf.reshape(y_true,(-1,n,N))
    if y_true.shape[0]==1:
        y=y_test
    y=y.astype(np.float32)
    y=tf.reshape(y,(-1,1,N))
    y_true=y_true/y
    y_true=tf.reshape(y_true,(-1,n*N))
    y_pred=tf.reshape(y_pred,(-1,n,N))
    y_pred=y_pred/y
    y_pred=tf.reshape(y_pred,(-1,n*N))
    # else:
    #     y_true=tf.reshape(y_true,(-1,173*4,100))
    #     if y_true.shape[0]==1:
    #         y=y_test
    #     y=y.astype(np.float32)
    #     y=tf.reshape(y,(-1,1,100))
    #     y_true=y_true/y
    #     y_true=tf.reshape(y_true,(-1,173*400))
    #     y_pred=tf.reshape(y_pred,(-1,173*4,100))
    #     y_pred=y_pred/y
    #     y_pred=tf.reshape(y_pred,(-1,173*400))
    # loss=bkd.reduce_mean(bkd.square(y_true - y_pred))
    # loss=bkd.square(bkd.norm(y_true - y_pred) / bkd.norm(y_true))
    loss=bkd.square(bkd.norm(y_true - y_pred))
    return loss
    # loss=bkd.reduce_mean(bkd.square(y_true - y_pred))

def softmax_cross_entropy(y_true, y_pred):
    # TODO: pytorch
    return tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)


def zero(*_):
    # TODO: pytorch
    return tf.constant(0, dtype=config.real(tf))


LOSS_DICT = {
    "def_l2_relative_error": def_l2_relative_error,
    "mean absolute error": mean_absolute_error,
    "MAE": mean_absolute_error,
    "mae": mean_absolute_error,
    "mean squared error": mean_squared_error,
    "MSE": mean_squared_error,
    "mse": mean_squared_error,
    "mean absolute percentage error": mean_absolute_percentage_error,
    "MAPE": mean_absolute_percentage_error,
    "mape": mean_absolute_percentage_error,
    "mean l2 relative error": mean_l2_relative_error,
    "softmax cross entropy": softmax_cross_entropy,
    "zero": zero,
    "def_mse": def_mse,
    "smooth_logl2":smooth_logl2,
    "def_l2_error_chatgpt":def_l2_error_chatgpt,
    "custom_mixed_loss": custom_mixed_loss,
}


def get(identifier):
    """Retrieves a loss function.

    Args:
        identifier: A loss identifier. String name of a loss function, or a loss function.

    Returns:
        A loss function.
    """
    if isinstance(identifier, (list, tuple)):
        return list(map(get, identifier))

    if isinstance(identifier, str):
        return LOSS_DICT[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret loss function identifier:", identifier)

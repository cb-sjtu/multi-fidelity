import numpy as np
from . import backend as bkd
from sklearn import metrics
from .backend import tf
from . import config
import numpy as np
from scipy import io

def accuracy(y_true, y_pred):
    return np.mean(np.equal(np.argmax(y_pred, axis=-1), np.argmax(y_true, axis=-1)))


def l2_relative_error(y_true, y_pred):
    return np.square(np.linalg.norm(y_true- y_pred) / np.linalg.norm(y_true))


def nanl2_relative_error(y_true, y_pred):
    """Return the L2 relative error treating Not a Numbers (NaNs) as zero."""
    err = y_true - y_pred
    err = np.nan_to_num(err)
    y_true = np.nan_to_num(y_true)
    return np.linalg.norm(err) / np.linalg.norm(y_true)


def mean_l2_relative_error(y_true, y_pred):
    """Compute the average of L2 relative error along the first axis."""
    return np.mean(
        np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
    )


def _absolute_percentage_error(y_true, y_pred):
    return 100 * np.abs(
        (y_true - y_pred) / np.clip(np.abs(y_true), np.finfo(config.real(np)).eps, None)
    )


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(_absolute_percentage_error(y_true, y_pred))


def max_absolute_percentage_error(y_true, y_pred):
    return np.amax(_absolute_percentage_error(y_true, y_pred))


def absolute_percentage_error_std(y_true, y_pred):
    return np.std(_absolute_percentage_error(y_true, y_pred))


def mean_squared_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

def def_l2_relative_error(y_true, y_pred):
    
    y=np.zeros((4,100))
    y[0]=(io.loadmat('data_1207/stat_Re180.mat')['y'][:100][::-1].reshape((100,)))*182.088

    y[1]=(io.loadmat('data_1207/stat_Re550.mat')['y'][:100][::-1].reshape((100,)))*543.496
    # y[2]=(io.loadmat('data_0605/database/stat_Re1000.mat')['y'].reshape((200,)))*1000.512
    y[2]=(io.loadmat('data_1207/stat_Re2000.mat')['y'][:100][::-1].reshape((100,)))*1994.756
    y[3]=(io.loadmat('data_1207/stat_Re5200.mat')['y'][:100][::-1].reshape((100,)))*5185.897

    y_test=np.zeros((1,100))
    y_test[0]=(io.loadmat('data_1207/stat_Re1000.mat')['y'][:100][::-1].reshape((100,)))*1000.512
    y_true=np.reshape(y_true,(-1,87,100))
    if y_true.shape[0]==1:
        y=y_test
    y=y.astype(np.float32)
    y=np.reshape(y,(-1,1,100))
    y_true=y_true/y
    y_true=np.reshape(y_true,(-1,8700))
    y_pred=np.reshape(y_pred,(-1,87,100))
    y_pred=y_pred/y
    y_pred=np.reshape(y_pred,(-1,8700))

    loss=np.square(np.linalg.norm(y_true - y_pred) )
    
    return loss



def get(identifier):
    metric_identifier = {
        "accuracy": accuracy,
        "l2 relative error": l2_relative_error,
        "nanl2 relative error": nanl2_relative_error,
        "mean l2 relative error": mean_l2_relative_error,
        "mean squared error": mean_squared_error,
        "MSE": mean_squared_error,
        "mse": mean_squared_error,
        "MAE": metrics.mean_absolute_error,
        "MAPE": mean_absolute_percentage_error,
        "max APE": max_absolute_percentage_error,
        "APE SD": absolute_percentage_error_std,
        "def_l2_relative_error": def_l2_relative_error,
    }

    if isinstance(identifier, str):
        return metric_identifier[identifier]
    if callable(identifier):
        return identifier
    raise ValueError("Could not interpret metric function identifier:", identifier)

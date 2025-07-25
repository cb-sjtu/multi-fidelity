from .data import Data
from .sampler import BatchSampler


class Quadruple(Data):
    """Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices],
            self.train_x[2][indices]),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y


class QuadrupleCartesianProd(Data):
    """Cartesian Product input data format for MIONet architecture.

    This dataset can be used with the network ``MIONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays. The first element has the shape (`N1`,
            `dim1`), the second element has the shape (`N1`, `dim2`), and the third
            element has the shape (`N2`, `dim3`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        if (
            len(X_train[0]) * len(X_train[2]) != y_train.size
            or len(X_train[1]) * len(X_train[2]) != y_train.size
            or len(X_train[0]) != len(X_train[1])
        ):
            raise ValueError(
                "The training dataset does not have the format of Cartesian product."
            )
        if (
            len(X_test[0]) * len(X_test[2]) != y_test.size
            or len(X_test[1]) * len(X_test[2]) != y_test.size
            or len(X_test[0]) != len(X_test[1])
        ):
            raise ValueError(
                "The testing dataset does not have the format of Cartesian product."
            )
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.train_sampler = BatchSampler(len(X_train[0]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            self.train_x[0][indices],
            self.train_x[1][indices],
            self.train_x[2],
        ), self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y


class Fifthple(Data):
    """Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices],
            self.train_x[2][indices], self.train_x[3][indices]),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y



class Sixthple(Data):
    """Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

         # 动态计算总样本数
        if isinstance(self.train_y, tuple) or isinstance(self.train_y, list):
            # 如果 train_y 分为两部分，取其中一部分的长度
            total_samples = self.train_y[0].shape[0]
        else:
            # 如果 train_y 是单一部分
            total_samples = self.train_y.shape[0]

        self.train_sampler = BatchSampler(total_samples, shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y

        # 获取 batch 的索引
        indices = self.train_sampler.get_next(batch_size)

        # 总样本数
        total_samples = self.train_x[0].shape[0]
        half_samples = total_samples // 2

        # 初始化 batch_train_x 列表
        batch_train_x_high = []
        batch_train_x_low = []

        for data in self.train_x:
            high_precision_indices = indices[indices < half_samples]
            low_precision_indices = indices[indices >= half_samples] - half_samples

            if isinstance(data, tuple) or isinstance(data, list):
                # 分为两部分（高低精度各自采样）
                high_precision = data[0][high_precision_indices]
                low_precision = data[1][low_precision_indices]
            else:
                # 单一部分，但也分为两组
                high_precision = data[high_precision_indices]
                low_precision = data[low_precision_indices]

            batch_train_x_high.append(high_precision)
            batch_train_x_low.append(low_precision)

        # 最终打包
        batch_train_x = batch_train_x_high + batch_train_x_low


        # 处理 train_y（假设 train_y 也分为两部分）
        if isinstance(self.train_y, tuple) or isinstance(self.train_y, list):
            high_precision_indices = indices[indices < half_samples]
            low_precision_indices = indices[indices >= half_samples] - half_samples

            high_precision_y = self.train_y[0][high_precision_indices]
            low_precision_y = self.train_y[1][low_precision_indices]

            batch_train_y = (high_precision_y, low_precision_y)
        else:
            batch_train_y = self.train_y[indices]

        return batch_train_x, batch_train_y

    def test(self):
        # 假定 self.test_x 是和 self.train_x 一样的数据结构
        # 一般为：list，每个元素要么是 tuple/list（高低精度），要么是单一 np.ndarray
        batch_test_x_high = []
        batch_test_x_low = []

        # 不需要 indices，直接取所有样本
        for data in self.test_x:
            if isinstance(data, (tuple, list)):
                # 两部分都取全部
                high_precision = data[0]
                low_precision = data[1]
            else:
                # 假设是一个高，一个低的数据
                high_precision = data[0:1]
                low_precision = data[1:2]

            batch_test_x_high.append(high_precision)
            batch_test_x_low.append(low_precision)

        # 展平成一个平铺 list，和 train 完全一致
        test_x = batch_test_x_high + batch_test_x_low

        # 如果 label 结构同理，也分两路（高/低精度），展平
        if isinstance(self.test_y, (tuple, list)):
            high_precision_y = self.test_y[0]
            low_precision_y = self.test_y[1]
            test_y = (high_precision_y, low_precision_y)
        else:
            test_y = [self.test_y]

        return test_x, test_y

    



class Seventhple(Data):
    """Dataset with each data point as a quadruple.

    The couple of the first three elements are the input, and the fourth element is the
    output. This dataset can be used with the network ``MIONet`` for operator
    learning.

    Args:
        X_train: A tuple of three NumPy arrays.
        y_train: A NumPy array.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            (self.train_x[0][indices], self.train_x[1][indices],
            self.train_x[2][indices], self.train_x[3][indices],
            self.train_x[4][indices], self.train_x[5][indices]),
            self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y
from .nn import NN
from .. import activations
from .. import initializers
from .. import regularizers
from ... import config
from ...backend import tf
from ...utils import timing
from keras import layers
import numpy as np
# import tensorflow as tf


class DeepONet_resolvent_3d(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk1,
        layer_sizes_trunk2,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_trunk1 = layer_sizes_trunk1
        self.layer_trunk2 = layer_sizes_trunk2
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_trunk1 = activations.get(activation["trunk"])
            self.activation_trunk2 = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, 87,87,4800])
        self.X_loc = tf.placeholder(config.real(tf), [None,87*87*24,self.layer_trunk1[0]])
        self.trunk_out = tf.placeholder(tf.complex64, [None,87*87*24,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,23])
        self.dkxs_s = tf.placeholder(config.real(tf), [None,86])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s,self.dkxs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        self.X_loc= tf.reshape(self.X_loc,(-1,self.layer_trunk1[0]))
        # Trunk net1
        if callable(self.layer_trunk1[1]):
            # User-defined network
            y_loc1 = self.layer_trunk1[1](self.X_loc)
        else:
            y_loc1 = self._net(self.X_loc, self.layer_trunk1[1:], self.activation_trunk1)

        #Trunk net2
        if callable(self.layer_trunk2[1]):
            # User-defined network
            y_loc2 = self.layer_trunk2[1](self.X_loc)
        else:
            y_loc2 = self._net(self.X_loc, self.layer_trunk2[1:], self.activation_trunk2)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc1, axis=2)), 2))
           
        else:
            #实部
            y_loc1=tf.reshape(y_loc1,(-1,87*87*24,200))
            # y_func1=tf.reshape(y_func1,(-1,1,500))
            self.y_net1 = tf.einsum('bj,bkj->bk', y_func1, y_loc1)
            self.y_net1 = tf.expand_dims(self.y_net1, axis=-1)  # shape: [batch, N, 1]



            #虚部
            y_loc2=tf.reshape(y_loc2,(-1,87*87*24,200))
            self.y_net2 = tf.einsum('bj,bkj->bk', y_func1, y_loc2)
            self.y_net2 = tf.expand_dims(self.y_net2, axis=-1)  # shape: [batch, N, 1]
        #组合为复数
        self.y_net=tf.complex(self.y_net1,self.y_net2)

        # self.y_net=tf.reshape(self.y_net,(-1,87*87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b


        #积分
       
        q=self.trunk_out*self.y_net

        #将q的实部与虚部平方后相加
        q=tf.square(tf.real(q))+tf.square(tf.imag(q))

        # first=self.dcPs_s
        # second=self.dkxs_s
       
        # 第一轮积分
        x1 = tf.reshape(q, (-1, 87, 87, 24, 100))
        x_sum1 = x1[:, :, :, :-1, :] + x1[:, :, :, 1:, :]
        x_1 = 0.5 * tf.einsum('bhwkc,bk->bhwc', x_sum1, self.dcPs_s)

        # 第二轮积分
        x2 = tf.reshape(x_1, (-1, 87, 87, 100))
        x_sum3 = x2[:, :, :-1, :] + x2[:, :, 1:, :]
        x_2 = 0.5 * tf.einsum('bhkc,bk->bhc', x_sum3, self.dkxs_s)



        

        self.y=tf.reshape(x_2,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        # self.y=self.y_net

        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1],kernel_regularizer=self.regularizer)



class DeepONet_resolvent_3d_mix(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk1,
        layer_sizes_trunk2,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        data,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_trunk1 = layer_sizes_trunk1
        self.layer_trunk2 = layer_sizes_trunk2
        self.layer_dot = layer_sizes_dot
        self.data=data

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_trunk1 = activations.get(activation["trunk"])
            self.activation_trunk2 = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.regularizer = regularizers.get(regularization)

        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self.y

    @property
    def targets(self):
        return self.target

    @timing
    def build(self):
        print("Building MIONet...")
        self.X_func1 = tf.placeholder(config.real(tf), [None, 87,87,4800])
        self.X_loc = tf.placeholder(config.real(tf), [None,87*87*24,self.layer_trunk1[0]])
        self.trunk_out = tf.placeholder(tf.complex64, [None,87*87*24,200,2])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,23])
        self.dkxs_s = tf.placeholder(config.real(tf), [None,86])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s,self.dkxs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        self.X_loc= tf.reshape(self.X_loc,(-1,self.layer_trunk1[0]))
        # Trunk net1
        if callable(self.layer_trunk1[1]):
            # User-defined network
            y_loc1 = self.layer_trunk1[1](self.X_loc)
        else:
            y_loc1 = self._net(self.X_loc, self.layer_trunk1[1:], self.activation_trunk1)

        #Trunk net2
        if callable(self.layer_trunk2[1]):
            # User-defined network
            y_loc2 = self.layer_trunk2[1](self.X_loc)
        else:
            y_loc2 = self._net(self.X_loc, self.layer_trunk2[1:], self.activation_trunk2)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc1, axis=2)), 2))
           
        else:
            #实部
            y_loc1=tf.reshape(y_loc1,(-1,87*87*24,200))
            # y_func1=tf.reshape(y_func1,(-1,1,500))
            self.y_net1 = tf.einsum('bj,bkj->bk', y_func1, y_loc1)
            self.y_net1 = tf.expand_dims(self.y_net1, axis=-1)  # shape: [batch, N, 1]



            #虚部
            y_loc2=tf.reshape(y_loc2,(-1,87*87*24,200))
            self.y_net2 = tf.einsum('bj,bkj->bk', y_func1, y_loc2)
            self.y_net2 = tf.expand_dims(self.y_net2, axis=-1)  # shape: [batch, N, 1]
        #组合为复数
        self.y_net=tf.complex(self.y_net1,self.y_net2)

        # self.y_net=tf.reshape(self.y_net,(-1,87*87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b


        #积分
       
        trunk_u = self.trunk_out[:,:,:,0]
        trunk_v = self.trunk_out[:,:,:,1]
        q_u = trunk_u * self.y_net
        q_v = trunk_v * self.y_net
        #两个数相乘后取实部
        q=q_u*q_v
        q=tf.real(q)

       
        # 第一轮积分
        x1 = tf.reshape(q, (-1, 87, 87, 24, 200))
        x_sum1 = x1[:, :, :, :-1, :] + x1[:, :, :, 1:, :]
        x_1 = 0.5 * tf.einsum('bhwkc,bk->bhwc', x_sum1, self.dcPs_s)

        # 第二轮积分
        x2 = tf.reshape(x_1, (-1, 87, 87, 200))
        x_sum3 = x2[:, :, :-1, :] + x2[:, :, 1:, :]
        x_2 = 0.5 * tf.einsum('bhkc,bk->bhc', x_sum3, self.dkxs_s)

        # 第三轮积分
        x3 = tf.reshape(x_2, (-1, 87, 200))
        x_sum4 = x3[:, :-1, :] + x3[:, 1:, :]
        x_3 = 0.5 * tf.einsum('bkc,bk->bc', x_sum4, self.dkxs_s)


        # 获取数据的总数量
        total_count = tf.shape(x_2)[0]

        # 
        batch_train_x, batch_train_y, batch_indices = self.data.train_next_batch(config.batch_size)

        # 高精度样本的索引
        high_precision_indices = tf.where(batch_indices < total_count // 2)[:, 0]

        # 低精度样本的索引
        low_precision_indices = tf.where(batch_indices >= total_count // 2)[:, 0]

        # 从 x_2 和 x_3 中提取高精度和低精度样本
        x_2_high = tf.gather(x_2, high_precision_indices)
        x_3_low = tf.gather(x_3, low_precision_indices)

        # 将高精度样本 reshape 为 target_1 的形状
        self.y1 = tf.reshape(x_2_high, (-1, 87 * 200))

        # 将低精度样本 reshape 为 target_2 的形状
        self.y2 = tf.reshape(x_3_low, (-1, 200))

        # 将 y1 和 y2 组合为最终输出
        self.y = (self.y1, self.y2)
  
        # if self._output_transform is not None:
        #     self.y = self._output_transform(self.y)

        # self.y=self.y_net

        self.target1 = tf.placeholder(config.real(tf), [None, 87 * 200])
        self.target2 = tf.placeholder(config.real(tf), [None, 200])
        self.target = (self.target1, self.target2)

        self.built = True

    def _net(self, X, layer, activation):
        output = X
        for i in range(len(layer) - 1):
            output = tf.layers.dense(
                output,
                layer[i],
                activation=activation,
                kernel_regularizer=self.regularizer,
            )
        return tf.layers.dense(output, layer[-1],kernel_regularizer=self.regularizer)

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


class MIONet(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_loc)
        self.y = tf.multiply(self.y, y_func2)
        self.y = tf.reduce_sum(self.y, 1, keepdims=True)
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 1])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONetCartesianProd(MIONet):
    """MIONet with two input functions for Cartesian product format."""

    @timing
    def build(self):
        print("Building MIONetCartesianProd...")

        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None, self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = tf.multiply(y_func1, y_func2)
        self.y = tf.einsum("ip,jp->ij", self.y, y_loc)

        b = tf.Variable(tf.zeros(1))
        self.y += b
        self.target = tf.placeholder(config.real(tf), [None, None])
        self.built = True




class   MIONet_CNN(NN):
    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 = (
                self.activation_branch2
            ) = self.activation_trunk = activations.get(activation)
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,85*85,self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_func2, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2),tf.expand_dims(y_func2, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            self.y = tf.multiply(y_func1, y_loc)
            self.y = tf.multiply(self.y, y_func2)
            self.y = tf.reduce_sum(self.y, 1, keepdims=True)



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None,None])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class MIONet_CNN_no_average_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(tf.complex(config.real(tf), config.real(tf)), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            self.y_net = y_func1



        
        b = tf.Variable(tf.zeros(1))
        self.y_net += b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class SVD_DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func=y_func1[:,:24*10]
            y_c=y_func1[:,24*10:24*10+87*24]
            y_d=y_func1[:,24*10+87*24:]
            y_func=tf.reshape(y_func,(-1,1,24,10))
            self.y_net = tf.multiply(y_func, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24))
        self.y_net=tf.multiply(self.y_net,y_c)+y_d

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

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
        return tf.layers.dense(output, layer[-1],activation=activation, kernel_regularizer=self.regularizer)

class flex_DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_branch2,
        layer_sizes_trunk,
        # layer_sizes_trunk2,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_branch2 = layer_sizes_branch2
        self.layer_trunk = layer_sizes_trunk
        # self.layer_trunk2 = layer_sizes_trunk2
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_branch2 = activations.get(activation["branch2"])
            self.activation_trunk = activations.get(activation["trunk"])
            # self.activation_trunk2 = activations.get(activation["trunk2"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_func2 = tf.placeholder(config.real(tf), [None, self.layer_branch2[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        # self.X_loc2 = tf.placeholder(config.real(tf), [None,self.layer_trunk2[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1,self.X_func2, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Branch net 2
        if callable(self.layer_branch2[1]):
            # User-defined network
            y_func2 = self.layer_branch2[1](self.X_func2)
        else:
            y_func2 = self._net(
                self.X_func2, self.layer_branch2[1:], self.activation_branch2
            )
        y_func2_1=tf.reshape(y_func2[:,:87],(-1,1))
        # y_func2_2=tf.reshape(y_func2[:,87:],(-1,1))
        self.X_loc=tf.multiply(self.X_loc,y_func2_1)

        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)
        
        

        # # Trunk net2
        # if callable(self.layer_trunk2[1]):
        #     # User-defined network
        #     y_loc2 = self.layer_trunk2[1](self.X_loc2)
        # else:   
        #     y_loc2 = self._net(self.X_loc2, self.layer_trunk2[1:], self.activation_trunk2)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func=y_func1[:,:24*10]
            y_c=y_func1[:,24*10:24*10+87*24]
            y_d=y_func1[:,24*10+87*24:]
            y_func=tf.reshape(y_func,(-1,1,24,10))
            self.y_net = tf.multiply(y_func, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24))
        self.y_net=tf.multiply(self.y_net,y_c)+y_d
        

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        # b = tf.Variable(tf.zeros(1))
        # self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        # y_loc2=tf.reshape(y_loc2,(-1,1,100))
        # print(y_loc2.shape)
        # q=tf.multiply(q,y_loc2)
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)

class DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,10))
            y_func1=tf.reshape(y_func1,(-1,1,24,10))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)
    
class DeepONet_resolvent_2d(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,24,50))
            y_func1=tf.reshape(y_func1,(-1,1,24,50))
          
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')
    
        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

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
        return tf.layers.dense(output, layer[-1],activation=activation, kernel_regularizer=self.regularizer)
    

class DeepONet_resolvent_batch(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,24,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,24,1))
            y_func1=tf.reshape(y_func1,(-1,24,1))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        # self.y_net=tf.reshape(self.y_net,(-1,87*24,1))

        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net + b*b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        #x1=tf.reshape(q,(-1,24,100))
        x1=q
        x_f=x1[:,:-1,:]
        x_a=x1[:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=1) 
        # print(x_1,'over')

        self.y=x_1
        #self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

        #self.y=self.y_net

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
        return tf.layers.dense(output, layer[-1],activation=activation, kernel_regularizer=self.regularizer)

class D_DeepONet_resolvent(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,2088,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,23])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network 有点乘
            y_loc=tf.reshape(y_loc,(-1,87,24,50))
            y_func1=tf.reshape(y_func1,(-1,1,24,50))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net = self.layer_dot[0](self.y_net)

            #没点乘
            # self.y_net = self.layer_dot[0](tf.concat([y_func1,y_loc], 1))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,87,50,24))
            y_func1=tf.reshape(y_func1,(-1,1,50,24))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,2,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,87*24,1))

        
        b = tf.Variable(tf.zeros(1))
        self.y_net += b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,87,24,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,23,1))
        x_sum3=(x_sum1*first*0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        self.y=tf.reshape(x_1,(-1,87*100))
  
        if self._output_transform is not None:
            self.y = self._output_transform( self.y)

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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class DeepONet_resolvent_jiami(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None, 173 * 47, 200])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,46])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,173,47,50))
            y_func1=tf.reshape(y_func1,(-1,1,47,50))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,173*47,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b


        #积分
       
        q=self.trunk_out*self.y_net

        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,173,47,200))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,46,1))
        x_sum3=tf.multiply(x_sum1,first)
        x_sum3=tf.multiply(x_sum3,0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        

        self.y=tf.reshape(x_1,(-1,173*200))
  
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


class DeepONet_resolvent_jiami_1000(NN):


    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,173*47,800])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,46])
        self._inputs = [self.X_func1, self.X_loc,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,173,47,50))
            y_func1=tf.reshape(y_func1,(-1,1,47,50))
            self.y_net = tf.multiply(y_func1, y_loc)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,173*47,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net
        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,173,47,800))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,46,1))
        x_sum3=tf.multiply(x_sum1,first)
        x_sum3=tf.multiply(x_sum3,0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        
        
        # self.X_loc=tf.reshape(self.X_loc,(-1,173,1))
        # self.y=tf.multiply(x_1,self.X_loc)

        self.y=tf.reshape(x_1,(-1,173*800))
  
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

class DeepONet_resolvent_jiami_y_trunk(NN):
    """Multiple-input operator network with two input functions."""

   
    

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_y_trunk,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        self.layer_y_trunk = layer_sizes_y_trunk
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot

        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            self.activation_y_trunk = activations.get(activation["y_trunk"])
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0]])
        self.y_trunk=tf.placeholder(config.real(tf), [None,self.layer_y_trunk[0]])
        self.trunk_out = tf.placeholder(config.real(tf), [None,173*47,100])
        self.dcPs_s = tf.placeholder(config.real(tf), [None,1,46])
        self._inputs = [self.X_func1, self.X_loc,self.y_trunk,self.trunk_out,self.dcPs_s]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # y_trunk net
        if callable(self.layer_y_trunk[1]):
            # User-defined network
            y_trunk = self.layer_y_trunk[1](self.y_trunk)
        else:
            y_trunk = self._net(self.y_trunk, self.layer_y_trunk[1:], self.activation_y_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y_net = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            y_loc=tf.reshape(y_loc,(-1,173,47,50))
            y_func1=tf.reshape(y_func1,(-1,1,47,50))
            # y_trunk=tf.reshape(y_trunk,(-1,173,1,1))
            self.y_net = tf.multiply(y_func1, y_loc)
            # self.y_net=tf.multiply(self.y_net,y_trunk)
            self.y_net=tf.reduce_sum(self.y_net,-1,keepdims=True)

        self.y_net=tf.reshape(self.y_net,(-1,173*47,1))
        
        b = tf.Variable(tf.zeros(1))
        self.y_net=self.y_net+b

        #积分
       
            

        q=self.trunk_out*self.y_net

        
        y_trunk=tf.reshape(y_trunk,(-1,100,47))
        y_trunk=tf.transpose(y_trunk,perm=[0,2,1])
        y_trunk=tf.reshape(y_trunk,(-1,1,47,100))
        q=tf.reshape(q,(-1,173,47,100))
        q=tf.multiply(q,y_trunk)

        # q=tf.reshape(q,(-1,173*47,1))

        first=self.dcPs_s
       

        #first round
        x1=tf.reshape(q,(-1,173,47,100))
        x_f=x1[:,:,:-1,:]
        x_a=x1[:,:,1:,:]
        x_sum1=x_f+x_a
        
        # x_sum2=tf.reshape(x_sum1,(-1,100*87,23))
        first=tf.reshape(first,(-1,1,46,1))
        x_sum3=tf.multiply(x_sum1,first)
        x_sum3=tf.multiply(x_sum3,0.5)
        x_1=tf.reduce_sum(x_sum3,axis=2) 
        # print(x_1,'over')

        

        self.y=tf.reshape(x_1,(-1,173*100))
  
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


class MIONet_CNN_no_average(NN):
    """Multiple-input operator network with two input functions."""


    def __init__(
        self,
        layer_sizes_branch1,
        
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot



        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        
        self.X_loc = tf.placeholder(config.real(tf), [None,None,self.layer_trunk[0]])
        self._inputs = [self.X_func1, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=2), tf.expand_dims(y_loc, axis=2)), 2))
           
        else:
            self.y = tf.multiply(y_func1, y_loc)
            self.y = tf.reduce_sum(self.y, 1, keepdims=True)



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class Transformer_decoder_deeponet(NN):
    """Multiple-input operator network with two input functions."""




    def __init__(
        self,
        # training,
        layer_sizes_branch1,
        
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        embed_dim, num_heads, ff_dim,ff_dim_final, transformer_layers_b,transformer_layers_t,rate=0.1,patch_size=16,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        # self.training=training
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.ff_dim=ff_dim
        self.ff_dim_final=ff_dim_final
        self.rate=rate

        #copy from transformer.py
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), 
             tf.keras.layers.Dense(embed_dim),]
        )
       
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.patch_size = patch_size
        # patch_size_squared = self.patch_size * self.patch_size
        self.embedding_b1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        self.embedding_t = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        self.pos_emb_b1 = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.pos_emb_t = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.transformer_layers_b = transformer_layers_b
        self.transformer_layers_t = transformer_layers_t
        #copy end


        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0],self.layer_branch1[1],self.layer_branch1[2]])
        
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0],self.layer_trunk[1],self.layer_trunk[2]])
        self._inputs = [self.X_func1, self.X_loc]

        

        # Branch net 1
      
        #patch
        batch_size_b1 = tf.shape(self.X_func1)[0]
        patches_b1 = tf.image.extract_patches(
            images=self.X_func1,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims_b1 = patches_b1.shape[-1]
        patches_b1 = tf.reshape(patches_b1, [batch_size_b1, patches_b1.shape[1]*patches_b1.shape[2], patch_dims_b1])

        #embedding
        positions_b1 = tf.range(start=0, limit=288, delta=1)
        embeddings_b1 = self.embedding_b1(patches_b1)+self.pos_emb_b1(positions_b1)
     
        x_b1=embeddings_b1
        #transformer
        transformer_layers_b1 = []
        for _ in range(self.transformer_layers_b):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim,ff_dim_final=self.ff_dim_final, rate=self.rate)
            transformer_layers_b1.append(layer)
        
        for layer in transformer_layers_b1:
            x_b1 = layer(x_b1,None)
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        y_func1=x_b1
        # for _ in range(self.transformer_layers):
        #     attn_output_b1 = self.att(x_b1 , x_b1)
        #     attn_output_b1 = self.dropout1(attn_output_b1)
        #     out1_b1 = self.layernorm1(x_b1 + attn_output_b1)
        #     ffn_output_b1 = self.ffn(out1_b1)
        #     ffn_output_b1 = self.dropout2(ffn_output_b1)
        #     x_b1=self.layernorm2(out1_b1 + ffn_output_b1)
        # y_func1=x_b1

        # if callable(self.layer_branch1[1]):
        #     # User-defined network
        #     y_func1 = self.layer_branch1[1](self.X_func1)
        # else:
        #     y_func1 = self._net(
        #         self.X_func1, self.layer_branch1[1:], self.activation_branch1
        #     )
        
        # Trunk net
        
        #patch
        batch_size_t = tf.shape(self.X_loc)[0]
        patches_t = tf.image.extract_patches(
            images=self.X_loc,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims_t = patches_t.shape[-1]
        patches_t = tf.reshape(patches_t, [batch_size_t, patches_t.shape[1]*patches_t.shape[2], patch_dims_t])

        #embedding
        positions_t = tf.range(start=0, limit=288, delta=1)
        embeddings_t = self.embedding_t(patches_t)+self.pos_emb_t(positions_t)
        x_t=embeddings_t
        #transformer
        transformer_layers_t = []
        for _ in range(self.transformer_layers_b):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim,ff_dim_final=self.ff_dim_final, rate=self.rate)
            transformer_layers_t.append(layer)
        
        for layer in transformer_layers_t:
            x_t = layer(x_t,None)
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        y_loc=x_t
        # for _ in range(self.transformer_layers):
        #     attn_output_t = self.att(x_t, x_t)
        #     attn_output_t = self.dropout1(attn_output_t)
        #     out1_t = self.layernorm1(x_t + attn_output_t)
        #     ffn_output_t = self.ffn(out1_t)
        #     ffn_output_t = self.dropout2(ffn_output_t)
        #     x_t=self.layernorm2(out1_t + ffn_output_t)
        # y_loc=x_t

        # if callable(self.layer_trunk[1]):
        #     # User-defined network
        #     y_loc = self.layer_trunk[1](self.X_loc)
        # else:
        #     y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product



        if callable(self.layer_dot[0]):
            # User-defined network
            
            self.y = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=3), tf.expand_dims(y_loc, axis=3)), 3))
           
        else:
            self.y = tf.multiply(y_func1, y_loc)
            self.y = tf.reduce_sum(self.y, 1, keepdims=True)



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

        self.target = tf.placeholder(config.real(tf), [None, 221184])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class decoder_deeponet_Transformer(NN):
    """Multiple-input operator network with two input functions."""




    def __init__(
        self,
        # training,
        layer_sizes_branch1,
        layer_sizes_trunk,
       
        activation,
        kernel_initializer,
        regularization,
        embed_dim, num_heads, ff_dim, transformer_layers_d,ff_dim_final,rate=0.1,patch_size=16,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        # self.training=training
        self.layer_trunk = layer_sizes_trunk
       
        self.layer_trunk = layer_sizes_trunk
        
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.ff_dim=ff_dim
        # self.ff_dim_final=ff_dim_final
        self.rate=rate


        #copy from transformer.py
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            # [ tf.keras.layers.Flatten(),
              [tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(ff_dim, activation="relu"),
            #   tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(embed_dim,activation="relu")]
            #   tf.keras.layers.Reshape((2,64)),]
        )
        self.ffn_final = ff_dim_final
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.patch_size = patch_size
        # patch_size_squared = self.patch_size * self.patch_size
        # self.embedding_b1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        # self.embedding_t = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        # self.pos_emb_b1 = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        # self.pos_emb_t = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.transformer_layers_d = transformer_layers_d
        #copy end


        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, 6,5])
        
        self.X_loc = tf.placeholder(config.real(tf), [None,6])
        self._inputs = [self.X_func1, self.X_loc]



        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            self.y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            self.y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # self.y_func1_0=self.y_func1

        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            self.y_loc = self.layer_trunk[1](self.X_loc)
        else:
            self.y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)
        
        # self.y_loc_0=self.y_loc

        
        #transformer
        transformer_layers_d = []
        for _ in range(self.transformer_layers_d):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim, rate=self.rate)
            transformer_layers_d.append(layer)
        # i=0
        for layer in transformer_layers_d:
            
            self.y_loc= layer(self.y_func1,self.y_loc)
            # if i==11:
            #     self.y_loc_0=self.y_loc
            # y_loc = y_func1
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
            # i=i+1
        self.y=self.ffn_final(self.y_loc)
        # self.y=y_func1
        # self.y0=tf.zeros_like(self.y)
        # a=tf.keras.layers.Flatten()(self.y_loc_0)
        # # a=tf.reduce_sum(self.y_loc,axis=1)
        # # a=tf.reshape(self.y_loc,(20,-1))
        # b=tf.zeros((20,23844))
        # self.y0=tf.concat((a,b),axis=1)
        



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

        if self._output_transform is not None:
            self.y = self._output_transform(self._inputs,self.y)
        # self.y=self.y0
        self.target = tf.placeholder(config.real(tf), [None, 1])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)



class decoder_deeponet_multiply_Transformer(NN):
    """Multiple-input operator network with two input functions."""




    def __init__(
        self,
        # training,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        embed_dim, num_heads, ff_dim, transformer_layers_d,ff_dim_final,rate=0.1,patch_size=16,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        # self.training=training
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.ff_dim=ff_dim
        # self.ff_dim_final=ff_dim_final
        self.rate=rate


        #copy from transformer.py
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            # [ tf.keras.layers.Flatten(),
              [tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(ff_dim, activation="relu"),
            #   tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(embed_dim,activation="relu")]
            #   tf.keras.layers.Reshape((2,64)),]
        )
        self.ffn_final = ff_dim_final
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.patch_size = patch_size
        # patch_size_squared = self.patch_size * self.patch_size
        # self.embedding_b1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        # self.embedding_t = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        # self.pos_emb_b1 = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        # self.pos_emb_t = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.transformer_layers_d = transformer_layers_d
        #copy end


        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0]])
        
        self.X_loc = tf.placeholder(config.real(tf), [None,24100,2])
        self._inputs = [self.X_func1, self.X_loc]



        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        
        # Trunk net
        if callable(self.layer_trunk[1]):
            # User-defined network
            y_loc = self.layer_trunk[1](self.X_loc)
        else:
            y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)
        

        y_merge=tf.multiply(y_func1, y_loc)
        
        #transformer
        transformer_layers_d = []
        for _ in range(self.transformer_layers_d):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim, rate=self.rate)
            transformer_layers_d.append(layer)
        
        for layer in transformer_layers_d:
            y_func1 = layer(y_merge,None)
            # y_loc = y_func1
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        self.y=self.ffn_final(y_loc)



        
        b = tf.Variable(tf.zeros(1))
        self.y += b

        if self._output_transform is not None:
            self.y = self._output_transform(self._inputs,self.y)

        self.target = tf.placeholder(config.real(tf), [None, 24100])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)


class All_Transformer(NN):


    """Multiple-input operator network with two input functions."""




    def __init__(
        self,
        # training,
        layer_sizes_branch1,
        layer_sizes_trunk,
        layer_sizes_dot,
        activation,
        kernel_initializer,
        regularization,
        embed_dim, num_heads, ff_dim, ff_dim_final,
        transformer_layers_b,transformer_layers_t,transformer_layers_d,
        rate=0.1,patch_size=16,
       ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
        # self.training=training
        self.layer_trunk = layer_sizes_trunk
        self.layer_dot = layer_sizes_dot
        self.num_heads=num_heads
        self.embed_dim=embed_dim
        self.ff_dim=ff_dim
        self.ff_dim_final=ff_dim_final
        self.rate=rate


        #copy from transformer.py
        # self.att =tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # self.ffn_b = tf.keras.Sequential(
        #     [tf.keras.layers.Dense(ff_dim, activation="relu"), 
        #      tf.keras.layers.Dense(embed_dim),]
        # )
      
        # self.ffn_t = tf.keras.Sequential(
        #     [tf.keras.layers.Dense(ff_dim, activation="relu"), 
        #      tf.keras.layers.Dense(embed_dim),]
        # )

        # self.ffn_decoder= tf.keras.Sequential(
        #     # [ tf.keras.layers.Flatten(),
        #       [tf.keras.layers.Dense(ff_dim, activation="relu"),
        #       tf.keras.layers.Dense(ff_dim, activation="relu"),
        #     #   tf.keras.layers.Dense(ff_dim, activation="relu"),
        #       tf.keras.layers.Dense(embed_dim,activation="relu")]
        #     #   tf.keras.layers.Reshape((2,64)),]
        # )
        self.ffn_final = tf.keras.Sequential(
            [
             tf.keras.layers.Dense(ff_dim_final, activation="relu"),
            #  tf.keras.layers.Reshape((288,-1)), 
             tf.keras.layers.Dense(768),
             tf.keras.layers.Flatten(),]
        )
       
        # self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # self.dropout1 = tf.keras.layers.Dropout(rate)
        # self.dropout2 = tf.keras.layers.Dropout(rate)
        self.patch_size = patch_size
        # patch_size_squared = self.patch_size * self.patch_size
        self.embedding_b1 = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        self.embedding_t = tf.keras.layers.Conv1D(embed_dim, kernel_size=1, padding="valid", use_bias=False)
        self.pos_emb_b1 = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.pos_emb_t = tf.keras.layers.Embedding(input_dim=288, output_dim=embed_dim)
        self.transformer_layers_b = transformer_layers_b
        self.transformer_layers_t = transformer_layers_t
        self.transformer_layers_d = transformer_layers_d
        #copy end


        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
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
        self.X_func1 = tf.placeholder(config.real(tf), [None, self.layer_branch1[0],self.layer_branch1[1],self.layer_branch1[2]])
        
        self.X_loc = tf.placeholder(config.real(tf), [None,self.layer_trunk[0],self.layer_trunk[1],self.layer_trunk[2]])
        self._inputs = [self.X_func1, self.X_loc]

        NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("Number of parameters: %d" % NumParameters)

        # Branch net 1

        #patch
        batch_size_b1 = tf.shape(self.X_func1)[0]
        patches_b1 = tf.image.extract_patches(
            images=self.X_func1,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims_b1 = patches_b1.shape[-1]
        patches_b1 = tf.reshape(patches_b1, [batch_size_b1, patches_b1.shape[1]*patches_b1.shape[2], patch_dims_b1])
        NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("Number of parameters: %d" % NumParameters)
        #embedding
        positions_b1 = tf.range(start=0, limit=288, delta=1)
        embeddings_b1 = self.embedding_b1(patches_b1)+self.pos_emb_b1(positions_b1)
     
        x_b1=embeddings_b1
        NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("Number of parameters: %d" % NumParameters)
        #transformer
        transformer_layers_b1 = []
        for _ in range(self.transformer_layers_b):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim,ff_dim_final=self.ff_dim_final, rate=self.rate)
            transformer_layers_b1.append(layer)
        
        for layer in transformer_layers_b1:
            x_b1 = layer(x_b1,None)
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        y_func1=x_b1
        
        # Trunk net
        
        #patch
        batch_size_t = tf.shape(self.X_loc)[0]
        patches_t = tf.image.extract_patches(
            images=self.X_loc,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims_t = patches_t.shape[-1]
        patches_t = tf.reshape(patches_t, [batch_size_t, patches_t.shape[1]*patches_t.shape[2], patch_dims_t])

        #embedding
        positions_t = tf.range(start=0, limit=288, delta=1)
        embeddings_t = self.embedding_t(patches_t)+self.pos_emb_t(positions_t)
        x_t=embeddings_t
        #transformer
        transformer_layers_t = []
        for _ in range(self.transformer_layers_b):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim,ff_dim_final=self.ff_dim_final, rate=self.rate)
            transformer_layers_t.append(layer)
        
        for layer in transformer_layers_t:
            x_t = layer(x_t,None)
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        y_loc=x_t
      
        #Decoder transformer
        transformer_layers_d = []
        for _ in range(self.transformer_layers_d):
            layer = MyTransformerLayer(num_heads=self.num_heads, embed_dim=self.embed_dim, ff_dim=self.ff_dim,ff_dim_final=self.ff_dim_final, rate=self.rate)
            transformer_layers_d.append(layer)
        
        for layer in transformer_layers_d:
            y_func1 = layer(y_func1,y_loc)
            y_loc = y_func1
            NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
            print("Number of parameters: %d" % NumParameters)
        self.y=self.ffn_final(y_loc)

        # for _ in range(self.transformer_layers_d):
        #     attn_output_b1 = self.att(y_func1 , y_func1, y_loc)
        #     attn_output_b1 = self.dropout1(attn_output_b1)
        #     out1_b1 = self.layernorm1(y_func1 + attn_output_b1)
        #     ffn_output_b1 = self.ffn_decoder(out1_b1)
        #     ffn_output_b1 = self.dropout2(ffn_output_b1)
        #     #x_b1=self.layernorm2(out1_b1 + ffn_output_b1)
        #     y_func1=self.layernorm2(out1_b1 + ffn_output_b1)
        #     y_loc=self.layernorm2(out1_b1 + ffn_output_b1)
        #     NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        #     print("Number of parameters: %d" % NumParameters)
        # self.y=self.ffn_final(y_loc)



        # if callable(self.layer_dot[0]):
        #     # User-defined network
            
        #     self.y = self.layer_dot[0](tf.concat((tf.expand_dims(y_func1, axis=3), tf.expand_dims(y_loc, axis=3)), 3))
           
        # else:
        #     self.y = tf.multiply(y_func1, y_loc)
        #     self.y = tf.reduce_sum(self.y, 1, keepdims=True)


        NumParameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()])
        print("Number of parameters: %d" % NumParameters)
        
        b = tf.Variable(tf.zeros(1))
        self.y += b
        if self._output_transform is not None:
            self.y = self._output_transform(self._inputs,self.y)
            
        self.target = tf.placeholder(config.real(tf), [None, 221184])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)



class MyTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, embed_dim, ff_dim, rate):
        super(MyTransformerLayer, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"), 
            tf.keras.layers.Dense(embed_dim),
        ])
        self.ffn_d= tf.keras.Sequential(
            # [ tf.keras.layers.Flatten(),
              [tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(ff_dim, activation="relu"),
            #   tf.keras.layers.Dense(ff_dim, activation="relu"),
              tf.keras.layers.Dense(embed_dim,activation="relu")]
            #   tf.keras.layers.Reshape((2,64)),]
        )
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs,v):
        if v is None:
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output)
            return self.layernorm2(out1 + ffn_output)
        else:
            attn_output = self.att(inputs, v,v)
            attn_output = self.dropout1(attn_output)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn_d(out1)
            ffn_output = self.dropout2(ffn_output)
            return self.layernorm2(out1 + ffn_output)
            
        # attn_output = self.att(inputs, inputs,)
        


class Direct(NN):





    """Multiple-input operator network with two input functions."""

    def __init__(
        self,
        layer_sizes_branch1,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()

        self.layer_branch1 = layer_sizes_branch1
       
        self.layer_trunk = layer_sizes_trunk
        if isinstance(activation, dict):
            self.activation_branch1 = activations.get(activation["branch1"])
            
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch1 =self.activation_trunk = activations.get(activation)
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
        self.X_func1 = tf.placeholder(config.real(tf), [None,self.layer_branch1[0]])
   
        self.X_loc = tf.placeholder(config.real(tf), [None,1])
        self._inputs = [self.X_func1, self.X_loc]

        # Branch net 1
        if callable(self.layer_branch1[1]):
            # User-defined network
            y_func1 = self.layer_branch1[1](self.X_func1)
        else:
            y_func1 = self._net(
                self.X_func1, self.layer_branch1[1:], self.activation_branch1
            )
        # # Branch net 2
        # if callable(self.layer_branch2[1]):
        #     # User-defined network
        #     y_func2 = self.layer_branch2[1](self.X_func2)
        # else:
        #     y_func2 = self._net(
        #         self.X_func2, self.layer_branch2[1:], self.activation_branch2
        #     )
        # # Trunk net
        # y_loc = self._net(self.X_loc, self.layer_trunk[1:], self.activation_trunk)

        # Dot product
        self.y = y_func1
        b = tf.Variable(tf.zeros(1))
        self.y += b
        if self._output_transform is not None:
            self.y = self._output_transform(self.y)
        self.target = tf.placeholder(config.real(tf), [None, 173*200])
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
        return tf.layers.dense(output, layer[-1], kernel_regularizer=self.regularizer)




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

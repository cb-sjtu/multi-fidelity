import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import math
from deepxde.nn.tensorflow_compat_v1.mionet import DeepONet_resolvent_3d
from deepxde.data.triple import Triple
# scale_0=io.loadmat('data_0605/database/stat_Re1000.mat')
#最优的结构
# num = [670,1000,1410,2000,2540,3030,3270,3630,3970,4060]
def network(problem, m,N_points):
    if problem == "ODE":
        branch_1 = [m, 200, 200]
        branch_2 = [N_points, 200, 200]
        trunk = [1, 200, 200]
    elif problem == "DR":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 200, 200]
    elif problem == "ADVD":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 300, 300, 300]
    elif problem == "flow":
        branch = tf.keras.Sequential(
            [
                
                tf.keras.layers.InputLayer(input_shape=(87*87*24*2*100)),
                tf.keras.layers.Reshape((87,87,24*200)),
                # tf.keras.layers.transpose,
                tf.keras.layers.Conv2D(2000, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(500, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(500, (3, 3), strides=1, activation="ReLU"),
                #池化层
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),
                tf.keras.layers.Conv2D(1000, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(1000, (3, 3), strides=1, activation="ReLU"),
                #池化层
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"),
                tf.keras.layers.Conv2D(500, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(100, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(10, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(1, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(500, activation="ReLU"),
                tf.keras.layers.Dense(200, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(500),
              
                #tf.keras.layers.Reshape((47,50)),
                # tf.keras.layers.Lambda(lambda x: transpose_last_two_dims(x))

            ]
        )
        branch.summary()

        branch=[m,branch]
       
       #分别是实部和虚部
        trunk_1=[3,128,256,515,256,500]
        trunk_2=[3,128,256,512,256,500]





        dot= tf.keras.Sequential(
        [   tf.keras.layers.InputLayer(input_shape=(87*24,2,)),
            # tf.keras.layers.Reshape((2,200, 1)),
            # # tf.keras.layers.Dense(1000, activation="relu"),
            # # tf.keras.layers.Dense(581248, activation="relu"),
            # # tf.keras.layers.Reshape((38,239,64)),
            # # tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation="relu"),
            # tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[2,2], strides=[5,5], activation="relu",padding="valid",data_format='channels_last'),
            # tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], strides=[1,1], activation="relu",padding="valid",data_format='channels_last'),
            tf.keras.layers.Flatten(),   
            # tf.keras.layers.Dense(1000, activation="relu"),
            # # tf.keras.layers.Dense(1000, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(87*24*1, activation="ReLU"),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(87*24*100),
            # tf.keras.layers.Reshape((87*24,100)),

        ]
      )
    dot.summary()
    dot=[0,dot]
    return branch,trunk_1,trunk_2,dot

def get_data():
    
    kzs=np.load('data_gen/data_kzs87.npy').astype(np.float32).reshape((5,1,87))
    uum=np.load('data_gen/data_um_8700.npy').astype(np.float32)

    uum=kzs*uum

    uum_new=uum[[0,1,3,4],0:100].reshape((4,-1))
    uum_new_test=uum[[2],0:100].reshape((1,-1))
    
    trunk_out=np.load('data_gen/data_utrunkout_5.npy')[[0,1,3,4],:100].reshape((4,100,-1))
    trunk_out_test=np.load('data_gen/data_utrunkout_5.npy')[[2],:100].reshape((1,100,-1))
    motai=trunk_out.transpose(0,2,1)
    motai_test=trunk_out_test.transpose(0,2,1)
    #把实部和虚部分别占据两个channel，合在一起
    branch_in=np.concatenate((trunk_out.real[..., np.newaxis], trunk_out.imag[..., np.newaxis]), axis=-1).astype(np.float32).reshape((4,100,-1))
    branch_in_test=np.concatenate((trunk_out_test.real[..., np.newaxis], trunk_out_test.imag[..., np.newaxis]), axis=-1).astype(np.float32).reshape((1,100,-1))
    
    
    

    for i in range(4):
        scaler_Euuc = StandardScaler().fit(branch_in[i])
        std_Euuc = np.sqrt(scaler_Euuc.var_.astype(np.float32))
        branch_in[i]=(branch_in[i]-scaler_Euuc.mean_.astype(np.float32))/std_Euuc
    #np.save("data_gen/data_motai_zz.npy",trunk_out)
    #np.save("data_gen/data_motai_zz.npy",trunk_out)
    for i in range(1):  
        scaler_Euuc_test = StandardScaler().fit(branch_in_test[i])
        std_Euuc_test = np.sqrt(scaler_Euuc_test.var_.astype(np.float32))
        branch_in_test[i]=(branch_in_test[i]-scaler_Euuc_test.mean_.astype(np.float32))/std_Euuc_test

    kzs_s=kzs[[0,1,3,4]]
    kzs_s[0]=2*math.pi/kzs_s[0]*182.088
    kzs_s[1]=2*math.pi/kzs_s[1]*543.496
    kzs_s[2]=2*math.pi/kzs_s[2]*1000.512
    kzs_s[3]=2*math.pi/kzs_s[3]*1994.756
    #kzs_s[4]=2*math.pi/kzs_s[4]*5185.897
    kzs_s_test=kzs[[2]]
    kzs_s_test=2*math.pi/kzs_s_test*1000.512

    dcPs_s=np.load('data_gen/data_dcp23.npy').astype(np.float32)[[0,1,3,4]].reshape((-1,1,23))
    dcPs_s_test=np.load('data_gen/data_dcp23.npy').astype(np.float32)[[2]].reshape((-1,1,23))
    dkxs_s=np.load('data_gen/data_dkx86.npy').astype(np.float32)[[0,1,3,4]].reshape((-1,1,86))
    dkxs_s_test=np.load('data_gen/data_dkx86.npy').astype(np.float32)[[2]].reshape((-1,1,86))
 
    y=np.load('data_gen/data_y100.npy')
    real_2d_y=y
    real_2d_x=np.zeros((5,87))
    real_2d_x[[0,1,3,4]]=kzs_s.reshape((4,87))
    real_2d_x[2]=kzs_s_test.reshape((1,87))
    
    y[0]=y[0]*182.088
    y[1]=y[1]*543.496
    y[2]=y[2]*1000.512
    y[3]=y[3]*1994.756
    y[4]=y[4]*5185.897
    y=y[[0,1,3,4]]
    y_test=y[[2]]

    real_2d=np.zeros((5,100,87,2))

    for i in range(5):
    # 如果y的长度大于100则裁剪，如果小于则重复直到100
        real_2d_y_i = real_2d_y[i]
        
        repeats = 100 // real_2d_y_i.shape[0] + 1
        real_2d_y_i = np.tile(real_2d_y_i, repeats)[:100]  # 重复直到100个
        
        # 填充坐标
        for j in range(100):
            real_2d[i, j, :, 0] = real_2d_x[i]  # 复制x坐标对该列
            real_2d[i, j, :, 1] = real_2d_y_i[j]
    coodinates = np.load('data_gen/3d_coordinates.npy').astype(np.float32)[[0,1,3,4]].reshape((4,-1,3))
    coodinates_test = np.load('data_gen/3d_coordinates.npy').astype(np.float32)[[2]].reshape((1,-1,3))
    
    

    return uum_new,uum_new_test,branch_in, branch_in_test,dcPs_s,dcPs_s_test,dkxs_s,dkxs_s_test,kzs_s,kzs_s_test,y,y_test,real_2d,kzs,motai,motai_test,coodinates,coodinates_test



def main():
    dataframe="DeepONet_resolvent_3d"
    uum,uum_test,trunk_out, trunk_out_test, dcPs_s ,dcPs_s_test,dkxs_s,dkxs_s_test,lambda_z,lambda_z_test,yy,yy_test,real_2d,kzs,motai,motai_test,coodinates,coodinates_test = get_data()
    lambda_z=np.reshape(lambda_z,(-1,1)).astype(np.float32)
    lambda_z_test=np.reshape(lambda_z_test,(-1,1)).astype(np.float32)
    coodinates=np.reshape(coodinates,(-1,3)).astype(np.float32)
    coodinates_test=np.reshape(coodinates_test,(-1,3)).astype(np.float32)
    uum=np.reshape(uum,(-1,100,87)).transpose(0,2,1).reshape((-1,100*87))
    # uum=(uum,y)
    uum_test=np.reshape(uum_test,(-1,100,87)).transpose(0,2,1).reshape((-1,100*87))
    # uum_test=(uum_test,y_test)
    trunk_out=trunk_out.transpose(0,2,1)
    trunk_out_test=trunk_out_test.transpose(0,2,1)
    trunk_out_input=(trunk_out.reshape((-1,87*87*24*2*100)),coodinates,motai ,dcPs_s,dkxs_s)
    trunk_out_input_test=(trunk_out_test.reshape((-1,87*87*24*2*100)),coodinates_test,motai_test,dcPs_s_test,dkxs_s_test)
    
    # #归一化
    # scaler_Euuc = MinMaxScaler().fit(trunk_out_input[0])
    # scaler_uum = MinMaxScaler().fit(uum)
    # trunk_out_input = (scaler_Euuc.transform(trunk_out_input[0]),scaler_Euuc.transform(trunk_out_input[1]))
    # trunk_out_input_test = (scaler_Euuc.transform(trunk_out_input_test[0]),scaler_Euuc.transform(trunk_out_input_test[1]))
    # # uum = scaler_uum.transform(uum)
    problem = "flow"
    N_points = 87*24
    data = dde.data.Sixthple(trunk_out_input,uum,trunk_out_input_test,uum_test)
    m = 100*87*87*24*2
    activation = (
        ["relu", "relu", "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    )
    initializer = "Glorot normal"
    

    branch_net,trunk_net_1,trunk_net_2,dot = network(problem,m,N_points)
   
   
    net = DeepONet_resolvent_3d(
        branch_net,
        trunk_net_1,
        trunk_net_2,
        dot,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        kernel_initializer=initializer,
        regularization=None,
    )
    
    if isinstance(uum, tuple):
        uum_zhengze = uum[0]
    else:
        uum_zhengze = uum
    scaler = StandardScaler().fit(uum_zhengze)
    std = np.sqrt(scaler.var_.astype(np.float32))
    # np.save("std.npy", std)
    # np.save("scalernpy", scaler.mean_.astype(np.float32))

    def output_transform( outputs):
        return outputs * std + scaler.mean_.astype(np.float32)           

    net.apply_output_transform(output_transform)
# andom_uniform
    model = dde.Model(data, net)
    model.compile("adam", lr=0.0001,
        loss="mse",
        metrics=["l2 relative error"], 
        decay=("inverse time", 1, 1e-4))
    checker = dde.callbacks.ModelCheckpoint(
        dataframe+"/model.ckpt", save_better_only=False, period=200
    )
    losshistory, train_state = model.train(epochs=30000, batch_size=None,display_every=50,callbacks=[checker],model_save_path=dataframe
    )  
    dde.saveplot(losshistory, train_state,issave=True,isplot=True,loss_fname=dataframe+"/loss.dat")



if __name__ == "__main__":
   
       
    main()

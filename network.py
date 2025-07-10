import tensorflow as tf

def network(problem, m, N_points):
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
        branch = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(87, 87, 4800)),
        tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
    # Block 1
        tf.keras.layers.Reshape((87, 87, 128, 1)),  # Reshape to 5D for Conv3D
        tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 5), strides=(1, 1, 2), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),


    # Block 2
        tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 5), strides=(1, 1, 2), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Block 3
        tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Block 4
        tf.keras.layers.Conv3D(512, kernel_size=(3, 3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),

    # Block 4
        tf.keras.layers.Conv3D(1024, kernel_size=(3, 3, 3), strides=1, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling3D(),  # 得到全局特征

    # Dense projection to latent space
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200)  # 输出 latent vector
])

        branch.summary()
        branch = [m, branch]
        trunk_1 = [3, 128, 256, 256, 200]
        trunk_2 = [3, 128, 256, 256, 200]
        dot = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(87 * 24, 2)),
                tf.keras.layers.Flatten(),
            ]
        )
        dot.summary()
        dot = [0, dot]
        return branch, trunk_1, trunk_2, dot

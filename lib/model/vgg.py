from keras import layers
import keras


def create_vgg_11(num_classes, input_shape=(224, 224, 3)) -> keras.Sequential:
    return keras.Sequential(
        [
            ### Conv block 1 ##
            layers.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 2 ##
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 3 ##
            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 4 ##
            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 5 ##
            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Output block 1 ##
            layers.Flatten(),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(4096, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

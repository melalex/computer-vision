from keras import layers
import keras
import keras_cv


def create_vgg_11(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    return keras.Sequential(
        [
            ### Conv block 1 ###
            layers.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 2 ###
            layers.Conv2D(128, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 3 ###
            layers.Conv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 4 ###
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 5 ###
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Output block ###
            layers.Flatten(),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
            layers.Softmax(),
        ]
    )

def create_depthwise_separable_vgg_11(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    return keras.Sequential(
        [
            ### Conv block 1 ###
            layers.Input(shape=input_shape),
            layers.SeparableConv2D(64, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 2 ###
            layers.SeparableConv2D(128, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 3 ###
            layers.SeparableConv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.SeparableConv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 4 ###
            layers.SeparableConv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.SeparableConv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Conv block 5 ###
            layers.SeparableConv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.SeparableConv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            ### Output block ###
            layers.Flatten(),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
            layers.Softmax(),
        ]
    )

def create_squeeze_and_excitation_vgg_11(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    return keras.Sequential(
        [
            ### Conv block 1 ###
            layers.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            keras_cv.layers.SqueezeAndExcite2D(64),
            ### Conv block 2 ###
            layers.Conv2D(128, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            keras_cv.layers.SqueezeAndExcite2D(128),
            ### Conv block 3 ###
            layers.Conv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(256, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            keras_cv.layers.SqueezeAndExcite2D(256),
            ### Conv block 4 ###
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            keras_cv.layers.SqueezeAndExcite2D(512),
            ### Conv block 5 ###
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.Conv2D(512, kernel_size=(3, 3), padding="same"),
            layers.BatchNormalization(),
            layers.ELU(),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            keras_cv.layers.SqueezeAndExcite2D(512),
            ### Output block ###
            layers.Flatten(),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(4096),
            layers.ELU(),
            layers.Dropout(0.5),
            layers.Dense(num_classes),
            layers.Softmax(),
        ]
    )


def create_attention_vgg_11(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    input = layers.Input(shape=input_shape)

    ### Conv block 1 ###
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same")(input)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    ### Conv block 2 ###
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    ### Conv block 3 ###
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    ### Conv block 4 ###
    x = layers.Conv2D(512, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    ### Conv block 5 ###
    x = layers.Conv2D(512, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.Conv2D(512, kernel_size=(3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    ### Attention block ###
    x = layers.Attention()([x, x])

    ### Output block ###
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation="elu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation="elu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=input, outputs=x)

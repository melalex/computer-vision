from keras import layers
import keras


def create_inception_res_net_v2(num_classes, input_shape=(299, 299, 3), scale=0.1):
    input = layers.Input(input_shape)

    x = __stem(input)

    # 10 x Inception Resnet A
    for _ in range(10):
        x = __inception_res_net_a(x, scale=scale)

    # Reduction A
    x = __reduction_a(x)

    # 20 x Inception Resnet B
    for _ in range(20):
        x = __inception_resnet_b(x, scale=scale)

    # Reduction Resnet B
    x = __reduction_b(x)

    # 10 x Inception Resnet C
    for _ in range(10):
        x = __inception_resnet_c(x, scale=scale)

    # Average Pooling
    x = layers.AveragePooling2D((8, 8))(x)

    # Dropout
    x = layers.Dropout(0.8)(x)
    x = layers.Flatten()(x)

    # Output
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(input, outputs=out, name="Inception-ResNet-v2")

    return model


def __stem(input):
    c = layers.Conv2D(32, 3, activation="relu", strides=(2, 2))(input)
    c = layers.Conv2D(32, 3, activation="relu")(c)
    c = layers.Conv2D(64, 3, activation="relu", padding="same")(c)

    c1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(c)
    c2 = layers.Conv2D(96, 3, activation="relu", strides=(2, 2))(c)

    m = layers.Concatenate(axis=-1)([c1, c2])

    c1 = layers.Conv2D(64, 1, activation="relu", padding="same")(m)
    c1 = layers.Conv2D(96, 3, activation="relu")(c1)

    c2 = layers.Conv2D(64, 1, activation="relu", padding="same")(m)
    c2 = layers.Conv2D(64, (7, 1), activation="relu", padding="same")(c2)
    c2 = layers.Conv2D(64, (1, 7), activation="relu", padding="same")(c2)
    c2 = layers.Conv2D(96, 3, activation="relu", padding="valid")(c2)

    m2 = layers.Concatenate(axis=-1)([c1, c2])

    p1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(m2)
    p2 = layers.Conv2D(192, 3, activation="relu", strides=(2, 2))(m2)

    m3 = layers.Concatenate(axis=-1)([p1, p2])
    m3 = layers.BatchNormalization(axis=-1)(m3)

    return layers.Activation("relu")(m3)


def __inception_res_net_a(input, scale=1):
    b1 = layers.Conv2D(32, 1, activation="relu", padding="same")(input)

    b2 = layers.Conv2D(32, 1, activation="relu", padding="same")(input)
    b2 = layers.Conv2D(32, 3, activation="relu", padding="same")(b2)

    b3 = layers.Conv2D(32, 1, activation="relu", padding="same")(input)
    b3 = layers.Conv2D(48, 3, activation="relu", padding="same")(b3)
    b3 = layers.Conv2D(64, 3, activation="relu", padding="same")(b3)

    b_merge = layers.Concatenate(axis=-1)([b1, b2, b3])

    b_conv = layers.Conv2D(384, 1, padding="same")(b_merge)
    b_conv = layers.Lambda(lambda x: x * scale)(b_conv)

    out = layers.Add()([input, b_conv])
    out = layers.BatchNormalization(axis=-1)(out)
    out = layers.Activation("relu")(out)

    return out


def __inception_resnet_b(input, scale=1):
    b1 = layers.Conv2D(192, 1, activation="relu", padding="same")(input)

    b2 = layers.Conv2D(128, 1, activation="relu", padding="same")(input)
    b2 = layers.Conv2D(160, (1, 7), activation="relu", padding="same")(b2)
    b2 = layers.Conv2D(192, (7, 1), activation="relu", padding="same")(b2)

    b_merge = layers.Concatenate(axis=-1)([b1, b2])

    b_conv = layers.Conv2D(1152, 1, padding="same")(b_merge)
    b_conv = layers.Lambda(lambda x: x * scale)(b_conv)

    out = layers.Add()([input, b_conv])
    out = layers.BatchNormalization(axis=-1)(out)
    out = layers.Activation("relu")(out)

    return out


def __inception_resnet_c(input, scale=1):
    b1 = layers.Conv2D(192, 1, activation="relu", padding="same")(input)

    b2 = layers.Conv2D(192, 1, activation="relu", padding="same")(input)
    b2 = layers.Conv2D(224, (1, 3), activation="relu", padding="same")(b2)
    b2 = layers.Conv2D(256, (3, 1), activation="relu", padding="same")(b2)

    b_merge = layers.Concatenate(axis=-1)([b1, b2])

    b_conv = layers.Conv2D(2144, 1, padding="same")(b_merge)
    b_conv = layers.Lambda(lambda x: x * scale)(b_conv)

    out = layers.Add()([input, b_conv])
    out = layers.BatchNormalization(axis=-1)(out)
    out = layers.Activation("relu")(out)

    return out


def __reduction_a(input):
    r1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = layers.Conv2D(384, 3, activation="relu", strides=(2, 2))(input)

    r3 = layers.Conv2D(256, 1, activation="relu", padding="same")(input)
    r3 = layers.Conv2D(256, 3, activation="relu", padding="same")(r3)
    r3 = layers.Conv2D(384, 3, activation="relu", strides=(2, 2))(r3)

    m = layers.Concatenate(axis=-1)([r1, r2, r3])
    m = layers.BatchNormalization(axis=1)(m)
    m = layers.Activation("relu")(m)

    return m


def __reduction_b(input):
    r1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(input)

    r2 = layers.Conv2D(256, 1, activation="relu", padding="same")(input)
    r2 = layers.Conv2D(384, 3, activation="relu", strides=(2, 2))(r2)

    r3 = layers.Conv2D(256, 1, activation="relu", padding="same")(input)
    r3 = layers.Conv2D(288, 3, activation="relu", strides=(2, 2))(r3)

    r4 = layers.Conv2D(256, 1, activation="relu", padding="same")(input)
    r4 = layers.Conv2D(288, 3, activation="relu", padding="same")(r4)
    r4 = layers.Conv2D(320, 3, activation="relu", strides=(2, 2))(r4)

    m = layers.Concatenate(axis=-1)([r1, r2, r3, r4])
    m = layers.BatchNormalization(axis=-1)(m)
    m = layers.Activation("relu")(m)

    return m

import keras

from keras import layers


def create_u_net(input_shape, num_classes=1):
    inputs = layers.Input(input_shape)

    ### Encoder ###
    x, e1 = encoder_block(inputs, 64)
    x, e2 = encoder_block(x, 128)
    x, e3 = encoder_block(x, 256)
    x, e4 = encoder_block(x, 512)

    ### Bottleneck ###
    b = layers.Conv2D(1024, 3, activation="relu", padding="same")(x)
    b = layers.Conv2D(1024, 3, activation="relu", padding="same")(b)

    ### Decoder ###
    d1 = decoder_block(b, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)

    #### Output ###
    outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(d4)

    return keras.Model(inputs=inputs, outputs=outputs, name="U-Net")


def encoder_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)

    return layers.MaxPool2D(pool_size=(2, 2), strides=2)(x), x


def decoder_block(inputs, encoded, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(
        inputs
    )
    x = layers.Concatenate()([encoded, x])

    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(num_filters, (3, 3), activation="relu", padding="same")(x)

    return x

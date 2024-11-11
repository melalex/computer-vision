from keras import layers
import keras


def create_res_net(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    # Step 1 (Setup Input Layer)
    x_input = keras.layers.Input(input_shape)
    x = keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)

    # Step 4 End Dense Network
    x = keras.layers.AveragePooling2D((2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=x_input, outputs=x)


def create_plain_net(num_classes, input_shape=(224, 224, 3)) -> keras.Model:
    # Step 1 (Setup Input Layer)
    x_input = keras.layers.Input(input_shape)
    x = keras.layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = keras.layers.Conv2D(64, kernel_size=7, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64

    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = plain_identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = plain_convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = plain_identity_block(x, filter_size)

    # Step 4 End Dense Network
    x = keras.layers.AveragePooling2D((2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=x_input, outputs=x)


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    # Layer 2
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = layers.Add()([x, x_skip])
    x = layers.ReLU()(x)
    return x


def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = keras.layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.ReLU()(x)
    # Layer 2
    x = keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = keras.layers.Conv2D(filter, (1, 1), strides=(2, 2))(x_skip)
    # Add Residue
    x = keras.layers.Add()([x, x_skip])
    x = keras.layers.ReLU()(x)
    return x


def plain_identity_block(x, filter):
    # copy tensor to variable called x_skip
    # Layer 1
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.ReLU()(x)
    # Layer 2
    x = layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = layers.ReLU()(x)
    return x


def plain_convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    # Layer 1
    x = keras.layers.Conv2D(filter, (3, 3), padding="same", strides=(2, 2))(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.ReLU()(x)
    # Layer 2
    x = keras.layers.Conv2D(filter, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization(axis=3)(x)
    # Add Residue
    x = keras.layers.ReLU()(x)
    return x

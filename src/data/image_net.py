from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras import datasets, utils

from src.data.util.numpy_data_generator import NumpyDataGenerator
from src.visualization.image import sample_image_dataset

import tensorflow_datasets as tfds


def create_image_net_data_generator(
    batch_size: int, test_split: float, validation_split: float, size: tuple[int, int] = None
) -> tuple[NumpyDataGenerator, NumpyDataGenerator, NumpyDataGenerator, int, int]:
    all_x, all_y = tfds.load("imagenet2012", shuffle_files=True)

    all_train_x, all_train_y, test_x, test_y = train_test_split(
        all_x, all_y, test_size=test_split
    )

    train_x, valid_x, train_y, valid_y = train_test_split(
        all_train_x, all_train_y, test_size=validation_split
    )

    train_x = train_x / 255
    valid_x = valid_x / 255
    test_x = test_x / 255

    if size is not None:
        train_x = tf.image.resize(train_x, size=size).numpy()
        valid_x = tf.image.resize(valid_x, size=size).numpy()
        test_x = tf.image.resize(test_x, size=size).numpy()

    num_classes = tf.unique(train_y).y.shape[0]
    input_shape = train_x.shape[1:]

    sample_image_dataset(train_x, train_y)

    train_y = utils.to_categorical(train_y, num_classes)
    valid_y = utils.to_categorical(valid_y, num_classes)
    test_y = utils.to_categorical(test_y, num_classes)

    return (
        NumpyDataGenerator(train_x, train_y, batch_size),
        NumpyDataGenerator(valid_x, valid_y, batch_size),
        NumpyDataGenerator(test_x, test_y, batch_size),
        num_classes,
        input_shape,
    )

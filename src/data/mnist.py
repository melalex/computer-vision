from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras import datasets, utils

from src.data.util.numpy_data_generator import NumpyDataGenerator
from src.visualization.image import sample_image_dataset


def create_mnist_data_generator(
    batch_size: int, validation_split: float, size: tuple[int, int] = None
) -> tuple[NumpyDataGenerator, NumpyDataGenerator, NumpyDataGenerator, int, int]:
    (all_x, all_y), (test_x, test_y) = datasets.mnist.load_data()

    train_x, valid_x, train_y, valid_y = train_test_split(
        all_x, all_y, test_size=validation_split
    )

    train_x = tf.expand_dims(train_x / 255, -1)
    valid_x = tf.expand_dims(valid_x / 255, -1)
    test_x = tf.expand_dims(test_x / 255, -1)

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

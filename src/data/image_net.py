import tensorflow_datasets as tfds
import tensorflow as tf
from keras import utils

NUM_IMAGE_NET_CLASSES = 10
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


def create_image_net_data_generator(batch_size: int):

    def normalize_img(image, label):
        norm = tf.cast(image, tf.float32) / 255.0
        norm = tf.image.resize(norm, size=(256, 256))
        norm = tf.image.random_crop(norm, size=(224, 224, 3))
        norm = (norm - IMAGE_NET_MEAN) / IMAGE_NET_STD

        return norm, tf.one_hot(label, NUM_IMAGE_NET_CLASSES)

    train_ds, val_ds = tfds.load(
        "imagenette/320px-v2",
        split=["train", "validation"],
        as_supervised=True,
    )

    train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    shape = (224, 224, 3)

    return (train_ds, val_ds, shape, NUM_IMAGE_NET_CLASSES)

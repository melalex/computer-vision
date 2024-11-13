from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def sample_image_dataset(
    x: tf.Tensor,
    y: tf.Tensor,
    labels_map: dict[int, str] = {},
    cols: int = 3,
    rows: int = 3,
):
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, cols * rows + 1):
        sample_idx = np.random.random_integers(len(x), size=(1,)).item()
        img = x[sample_idx]
        label = y[sample_idx].item()

        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label] if label in labels_map else label)
        plt.axis("off")
        plt.imshow(img)

    plt.show()


def sample_image_and_mask(
    x: np.array, y: np.array, count: int, y_hat: Optional[np.array] = None
):
    width = 14 if y_hat is not None else 7
    cols = 3 if y_hat is not None else 2
    figure = plt.figure(figsize=(width, 20))

    for i in range(1, cols * count + 1, cols):
        sample_idx = np.random.randint(len(x))
        img = x[sample_idx]
        mask = y[sample_idx]

        figure.add_subplot(count, cols, i)
        plt.title(sample_idx)
        plt.axis("off")
        plt.imshow(img, interpolation="nearest")

        figure.add_subplot(count, cols, i + 1)
        plt.title(sample_idx)
        plt.axis("off")
        plt.imshow(mask, interpolation="nearest")

        if y_hat is not None:
            mask_hat = y_hat[sample_idx]

            figure.add_subplot(count, cols, i + 2)
            plt.title(sample_idx)
            plt.axis("off")
            plt.imshow(mask_hat, interpolation="nearest")

    plt.show()

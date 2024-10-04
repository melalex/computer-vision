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

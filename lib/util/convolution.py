import numpy as np
import tensorflow as tf


SOBEL_X = tf.convert_to_tensor([[1, 0, -1], [2, 0, -1], [1, 0, -1]], dtype=tf.float32)
SOBEL_Y = tf.convert_to_tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=tf.float32)


def conv(
    target: np.array,
    kernel: np.array,
    striding: tuple[int, int] = (1, 1),
    padding: list[tuple[int, int]] = [(0, 0), (0, 0)],
):
    padded = np.pad(target, padding, mode="constant")

    stride_height, stride_width = striding

    padded_height = padded.shape[0]
    padded_width = padded.shape[1]
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    result_height = int((padded_height - kernel_width) / stride_height + 1)
    result_width = int((padded_width - kernel_height) / stride_width + 1)

    result = np.zeros((result_height, result_width))

    for i in range(result_height):
        for j in range(result_width):
            i_target = i * stride_height
            j_target = j * stride_width

            it = padded[
                i_target : i_target + kernel_height,
                j_target : j_target + kernel_width,
            ]

            result[i, j] = np.sum(it * kernel)

    return result


def apply_filter(image, filter):
    image_r = image[:, :, 0]
    image_g = image[:, :, 1]
    image_b = image[:, :, 2]

    return tf.stack(
        [
            apply_filter_mono(image_r, filter),
            apply_filter_mono(image_g, filter),
            apply_filter_mono(image_b, filter),
        ],
        axis=2,
    )


def apply_filter_mono(image, filter):
    expanded_image = image[tf.newaxis, :, :, tf.newaxis]
    expanded_filter = filter[:, :, tf.newaxis, tf.newaxis]

    res_image = tf.nn.conv2d(
        expanded_image, expanded_filter, strides=[1, 1], padding="SAME"
    )

    return tf.cast(tf.squeeze(res_image), tf.uint8)


def create_box_filter(size):
    result = np.ones((size, size))
    return result / np.sum(result)


def unsharp_masking(image, blurred_image, amount):
    float_image = tf.cast(image, dtype=tf.float16)
    float_blurred_image = tf.cast(blurred_image, dtype=tf.float16)
    res_image = float_image + (float_image - float_blurred_image) * amount

    return tf.cast(res_image, tf.uint8)

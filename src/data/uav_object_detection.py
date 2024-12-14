import logging
import keras
import keras_cv
import xml.etree.ElementTree as ET
from pathlib import Path

import tensorflow as tf
from src.data.util.dataset_util import download_dataset, unzip_file
from src.definitions import EXTERNAL_DATA_FOLDER


def create_uav_object_detection_ds(
    batch_size: int = 64,
    test_ratio: float = 0.1,
    img_size: tuple[int, int] = (256, 256),
    logger: logging.Logger = logging.getLogger(__name__),
) -> Path:
    archive = download_dataset(
        "dasmehdixtr",
        "drone-dataset-uav",
        EXTERNAL_DATA_FOLDER,
        logger,
    )

    dir = unzip_file(archive, logger)

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xyxy"),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format="xyxy"
            ),
            keras_cv.layers.Resizing(
                height=img_size[0],
                width=img_size[1],
                pad_to_aspect_ratio=True,
                bounding_box_format="xyxy",
            ),
        ]
    )

    resizing = keras_cv.layers.Resizing(
        height=img_size[0],
        width=img_size[1],
        pad_to_aspect_ratio=True,
        bounding_box_format="xyxy",
    )

    ds = load_uav_ds(dir)
    ds = ds.shuffle(batch_size * 4)
    ds = ds.ragged_batch(batch_size, drop_remainder=True)

    ds_size = len(ds)

    test_size = int(test_ratio * ds_size)

    test_ds = ds.take(test_size)
    test_ds = test_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = ds.skip(test_size)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, test_ds


def load_uav_ds(raw_path: Path) -> tf.data.Dataset:
    dir_path = raw_path / "dataset_xml_format" / "dataset_xml_format"

    class_ids = ["drone"]
    class_mapping = dict(zip(range(len(class_ids)), class_ids))

    def read_label(stem):
        file_name = stem + ".xml"
        path = dir_path / file_name
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        classes = []
        for obj in root.iter("object"):
            cls = obj.find("name").text
            classes.append(cls)

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        class_ids = [
            list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
            for cls in classes
        ]
        return class_ids, boxes

    def load_dataset(image_path, classes, bbox):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        bbox = bbox.to_tensor()

        bounding_boxes = {
            "classes": classes,
            "boxes": bbox,
        }

        return {"images": image, "bounding_boxes": bounding_boxes}

    def create_img_path(stem):
        file_name = stem + ".jpg"
        path = dir_path / file_name
        return str(path.absolute())

    samples = [Path(it).stem for it in dir_path.glob("*.jpg")]
    image_paths = [create_img_path(it) for it in samples]
    labels = [read_label(it) for it in samples]
    classes, boxes = zip(*labels)

    image_paths = tf.ragged.constant(image_paths)
    classes = tf.ragged.constant(classes)
    boxes = tf.ragged.constant(boxes)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, classes, boxes))
    ds = ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

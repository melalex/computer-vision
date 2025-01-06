import keras
import keras_cv
import xml.etree.ElementTree as ET
from pathlib import Path

import tensorflow as tf
import yaml

from src.conf.cv_config import CvConfig
from roboflow import Roboflow

from src.definitions import EXTERNAL_DATA_FOLDER


def download_udacity_dataset(conf: CvConfig):
    rf = Roboflow(api_key=conf.roboflow.apiKey)
    project = rf.workspace("roboflow-gw7yv").project("self-driving-car")
    version = project.version(2)
    location = EXTERNAL_DATA_FOLDER / "self-driving-car"
    return Path(version.download("voc", location=str(location.absolute())).location)


def create_udacity_ds(
    path: Path,
    batch_size: int = 64,
    test_ratio: float = 0.1,
    img_size: tuple[int, int] = (256, 256),
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    box_format = "xyxy"

    class_ids = [
        "biker",
        "car",
        "pedestrian",
        "trafficLight",
        "trafficLight-Green",
        "trafficLight-GreenLeft",
        "trafficLight-Red",
        "trafficLight-RedLeft",
        "trafficLight-Yellow",
        "trafficLight-YellowLeft",
        "truck",
    ]

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(
                mode="horizontal", bounding_box_format=box_format
            ),
            keras_cv.layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format=box_format
            ),
            keras_cv.layers.Resizing(
                height=img_size[0],
                width=img_size[1],
                pad_to_aspect_ratio=True,
                bounding_box_format=box_format,
            ),
        ]
    )

    resizing = keras_cv.layers.Resizing(
        height=img_size[0],
        width=img_size[1],
        pad_to_aspect_ratio=True,
        bounding_box_format=box_format,
    )

    ds = load_yolo_ds(path / "export", class_ids)
    ds = ds.shuffle(batch_size * 4)
    ds = ds.ragged_batch(batch_size, drop_remainder=True)

    ds_size = len(ds)

    test_size = int(test_ratio * ds_size)

    test_ds = ds.take(test_size)
    test_ds = test_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = ds.skip(test_size)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, test_ds, class_ids, box_format


def load_yolo_ds(dir_path: Path, all_classes: list[str]) -> tf.data.Dataset:
    image_path = dir_path
    label_path = dir_path

    class_mapping = dict(zip(range(len(all_classes)), all_classes))

    def read_label(stem):
        file_name = stem + ".xml"
        path = label_path / file_name

        boxes = []
        classes = []

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
        path = image_path / file_name
        return str(path.absolute())

    samples = [Path(it).stem for it in image_path.glob("*.jpg")]
    image_paths = [create_img_path(it) for it in samples]
    labels = [read_label(it) for it in samples]
    classes, boxes = zip(*labels)

    image_paths = tf.ragged.constant(image_paths)
    classes = tf.ragged.constant(classes)
    boxes = tf.ragged.constant(boxes)

    ds = tf.data.Dataset.from_tensor_slices((image_paths, classes, boxes))
    ds = ds.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

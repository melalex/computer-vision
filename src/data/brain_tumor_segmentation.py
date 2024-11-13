import logging
from pathlib import Path

import cv2
import numpy as np
from src.data.util.dataset_util import download_dataset, unzip_file
from src.definitions import EXTERNAL_DATA_FOLDER
from pycocotools.coco import COCO
from skimage.draw import polygon


def create_brain_tumor_segmentation_ds(
    size: tuple[int, int] = (256, 256),
    logger: logging.Logger = logging.getLogger(__name__),
) -> tuple[
    tuple[np.array, np.array], tuple[np.array, np.array], tuple[np.array, np.array]
]:

    archive = download_dataset(
        "pkdarabi",
        "brain-tumor-image-dataset-semantic-segmentation",
        EXTERNAL_DATA_FOLDER,
        logger,
    )
    data_path = unzip_file(archive, logger)

    return (
        load_split(data_path / "train", size),
        load_split(data_path / "valid", size),
        load_split(data_path / "test", size),
    )


def load_split(path: Path, size):
    coco = COCO(path / "_annotations.coco.json")

    def load_mask(img_id):
        img_info = coco.loadImgs(img_id)[0]
        height = img_info["height"]
        width = img_info["width"]

        mask = np.zeros((height, width), dtype=np.uint8)
        annotation_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(annotation_ids)

        for annotation in annotations:
            mask += coco.annToMask(annotation)

        return cv2.resize(mask, size)

    def load_img(img_id):
        img_info = coco.loadImgs(img_id)[0]
        img_path = path / img_info["file_name"]

        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0

        return img

    imgs = np.array([load_img(it) for it in coco.getImgIds()])
    masks = np.array([load_mask(it) for it in coco.getImgIds()])

    imgs = np.expand_dims(imgs, axis=-1)
    masks = np.expand_dims(masks, axis=-1)

    return imgs, masks

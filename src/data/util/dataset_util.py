import logging
import zipfile
import kaggle
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from pathlib import Path


def download_dataset(owner: str, name: str, dest: Path, logger: logging.Logger) -> Path:
    dest.mkdir(parents=True, exist_ok=True)

    file_path = dest / f"{name}.zip"

    if file_path.is_file():
        logger.info(
            "Found [ %s ] dataset in [ %s ]. Skipping download...", name, file_path
        )
    else:
        logger.info("Downloading [ %s ] dataset to [ %s ]", name, file_path)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset=f"{owner}/{name}", path=dest)

    return file_path


def unzip_file(archive: Path, logger: logging.Logger) -> Path:
    archive_name = archive.stem
    dest_file = archive.parent / archive_name

    if dest_file.exists():
        logger.info("[ %s ] is already unzipped. Skipping ...", dest_file)
    else:
        logger.info("Unzipping [ %s ] to [ %s ]", archive, dest_file)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(dest_file)

    return dest_file


def extract_xy(dataset: tf.data.Dataset):
    xy = tfds.as_numpy(dataset)

    x = []
    y = []

    for img, lbl in xy:
        x.extend(img)
        y.extend(lbl)

    x = np.array(x)
    y = np.argmax(np.array(y), axis=1)

    return x, y

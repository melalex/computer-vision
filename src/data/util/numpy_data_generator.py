import numpy as np
from keras.utils import Sequence


class NumpyDataGenerator(Sequence):
    __x: np.array
    __y: np.array
    __batch_size: int

    def __init__(self, x: np.array, y: np.array, batch_size: int):
        self.__x = x
        self.__y = y
        self.__batch_size = batch_size

    def __len__(self) -> int:
        return int(np.ceil(len(self.__x) / float(self.__batch_size)))

    def __getitem__(self, idx) -> tuple[np.array, np.array]:
        batch_x = self.__x[idx * self.__batch_size : (idx + 1) * self.__batch_size]
        batch_y = self.__y[idx * self.__batch_size : (idx + 1) * self.__batch_size]

        return batch_x, batch_y

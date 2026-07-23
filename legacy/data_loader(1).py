from glob import glob
import numpy as np
import os
import random
from pandas import read_csv


class DataLoader:
    def __init__(self, dataset_name, img_res=(32, 32)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False, is_pred=False):
        data_type = "train" if not is_testing else "test"
        if is_pred:
            batch_images = ['test_images/' + x for x in os.listdir('test_images/')]
        else:
            path = glob('E:/data/southgan/hrpic_pca/*')
            batch_images = np.random.choice(path, size=batch_size)
        data_hr = []
        data_lr = []

        # data_path = ['E:/data/southgan/hrpic/' + x for x in os.listdir('E:/data/southgan/hrpic/')]
        for img_path in batch_images:
            img = read_csv(img_path, header=None)
            img = img.values
            img = np.reshape(img, [32, 32, 3])
            data_hr.append(img[:, :, 0:3])
            lr = np.zeros(shape=(8, 8, 3))

            for m in range(0, 8):
                for n in range(0, 8):
                    i = random.randint(0, 3)
                    if i > 1:
                        ii = random.randint(0, 1)
                    else:
                        ii = random.randint(2, 3)
                    j = random.randint(0, 1)
                    jj = random.randint(2, 3)
                    lr[m, n] = (img[4 * m + i, 4 * n + j, :] + img[4 * m + ii, 4 * n + jj, :]) / 2
            data_lr.append(lr[:, :, 0:3])

        return np.asarray(data_hr, np.float32), np.asarray(data_lr, np.float32)

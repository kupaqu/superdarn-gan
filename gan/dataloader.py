import os
import glob
import random
import datetime
import warnings
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm

class DataLoader:
    def __init__(self, paths, shuffle=True, beam=6):
        self.shuffle = shuffle
        self.data = {}

        # загрузка всего датасета в память
        for path in paths:
            for root, _, files in os.walk(path):
                for name in files:
                    filename = name.split('.')
                    if filename[4] == str(beam):
                        key = (filename[0] + filename[1][:2], filename[4]) # ключ – кортеж вида (дата и час, луч)
                        arr = np.load(os.path.join(root, name))
                        self.data[key] = arr

    def __call__(self):
        target_datetime = list(self.data.keys())
        if self.shuffle:
            random.shuffle(target_datetime)
        print('Total files:', len(target_datetime))

        dataset_size = 0
        missed_size = 0

        # итерация по ключам в словаре self.data, где ключи – название файла
        for key in target_datetime:
            seq = self.__getSequence(key) # ключи исторических данных
            arrays = []
            missData = False

            for item in seq:
                item = tuple(item)
                if item in self.data:
                    arrays.append(self.data[item])

                # некоторые исторические данные могут отсутствовать
                else:
                    missData = True
                    break

            # если есть пропуски, то пропускаем пример
            if missData:
                missed_size += 1
                continue
            else:
                dataset_size += 1
                x = np.concatenate(arrays, axis=1)
                y = self.data[key]
                print('dataset:', dataset_size, 'missed:', missed_size, end='\r')
                print()
                yield np.concatenate([x[:,:,2:3], x[:,:,1:2]], axis=-1), y[:,:,2:3]*y[:,:,1:2]

    def __getSequence(self, key):
        timestamp, beam = key
        filename_datetime = datetime.datetime.strptime(timestamp, '%Y%m%d%H')

        # список массивов за день до целевого массива
        dayBefore = []
        for i in range(24, 0, -2):
            hoursBefore = ((filename_datetime-datetime.timedelta(hours=i)).strftime('%Y%m%d%H'), beam)
            dayBefore.append(hoursBefore)

        # тот же час, но за неделю до целевого массива
        weekBeforeInThatHour = []
        for i in range(7, 1, -1):
            thatHour = ((filename_datetime-datetime.timedelta(days=i)).strftime('%Y%m%d%H'), beam)
            weekBeforeInThatHour.append(thatHour)

        # return weekBeforeInThatHour

        return dayBefore + weekBeforeInThatHour
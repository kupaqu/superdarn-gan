import os
import datetime
import bz2
import numpy as np
import pydarnio

from tqdm import tqdm

class DataConverter:

    def __init__(self,
                 src_dir: str,  # папка с исходными данными
                 dst_dir: str,  # папка с конвертированными данными
                 nrang=100,     # количество гейтов
                 reg_res=60,    # выходное разрешение
                 bmnum=16,      # количество лучей

                 keys=['pwr0',  # Lag zero power (actually SNR), estimated from voltage samples (not fitted)
                       'qflg',  # Quality of fit flag for ACF
                       'p_l',   # Power (actually SNR) from lambda fit of ACF
                       'p_s',   # Power (actually SNR) from sigma fit of ACF
                       'v',     # Velocity from fit of ACF
                       'w_l',   # Spectral width from lambda fit of ACF
                       'w_s']): # Spectral width from sigma fit of ACF

        # разрешение радара, у CVW – раз в минуту
        self.radar_res = 120

        # абсолютный путь до папки с исходными данными
        self.src_dir = os.path.abspath(src_dir)

        # абсолютный путь до папки с конвертированными данными
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        self.dst_dir = os.path.abspath(dst_dir)

        # индекс последнего разделителя в абсолютном пути до папки с исходными данными
        self.src_prefix = len(self.src_dir) + len(os.path.sep)

        self.nrang = nrang
        self.reg_res = reg_res
        self.bmnum = bmnum
        self.keys = keys

    # копирует структуру папки с исходными данными в папку с конвертированными данными
    def __copy_dir_tree(self):

        for root, dirs, _ in os.walk(self.src_dir):
            for dir in dirs:
                dirpath = os.path.join(self.dst_dir, root[self.src_prefix:], dir)
                try:
                    os.mkdir(dirpath)
                except FileExistsError:
                    continue


    def convert(self):

        self.__copy_dir_tree() # создаем структуру папки

        # итерация по папкам месяцов
        for root, _, files in os.walk(self.src_dir):
            if len(files) > 0:
                print(f'Converting FITACF\'s to NumPy arrays at {root}')
                for file in tqdm(files):

                    # название исходного файла
                    src_filename = os.path.join(root, file)
                    if not src_filename.endswith('fitacf.bz2'):
                        continue

                    # название конвертированного файла
                    dst_filename = os.path.join(self.dst_dir, root[self.src_prefix:], file)[:-len('.fitacf.bz2')]

                    # открытие исходного файла на чтение
                    with bz2.open(src_filename) as fp:
                        fitacf_stream = fp.read()
                        try:
                            reader = pydarnio.SDarnRead(fitacf_stream, True)
                            records = reader.read_fitacf()

                            # итерация по лучам. сохраняем в отдельные файлы, т.к. на некоторых лучах могут отсутствовать наблюдения
                            for beam in range(self.bmnum):
                                is_valid, content = self.__get_content(records, beam)
                                if is_valid:
                                    np.save(dst_filename + f'.{beam}' + '.npy', content)

                        except pydarnio.exceptions.dmap_exceptions.EmptyFileError:
                            continue

    def __get_content(self, records, beam):

            timeseries = np.zeros(shape=(self.nrang, self.radar_res, len(self.keys))) # разметка пустого массива для заполнения
            timestamps = []

            for record in records:

                # отбор данных по лучу
                if record['bmnum'] == beam and record['channel'] == 0:

                    # метка по времени, которая используется для расчета индекса
                    timestamps.append(datetime.datetime(year=record['time.yr'],
                                                        month=record['time.mo'],
                                                        day=record['time.dy'],
                                                        hour=record['time.hr'],
                                                        minute=record['time.mt']))

                    # индекс наблюдения в массиве
                    if timestamps:
                        index = (timestamps[-1] - timestamps[0]).seconds // 60
                        if index >= self.radar_res: # иногда при одинаковой регулярности, разность между наблюдениями больше 1 минуты
                            break
                    else:
                        index = 0

                    # конвертация
                    if 'slist' in record:
                        for gate_index, gate in enumerate(record['slist']):
                            for key_index, key in enumerate(self.keys):
                                if key == 'pwr0':
                                    timeseries[:, index, key_index] = record[key]
                                else:
                                    timeseries[gate, index, key_index] = record[key][gate_index]

            # если не было ни одного наблюдения
            if len(timestamps) == 0:
                return (False, None)

            # ужатие до reg_res
            return (True, timeseries[:, ::self.radar_res//self.reg_res])
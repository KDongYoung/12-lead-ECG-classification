import sklearn.model_selection
from torch.utils.data import Dataset
import wfdb
import torch
import numpy as np
import os

from dbloader import *


class CpscDataset(Dataset):
    def __init__(self, root_dir, record_list, pre_processing):
        self.sampling_rate = 500
        self.cls_list = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        self.root_dir = root_dir
        self.record_list = record_list
        self.pre_processing = pre_processing

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        record_name = self.record_list[idx]
        record = np.load(f'{self.root_dir}/{record_name}.npy', allow_pickle=True).item()
        record['ecg'] = np.transpose(record['ecg'])
        label = np.zeros(len(self.cls_list))
        for beat in record['beats']:
            label[list(self.cls_list).index(beat)] = 1
        record['beats'] = label

        if self.pre_processing:
            record = self.pre_processing(record) 

        return [record['ecg'], record['beats'], record['id']]

    def pre_pre_processing(self, root_dir, save_dir, base_sec=10, max_sec=20):
        """
        1. Selecting a record by limited duration
        2. Normalization along all leads
        3. Zero-padding
        :param root_dir: Directory where .hea and .mat files located
        :param save_dir:
        :param base_sec:
        :param max_sec:
        :return:
        """
        raw_record_list = np.loadtxt(f'propose/RECORDS', delimiter=',', dtype=str)
        file_dir = root_dir
        os.makedirs(save_dir, exist_ok=True)

        pass_cnt = 0
        cls_cnt = [0] * len(self.cls_list)
        multi_label_cnt = 0
        record_list = []
        for record_name in raw_record_list:
            record = wfdb.rdrecord(f'{file_dir}/{record_name}')
            ecg = record.p_signal  # [time, channel]
            beats = record.comments[2][4:].split(',')
            # Discard a record over [sec_limit]
            if len(ecg) > max_sec * self.sampling_rate:
                pass_cnt += 1
            else:  # Selected record
                record_list.append(record_name)
                # Class count
                if len(beats) > 1:
                    multi_label_cnt += 1
                for beat in beats:
                    cls_cnt[list(self.cls_list).index(beat)] += 1
                # Pick center of record when it over base duration
                if len(ecg) > base_sec * self.sampling_rate:
                    diff = len(ecg) - base_sec * self.sampling_rate
                    gap = int(diff / 2)
                    ecg = ecg[gap: len(ecg) - gap, :]
                    if len(ecg) != base_sec * self.sampling_rate:
                        ecg = ecg[1:, :]
                # Normalization by mean and std
                ecg = (ecg - np.mean(ecg)) / np.std(ecg)
                # Padding when record lower than base duration
                if len(ecg) < base_sec * self.sampling_rate:
                    diff = base_sec * self.sampling_rate - len(ecg)
                    zeros = np.zeros([diff, ecg.shape[1]])
                    ecg = np.append(ecg, zeros, axis=0)
                # Save with signals and class information
                item = {'ecg': ecg, 'beats': beats, 'id': record_name.split('A')[-1]}
                np.save(f'{save_dir}/{record_name}', item)
        np.savetxt(f'{save_dir}/record', record_list, fmt='%s', delimiter=',')
        print(f'{len(raw_record_list) - pass_cnt} are selected, '
              f'{pass_cnt} records are discarded from {len(raw_record_list)} records')
        print(f'There are {multi_label_cnt} records containing multi-label')
        for cls_name, cnt in zip(self.cls_list, cls_cnt):
            print(f'{cls_name} : {cnt}')
        print(f'Total : {np.sum(cls_cnt)}')
        tr, te = sklearn.model_selection.train_test_split(record_list, test_size=.2, random_state=4)
        np.savetxt(f'{save_dir}/train', tr, fmt='%s', delimiter=',')
        np.savetxt(f'{save_dir}/val', te, fmt='%s', delimiter=',')
        print(f'# of train records={len(tr)}')
        print(f'# of val records={len(te)}')


class ToTensor(object):
    def __call__(self, sample):
        for key, value in sample.items():
            if key != 'id':
                sample[key] = torch.tensor(value, dtype=torch.float)
            else:
                sample[key] = torch.tensor(int(value), dtype=torch.int)

        return sample

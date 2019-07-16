import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from utils.util import bit_division

class WaveRNNDataset(Dataset):
    def __init__(self,
                 path,
                 upsample_factor=200,
                 normalize="maxmin",
                 local_condition=True,
                 global_condition=False):

        self.path = path
        self.metadata = self.get_metadata(path)

        '''
        assert normalize in ['cmvn', 'maxmin']
        self.normalize = normalize
        if normalize == "cmvn":
            self.calculate_cmvn()
        elif normalize == "maxmin":
            self.calculate_maxmin()
        '''

        self.upsample_factor = upsample_factor

        self.local_condition = local_condition
        self.global_condition = global_condition

    def calculate_maxmin(self, dimension=80):
        min_value_matrix = np.zeros((len(self.metadata), dimension))
        max_value_matrix = np.zeros((len(self.metadata), dimension))
        for (i, filename) in enumerate(self.metadata):
            data = np.load(os.path.join(self.path, 'mel', filename))
            temp_min = np.amin(data, axis=0)
            temp_max = np.amax(data, axis=0)

            min_value_matrix[i, ] = temp_min
            max_value_matrix[i, ] = temp_max
        min_vector = np.amin(min_value_matrix, axis=0)
        max_vector = np.amax(max_value_matrix, axis=0)

        vari = max_vector - min_vector
        vari[vari == 0] = 1

        self.mean = min_vector
        self.vari = vari
        np.savez('norm.npz', mean=self.mean, vari=self.vari)

    def calculate_cmvn(self):
        frame_number = 0
        for filename in self.metadata:
            data = np.load(os.path.join(self.path, 'mel', filename))
            if frame_number == 0:
                ex_feature = np.sum(data, axis=0)
                ex2_feature = np.sum(data**2, axis=0)
            else:
                ex_feature += np.sum(data, axis=0)
                ex2_feature += np.sum(data**2, axis=0)
            frame_number += len(data)
        self.mean = ex_feature / frame_number
        self.vari = np.sqrt(ex2_feature / frame_number - self.mean**2)
        self.vari[self.vari == 0] = 1
        np.savez('norm.npz', mean=self.mean, vari=self.vari)

    def __getitem__(self, index):

        sample = np.load(os.path.join(self.path, 'audio', self.metadata[index]))
        condition = np.load(os.path.join(self.path, 'mel', self.metadata[index]))

        length = min([len(sample), len(condition) * self.upsample_factor])

        sample = sample[: length]
        condition = condition[: length // self.upsample_factor , :]

        if self.local_condition:
            return sample, condition
        else:
            return sample

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self, path):
        with open(os.path.join(path, 'names.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        return metadata

class WaveRNNCollate(object):

    def __init__(self,
                 upsample_factor=200,
                 condition_window=18,
                 local_condition=True,
                 global_condition=False):

        self.upsample_factor = upsample_factor
        self.condition_window = condition_window
        self.local_condition = local_condition
        self.global_condition = global_condition

    def __call__(self, batch):
        return self._collate_fn(batch)

    def _collate_fn(self, batch):

       c_batch = []
       f_batch = []
       max_offsets = [x[1].shape[0] - self.condition_window for x in batch]
       c_offsets = [np.random.randint(0, offset) for offset in max_offsets]
       s_offsets = [offset * self.upsample_factor for offset in c_offsets]
       for (i, x) in enumerate(batch):
           c, f = bit_division(x[0][s_offsets[i] :
               s_offsets[i] + self.condition_window * self.upsample_factor])
           c_batch.append(c)
           f_batch.append(f)
       c_batch = np.stack(c_batch)
       f_batch = np.stack(f_batch)

       if self.local_condition:
           condition_batch = []
           for (i, x) in enumerate(batch):
               condition_batch.append(x[1][c_offsets[i] :
                   c_offsets[i] + self.condition_window])
           condition_batch = np.stack(condition_batch)

           return torch.LongTensor(c_batch), torch.LongTensor(f_batch), torch.FloatTensor(condition_batch)

       else:
           return torch.LongTensor(c_batch), torch.LongTensor(f_batch)

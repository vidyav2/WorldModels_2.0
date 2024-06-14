""" Some data loading utilities """
from bisect import bisect
from os import listdir
import os
from os.path import join, isdir
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np

"""class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, buffer_size=200, train=True):
        self._transform = transform

        # Find all .npz files in the root directory
        #self._files = [join(root, f) for f in os.listdir(root) if f.endswith('.npz')]
        self._files = [
            join(root, sd)
            for sd in listdir(root) ]
        
        if train:
            self._files = self._files[:-600]
        else:
            self._files = self._files[-600:]
        
        split_index = int(len(self._files) * 0.8)  # 80% for training, 20% for testing

        if train:
            self._files = self._files[:split_index]
        else:
            self._files = self._files[split_index:]

        print(f"{'Training' if train else 'Testing'} dataset size after split: {len(self._files)}")


        print("Found hello {} files in {}".format(len(self._files), root))
        # Print statement to confirm files are found
        print(f"Found {len(self._files)} files in {root}")

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        if len(self._files) == 0:
            self._buffer = []
            self._cum_size = [0]
            return
        print(f"Buffer loaded with {len(self._buffer)} items, Cumulative size: {self._cum_size}")

        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index = (self._buffer_index + self._buffer_size) % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        pbar = tqdm(total=len(self._buffer_fnames), bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] + self._data_per_sequence(len(data['observations']))]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        if not self._cum_size or len(self._files) == 0:
            return 0
        return self._cum_size[-1]

    def __getitem__(self, i):
        if len(self._files) == 0:
            raise IndexError("No data files available.")
        print(f"Accessing index {i}, file_index: {file_index}, seq_index: {seq_index}")

        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        # Implement based on your specific requirements
        pass

    def _data_per_sequence(self, data_length):
        # Implement based on your specific requirements
        pass
        calculated_value = data_length - 1  # Adjust based on your needs
        print(f"Data per sequence: {calculated_value}, Total data length: {data_length}")
        return calculated_value"""

from bisect import bisect
from os import listdir
from os.path import join
from tqdm import tqdm
import numpy as np
import torch.utils.data

class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, buffer_size=200, train=True):
        self._transform = transform
        self._files = []
        for root in root:
            # Ensure the root is a directory
            if os.path.isdir(root):
                self._files.extend([join(root, f) for f in os.listdir(root) if f.endswith('.npz')])
            else:
                print(f"Warning: {root} is not a directory.")


        # Find all files in the root directory
        #self._files = [join(root, sd) for sd in listdir(root)]

        # Splitting files between training and testing
        #split_index = int(len(self._files) * 0.8)  # 80% for training, 20% for testing
        """if train:
            self._files = self._files[:split_index]
        else:
            self._files = self._files[split_index:]"""

        
        if train:
            self._files = self._files[:-600]
        else:
            self._files = self._files[-600:]

        print(f"{'Training' if train else 'Testing'} dataset size after split: {len(self._files)}")
        print(f"Found {len(self._files)} files in {root}")

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        """ Loads next buffer """
        self._buffer_fnames = self._files[self._buffer_index:self._buffer_index + self._buffer_size]
        self._buffer_index += self._buffer_size
        self._buffer_index = self._buffer_index % len(self._files)
        self._buffer = []
        self._cum_size = [0]

        # progress bar
        pbar = tqdm(total=len(self._buffer_fnames),
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}')
        pbar.set_description("Loading file buffer ...")

        for f in self._buffer_fnames:
            with np.load(f) as data:
                self._buffer += [{k: np.copy(v) for k, v in data.items()}]
                self._cum_size += [self._cum_size[-1] +
                                   self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
            #return 0
        return self._cum_size[-1]

    def __getitem__(self, i):
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]

        print(f"Accessing index {i}, file_index: {file_index}, seq_index: {seq_index}")
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        # Implement based on your specific requirements
        pass

    def _data_per_sequence(self, data_length):
        # Implement based on your specific requirements
        calculated_value = data_length  # Adjust based on your needs
        print(f"Data per sequence: {calculated_value}, Total data length: {data_length}")
        return calculated_value


class RolloutObservationDataset(_RolloutDataset):
    def __init__(self, root, transform, buffer_size=200, train=True):
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = 1  # Assuming each observation is independent

    """def _get_data(self, data, seq_index):
        obs = data['observations'][seq_index]
        obs = self._transform(obs.astype(np.float32))
        return obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len"""
    
    def _data_per_sequence(self, data_length):
        return data_length
        
    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])


class RolloutSequenceDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of tuples (obs, action, reward, terminal, next_obs):
    - obs: (seq_len, *obs_shape)
    - actions: (seq_len, action_size)
    - reward: (seq_len,)
    - terminal: (seq_len,) boolean
    - next_obs: (seq_len, *obs_shape)

    NOTE: seq_len < rollout_len in moste use cases

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def __init__(self, root, seq_len, transform, buffer_size=200, train=True): # pylint: disable=too-many-arguments
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index+1:seq_index + self._seq_len + 1]
        action = action.astype(np.float32)
        reward, terminal = [data[key][seq_index+1:
                                      seq_index + self._seq_len + 1].astype(np.float32)
                            for key in ('rewards', 'terminals')]
        # data is given in the form
        # (obs, action, reward, terminal, next_obs)
        return obs, action, reward, terminal, next_obs

    def _data_per_sequence(self, data_length):
        return data_length - self._seq_len

class RolloutObservationDataset(_RolloutDataset): # pylint: disable=too-few-public-methods
    """ Encapsulates rollouts.

    Rollouts should be stored in subdirs of the root directory, in the form of npz files,
    each containing a dictionary with the keys:
        - observations: (rollout_len, *obs_shape)
        - actions: (rollout_len, action_size)
        - rewards: (rollout_len,)
        - terminals: (rollout_len,), boolean

     As the dataset is too big to be entirely stored in rams, only chunks of it
     are stored, consisting of a constant number of files (determined by the
     buffer_size parameter).  Once built, buffers must be loaded with the
     load_next_buffer method.

    Data are then provided in the form of images

    :args root: root directory of data sequences
    :args seq_len: number of timesteps extracted from each rollout
    :args transform: transformation of the observations
    :args train: if True, train data, else test
    """
    def _data_per_sequence(self, data_length):
        return data_length

    def _get_data(self, data, seq_index):
        return self._transform(data['observations'][seq_index])
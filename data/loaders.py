import os
from bisect import bisect
from os.path import join
from tqdm import tqdm
import torch
import torch.utils.data
import numpy as np

class _RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, roots, transform, buffer_size=200, train=True):
        self._transform = transform
        self._files = []

        # Ensure roots is a list of directories
        if isinstance(roots, str):
            roots = [roots]

        for root in roots:
            if os.path.isdir(root):
                for dirpath, _, filenames in os.walk(root):
                    for fname in filenames:
                        fpath = join(dirpath, fname)
                        if os.path.isfile(fpath) and fpath.endswith('.npz'):
                            self._files.append(fpath)

        print(f"Total files found: {len(self._files)}")

        if train:
            self._files = self._files[:-600]
        else:
            self._files = self._files[-600:]

        print(f"{'Training' if train else 'Testing'} dataset size after split: {len(self._files)}")

        if not self._files:
            raise ValueError("No data files found in the specified directories.")

        self._cum_size = None
        self._buffer = None
        self._buffer_fnames = None
        self._buffer_index = 0
        self._buffer_size = buffer_size

    def load_next_buffer(self):
        """ Loads next buffer """
        if len(self._files) == 0:
            raise ValueError("No data files found.")

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
                self._cum_size += [self._cum_size[-1] + self._data_per_sequence(data['rewards'].shape[0])]
            pbar.update(1)
        pbar.close()

    def __len__(self):
        if not self._cum_size:
            self.load_next_buffer()
        return self._cum_size[-1]

    def __getitem__(self, i):
        file_index = bisect(self._cum_size, i) - 1
        seq_index = i - self._cum_size[file_index]
        data = self._buffer[file_index]
        return self._get_data(data, seq_index)

    def _get_data(self, data, seq_index):
        pass

    def _data_per_sequence(self, data_length):
        pass

class RolloutSequenceDataset(_RolloutDataset):
    def __init__(self, root, seq_len, transform, buffer_size=200, train=True):
        super().__init__(root, transform, buffer_size, train)
        self._seq_len = seq_len

    def _get_data(self, data, seq_index):
        obs_data = data['observations'][seq_index:seq_index + self._seq_len + 1]
        obs_data = self._transform(obs_data.astype(np.float32))
        obs, next_obs = obs_data[:-1], obs_data[1:]
        action = data['actions'][seq_index + 1:seq_index + self._seq_len + 1].astype(np.float32)
        reward, terminal = [data[key][seq_index + 1:seq_index + self._seq_len + 1].astype(np.float32) for key in ('rewards', 'terminals')]

        # Debugging: Print shapes of loaded data
        print(f"Loaded data shapes - obs: {obs.shape}, action: {action.shape}, reward: {reward.shape}, terminal: {terminal.shape}, next_obs: {next_obs.shape}")

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
import torch
import numpy as np


class SampleBuffer:
    def __init__(self, num_examples, max_pos=360):
        """
        Initialize a sample buffer (dictionary of lists) used to store and retrieve values per example, across epochs.

        :param num_examples: determines the size of the sample buffer
        :param max_pos: limits the size of the array at a given sample buffer index
        """
        self.num_examples = num_examples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_examples):
            self.buffer[i] = []

    def __len__(self, i):
        """
        Overloaded to get length of values at dictionary key.

        :param i: key index
        :return: length of values at key index
        """
        return len(self.buffer[i])

    def get_alpha(self, index, samples_per_example=1, device='cuda'):
        """
        Retrieve the lastest rotation `alpha` for a specific interaction by example index.

        :param index: position index in the data stream
        :param samples_per_example: number of times to sample rotation per example
        :param device: `cpu` or `cuda`
        :return: stack of `alpha` samples
        """
        samples = []
        for idx in index:
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                sample = torch.rand(samples_per_example, 1) * 2 * np.pi - np.pi
                samples.append(sample)
            else:
                sample = self.buffer[i][-1]
                samples.append(sample)

        samples = torch.stack(samples, dim=0).to(device=device)

        return samples

    def push_alpha(self, samples, index):
        """
        Push sampled alpha from the model to specific sample index.

        :param samples: sample(s) to push to buffer
        :param index: position index in the data stream
        """
        samples = samples.clone().detach().float().to(device='cpu')
        for sample, idx in zip(samples, index):
            i = idx.item()
            self.buffer[i].append((sample))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get_free_energies_indices(self, index, samples_per_example=1, device='cuda'):
        """
        Retrieve the unique previously visited rotation indices for a specific interaction by example index.
        All rotations visited have an associated free energy used in Monte Carlo free energy sufrace sampling

        :param index: position index in the data stream
        :param samples_per_example: number of times to sample rotation per example
        :param device: `cpu` or `cuda`
        :return: stack of unique rotation indices
        """
        samples = None
        for idx in index:
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                samples = torch.tensor([[]])
            else:
                samples = self.buffer[i][-1]

        samples = torch.unique(samples).unsqueeze(0).to(device=device)
        return samples

    def push_free_energies_indices(self, samples, index):
        """
        Push accumulated rotation indices visited from the model to specific sample index.
        All rotations visited have an associated free energy used in Monte Carlo free energy sufrace sampling

        :param samples: sample(s) to push to buffer
        :param index: position index in the data stream
        """
        samples = samples.clone().detach().float().to(device='cpu')
        i = index.item()
        self.buffer[i].append((samples))

import torch
import numpy as np


class SampleBuffer:
    def __init__(self, num_examples, max_pos=360):
        self.num_examples = num_examples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_examples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push_alpha(self, samples, index):
        # print('alphas push index', index)
        samples = samples.clone().detach().float().to(device='cpu')
        for sample, idx in zip(samples, index):
            i = idx.item()
            self.buffer[i].append((sample))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get_alpha(self, index, samples_per_example, device='cuda'):
        # print('alphas get index', index)
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

    def push_free_energies_indices(self, samples, index):
        # print('free energies push index', index)
        samples = samples.clone().detach().float().to(device='cpu')
        i = index.item()
        self.buffer[i].append((samples))

    def get_free_energies_indices(self, index, samples_per_example=1, device='cuda'):
        # print('free energies get index',index)
        # samples = torch.zeros(1, 1)
        for idx in index:
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                samples = torch.tensor([[]])
            else:
                samples = self.buffer[i][-1]

        # print(samples.shape)
        # print(samples)
        return samples.to(device=device)

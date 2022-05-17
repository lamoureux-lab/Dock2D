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

    def push(self, samples, index):
        # print('alphas push index', index)
        samples = samples.clone().detach().float().to(device='cpu')
        for sample, idx in zip(samples, index):
            i = idx.item()
            self.buffer[i].append((sample))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get(self, index, samples_per_example, device='cuda'):
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

    def push_free_energies(self, samples, index):
        # print('free energies push index', index)
        samples = samples.clone().detach().float().to(device='cpu')
        i = index.item()
        self.buffer[i].append((samples))

        # for sample, idx in zip(samples, index):
        #     i = idx.item()
        #     self.buffer[i].append((sample))
        #     if len(self.buffer[i]) > self.max_pos:
        #         self.buffer[i].pop(0)

    def get_free_energies(self, index, samples_per_example=1, device='cuda'):
        # print('free energies get index',index)
        samples = []
        for idx in index:
            # print('iterating idx in index', idx)
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                # print(str('*'*100)+'initializing 360 zeros')
                sample = torch.zeros(360)
                samples.append(sample)
            else:
                # print('hit else statement')
                sample = self.buffer[i][-1]
                samples.append(sample)
                # print('post buffer init')
                # print(sample)
                # print('entire sample buffer', self.buffer)

        samples = torch.stack(samples, dim=0).to(device=device).squeeze()

        return samples

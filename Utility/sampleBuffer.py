import torch
import numpy as np


class SampleBuffer:
    def __init__(self, num_examples, max_pos=100):
        self.num_examples = num_examples
        self.max_pos = max_pos
        self.buffer = {}
        for i in range(num_examples):
            self.buffer[i] = []

    def __len__(self, i):
        return len(self.buffer[i])

    def push(self, alphas, index):
        alphas = alphas.clone().detach().float().to(device='cpu')
        for alpha, idx in zip(alphas, index):
            i = idx.item()
            self.buffer[i].append((alpha))
            if len(self.buffer[i]) > self.max_pos:
                self.buffer[i].pop(0)

    def get(self, index, samples_per_example, device='cuda'):
        alphas = []
        for idx in index:
            i = idx.item()
            buffer_idx_len = len(self.buffer[i])
            if buffer_idx_len < samples_per_example:
                alpha = torch.rand(samples_per_example, 1) * 2 * np.pi - np.pi
                alphas.append(alpha)
            else:
                alpha = self.buffer[i][-1]
                alphas.append(alpha)

        alphas = torch.stack(alphas, dim=0).to(device=device)

        return alphas

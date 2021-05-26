import torch
from torch import optim
import torch.nn as nn
import numpy as np

from Models import ProteinConv2D
from tqdm import tqdm
import random
from math import cos, sin
import numpy as np

class DockingTrainer:
	def __init__(self, model, optimizer, num_angles=360, device='cuda'):
		self.model = model
		self.optimizer = optimizer
		self.device = device
		self.loss = nn.CrossEntropyLoss(size_average=True)
		
		self.num_angles = num_angles
		self.conv = ProteinConv2D()
		self.angles = torch.from_numpy(np.linspace(-np.pi, np.pi, num=num_angles)).to(device='cuda', dtype=torch.float32)
		self.sigmoid = nn.Sigmoid()

	def load_checkpoint(self, path):
		raw_model = self.model.module if hasattr(self.model, "module") else self.model
		checkpoint = torch.load(path)
		raw_model.load_state_dict(checkpoint)

	def rotate(self, repr, angle):
		alpha = angle.detach()
		T0 = torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.zeros_like(alpha)], dim=1)
		T1 = torch.stack([torch.sin(alpha), torch.cos(alpha), torch.zeros_like(alpha)], dim=1)
		R = torch.stack([T0, T1], dim=1)
		curr_grid = nn.functional.affine_grid(R, size=repr.size(), align_corners=True)
		return nn.functional.grid_sample(repr, curr_grid, align_corners=True)
	
	def dock_global(self, rec_repr, lig_repr):
		batch_size = lig_repr.size(0)
		
		angles = self.angles.unsqueeze(dim=0).repeat(batch_size, 1)
		angles = angles.view(batch_size*self.num_angles)
		
		lig_repr = lig_repr.unsqueeze(dim=1).repeat(1, self.num_angles, 1, 1, 1)
		rec_repr = rec_repr.unsqueeze(dim=1).repeat(1, self.num_angles, 1, 1, 1)
		
		rec_repr = rec_repr.view(batch_size*self.num_angles, rec_repr.size(-3), rec_repr.size(-2), rec_repr.size(-1))
		lig_repr = lig_repr.view(batch_size*self.num_angles, lig_repr.size(-3), lig_repr.size(-2), lig_repr.size(-1))
				
		rot_lig = self.rotate(lig_repr, angles)
		translations = self.conv(rec_repr, rot_lig)
		translations = translations.view(batch_size, self.num_angles, translations.size(-3), translations.size(-2), translations.size(-1))
		return translations

	def score(self, translations):
		batch_size = translations.size(0)
		num_rot = translations.size(1)
		num_feat = translations.size(2)
		L = translations.size(3)

		translations = translations.transpose(1,2).contiguous()
		translations = translations.view(batch_size, num_feat, num_rot*L*L)
		translations = translations.transpose(1,2).contiguous().view(batch_size*num_rot*L*L, num_feat)
		scores = self.model.scorer(translations).squeeze()
		return scores.view(batch_size, num_rot, L, L)

	def get_conformation(self, scores):
		minval_y, ind_y = torch.min(scores, dim=2, keepdim=False)
		minval_x, ind_x = torch.min(minval_y, dim=1)
		minval_angle, ind_angle = torch.min(minval_x, dim=0)
		x = ind_x[ind_angle].item()
		y = ind_y[ind_angle, x].item()
		
		best_score = scores[ind_angle, x, y].item()
		best_translation = torch.tensor([x-scores.size(1)/2.0, y-scores.size(1)/2.0], dtype=torch.float32)
		best_rotation = self.angles[ind_angle]
		
		#Best translations
		self.top_translations = scores[ind_angle,:,:].clone()
		#Best rotations
		self.top_rotations = [torch.min(scores[i,:,:]).item() for i in range(scores.size(0))]

		return best_score, best_rotation, best_translation

	def conf_idx(self, rotation, translation, scores, resolution=1.0):
		batch_size = rotation.size(0)
		s_a = scores.size(1)
		s_x = scores.size(2)
		s_y = scores.size(3)
		flat_idx = []
		for i in range(batch_size):
			angle_idx = int((rotation[i].item() + np.pi) * self.num_angles / (2.0*np.pi)) - 1
			tx_idx = int(translation[i, 0]/resolution + s_x/2)
			ty_idx = int(translation[i, 1]/resolution + s_y/2)
			flat_idx.append(angle_idx * (s_x*s_y) + tx_idx * s_y + ty_idx)
		flat_idx = torch.tensor(flat_idx, dtype=torch.long, device=rotation.device)
		return flat_idx

	def step(self, data, epoch=None):
		receptor, ligand, translation, rotation, pos_idx = data
		receptor = receptor.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		ligand = ligand.to(device=self.device, dtype=torch.float32).unsqueeze(dim=1)
		rotation = rotation.to(device=self.device, dtype=torch.float32)
		translation = translation.to(device=self.device, dtype=torch.float32)
		
		self.model.train()
		self.model.zero_grad()

		rec_repr = self.model.repr(receptor)
		lig_repr = self.model.repr(ligand)
		translations = self.dock_global(rec_repr.tensor, lig_repr.tensor)
		scores = -self.score(translations)
		
		flat_idx = self.conf_idx(rotation, translation, scores)
		probs = scores.contiguous().flatten(start_dim=1)
		loss = self.loss(probs, flat_idx)
		
		loss.backward()
		self.optimizer.step()
		
		return loss.item()

	def eval(self, data):
		pass
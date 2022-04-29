import numpy as np
import _pickle as pkl


import seaborn as sea
sea.set_style("whitegrid")

from DeepProteinDocking2D.DatasetGeneration.Protein import Protein
from tqdm import tqdm
import inspect


class ParamDistribution:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)
		
		all_attr = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		vars = [a for a in all_attr if not(a[0].startswith('__') and a[0].endswith('__'))]
		for param_name, distr in vars:
			self.normalize(param_name)
			
	def normalize(self, param_name):
		Z = 0.0
		param = getattr(self, param_name)
		for val, prob in param:
			Z += prob
		new_param = []
		for val, prob in param:
			new_param.append((val, prob/Z))
		setattr(self, param_name, new_param)

	def sample(self, param_name, return_stats=False):
		param = getattr(self, param_name)
		vals, prob = zip(*param)
		if return_stats:
			return vals, prob
		else:
			return np.random.choice(vals, p=prob)


class ProteinPool:
	def __init__(self, proteins):
		self.proteins = proteins
		self.params = []
		self.interactions = {}

	@classmethod
	def generate(cls, num_proteins, params, size=50):
		pool = cls([])
		stats_alpha = params.sample('alpha', return_stats=True)
		stats_num_points = params.sample('num_points', return_stats=True)
		for i in tqdm(range(num_proteins)):
			alpha = params.sample('alpha')
			num_points = params.sample('num_points')
			prot = Protein.generateConcave(size=size, alpha=alpha, num_points=num_points)
			pool.proteins.append(prot.bulk)
			pool.params.append({'alpha': alpha, 'num_points': num_points})
		return pool, (stats_alpha, stats_num_points)
	
	@classmethod
	def load(cls, filename):
		with open(filename, 'rb') as fin:
			proteins, params, interactions = pkl.load(fin)
		instance = cls(proteins)
		instance.params = params
		instance.interactions = interactions
		return instance
	
	def save(self, filename):
		with open(filename, 'wb') as fout:
			pkl.dump( (self.proteins, self.params, self.interactions), fout)

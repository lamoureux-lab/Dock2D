import numpy as np
import _pickle as pkl


import seaborn as sea
sea.set_style("whitegrid")

from Dock2D.DatasetGeneration.Protein import Protein
from tqdm import tqdm
import inspect


class ParamDistribution:
	def __init__(self, **kwargs):
		"""
		Unzip alpha (shape concavity) and number of points (number of points used to generate shape hulls)
		parameter distributions, then normalize probabilities.

		:param kwargs: list of tuples [(alpha, prob),...] or [(num_points, prob),...]
		"""
		for k, v in kwargs.items():
			setattr(self, k, v)
		
		all_attr = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		vars = [a for a in all_attr if not(a[0].startswith('__') and a[0].endswith('__'))]
		for param_name, distr in vars:
			self.normalize(param_name)
			
	def normalize(self, param_name):
		"""
		Normalize probabilities from parameter distribution.

		:param param_name:
		:return: alpha and number of points normalized probabilities
		"""
		Z = 0.0
		param = getattr(self, param_name)
		for val, prob in param:
			Z += prob
		new_param = []
		for val, prob in param:
			new_param.append((val, prob/Z))
		setattr(self, param_name, new_param)

	def sample(self, param_name):
		"""
		Randomly sample parameters from parameter distribution used in protein shape pool generation.

		:param param_name: `alpha` or `num_points`.
		:type param_name: [str](`alpha` or `num_points`)
		:return: sampled value and probability
		"""
		param = getattr(self, param_name)
		vals, prob = zip(*param)
		return vals, prob


class ProteinPool:
	def __init__(self, proteins):
		"""
		:param proteins: protein shape pools generated
		"""
		self.proteins = proteins
		self.params = []

	@classmethod
	def generate(cls, num_proteins, params, size=50):
		"""
		Generate protein shapes to be used in protein pool.

		:param num_proteins: number of proteins to generate in the pool
		:param params: parameters used in shape generation, list of tuples [(alpha, prob),...] or [(num_points, prob),...]
		:param size: size of the box to generate a shape within.
		:return: protein pool shapes and corresponding individual shape parameters
		"""
		pool = cls([])
		stats_alpha = params.sample('alpha')
		stats_num_points = params.sample('num_points')
		vals_alpha, prob_alpha = stats_alpha
		vals_num_points, prob_num_points = stats_num_points
		for i in tqdm(range(num_proteins)):
			alpha = np.random.choice(vals_alpha, p=prob_alpha)
			num_points = np.random.choice(vals_num_points, p=prob_num_points)
			prot = Protein.generateConcave(size=size, alpha=alpha, num_points=num_points)
			pool.proteins.append(prot.bulk)
			pool.params.append({'alpha': alpha, 'num_points': num_points})
		return pool, (stats_alpha, stats_num_points)
	
	@classmethod
	def load(cls, filename):
		"""
		Load protein pool .pkl

		:param filename: protein pool filename.pkl
		:return: protein pool
		"""
		with open(filename, 'rb') as fin:
			proteins, params = pkl.load(fin)
		instance = cls(proteins)
		instance.params = params
		return instance
	
	def save(self, filename):
		"""
		Save protein pool shapes and corresponding params to .pkl

		:param filename: protein pool filename.pkl
		"""
		with open(filename, 'wb') as fout:
			pkl.dump((self.proteins, self.params), fout)

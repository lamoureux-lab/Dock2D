import _pickle as pkl
import torch
from torch.utils.data import Dataset, RandomSampler
import numpy as np


class InteractionPoseDataset(Dataset):
	def __init__(self, path, max_size=None):
		r"""
		:param path: path to docking dataset .pkl file.
		:param max_size: number of docking examples to be loaded into data stream
		"""
		self.path = path
		with open(self.path, 'rb') as fin:
			self.data = pkl.load(fin)

		if not max_size:
			max_size = len(self.data)
		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print ("Dataset file: ", self.path)
		print ("Dataset size: ", self.dataset_size)

	def __getitem__(self, index):
		r"""
		:return: values at index of interaction data
		"""
		receptor, ligand, rotation, translation = self.data[index]
		return receptor, ligand, rotation, translation

	def __len__(self):
		r"""
		:return: length of the dataset
		"""
		return self.dataset_size


class InteractionFactDataset(Dataset):
	def __init__(self, path, number_of_pairs=None, randomstate=None):
		r"""
		Load data from .pkl dataset file. Build datastream from protein pool,
		interaction indices, and labels.
		The entire dataset is shuffled.

		:param path: path to interaction dataset .pkl file.
		:param number_of_pairs: specifies the data stream `max_size` as number of unique interactions.

			.. math::
				\frac{N(N+1)}{2}

		This is based on `N` interaction pairs. If `N == None`, the entire upper triangle plus diagonal of the interaction pairs array are used.
		"""

		self.path = path
		with open(self.path, 'rb') as fin:
			self.proteins, self.indices, self.labels = pkl.load(fin)

		self.data = []
		for i in range(len(self.labels)):
			receptor_index = self.indices[i][0]
			ligand_index = self.indices[i][1]
			receptor = self.proteins[receptor_index]
			ligand = self.proteins[ligand_index]
			label = self.labels[i]
			self.data.append([receptor, ligand, label])

		if not number_of_pairs:
			max_size = len(self.data)
		else:
			max_size = int((number_of_pairs*(number_of_pairs + 1))/2)

		if randomstate:
			randomstate.shuffle(self.data)
		else:
			np.random.shuffle(self.data)
		self.data = self.data[:max_size]
		self.dataset_size = len(list(self.data))

		print("Dataset file: ", self.path)
		print("Dataset size: ", self.dataset_size)

	def __getitem__(self, index):
		r"""
		:return: values at index of interaction data
		"""
		receptor, ligand, interaction = self.data[index]
		return receptor, ligand, interaction

	def __len__(self):
		r"""
		:return: length of the dataset
		"""
		return self.dataset_size


def get_docking_stream(data_path, shuffle=True, max_size=None, num_workers=0):
	'''
	Get docking data as a torch data stream that is randomly shuffled per epoch.

	:param data_path: path to dataset .pkl file.
	:param shuffle: shuffle using RandomSampler() or not
	:param max_size: number of docking examples to be loaded into data stream
	:param num_workers: number of cpu threads
	:return: docking data stream in format of [receptor, ligand, rotation, translation] (see DatasetGeneration).
	'''
	dataset = InteractionPoseDataset(path=data_path, max_size=max_size)
	if shuffle:
		sampler = RandomSampler(dataset)
	else:
		sampler = None
	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=num_workers)
	return trainloader


def get_interaction_stream(data_path, number_of_pairs=None, randomstate=None, num_workers=0):
	'''
	Get interaction data as a torch data stream, specifying `N` as `number_of_pairs` which results in
	:math:`\\frac{N(N+1)}{2}` unique interactions.
	The fact of interaction data stream shuffles examples when selecting `number_of_pairs` from the entire dataset,
	as well as shuffles the data stream before each epoch.

	.. note::
		For a resumable data stream (e.g. for the SampleBuffer in Monte Carlo fact of interaction),
		set the `randomstate ` by specifying a `numpy.random.RandomState()`,
		this will result in a one-time shuffle at data stream initialization that can be maintained across loading saved models.

	:param data_path: path to dataset .pkl file.
	:param number_of_pairs: number of interaction pair examples to be loaded into data stream
	:param num_workers: number of cpu threads
	:return: interaction data stream [receptor, ligand, 1 or 0] (see DatasetGeneration).
	'''
	dataset = InteractionFactDataset(path=data_path, number_of_pairs=number_of_pairs, randomstate=randomstate)
	if randomstate:
		sampler = None
	else:
		sampler = RandomSampler(dataset)

	trainloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1, num_workers=num_workers)
	return trainloader


def check_datastream_shuffle(data_stream, rand_index=42):
	"""
	Check that interaction stream is shuffled at stream level (for Monte Carlo `FI_MC`)
	and not at epoch level (each epoch needs to be the same for use with sample buffer).
	Using np.random.RandomState(), the shuffled example ordering can be preserved for resumable training.

	:param data_stream: torch data stream being checked
	:param rand_index: random index of data stream to check
	"""
	print('Checking shape at random index', rand_index)
	plots = []
	for epoch in range(3):
		counter = 0
		for data in tqdm(data_stream):
			receptor, ligand, interaction = data
			if counter == rand_index:
				plot = np.hstack((receptor.squeeze().detach().cpu(), ligand.squeeze().detach().cpu()))
				plots.append(plot)
			counter += 1
	plt.imshow(np.vstack((plots)))
	plt.show()


if __name__=='__main__':
	import timeit
	from tqdm import tqdm
	import matplotlib.pyplot as plt
	import seaborn as sea
	sea.set_style("whitegrid")

	randomstate = np.random.RandomState(42)

	train_datapath = '../Datasets/interaction_train_100pool.pkl'
	valid_datapath = '../Datasets/interaction_valid_100pool.pkl'
	test_datapath = '../Datasets/interaction_test_100pool.pkl'

	number_of_pairs = 100
	start = timeit.default_timer()
	train_stream = get_interaction_stream(train_datapath, number_of_pairs=number_of_pairs, randomstate=None)
	valid_stream = get_interaction_stream(valid_datapath, number_of_pairs=number_of_pairs, randomstate=None)
	test_stream = get_interaction_stream(test_datapath, number_of_pairs=number_of_pairs, randomstate=None)
	end = timeit.default_timer()
	print('Total time to load all 3 datasets:', end-start)

	rand_index = randomstate.randint(0, len(train_stream))
	print('Plots should be DIFFERENT shapes at this index...')
	check_datastream_shuffle(train_stream, rand_index=rand_index)
	print('Plots should be SAME shapes at this index...')
	train_stream = get_interaction_stream(train_datapath, number_of_pairs=number_of_pairs, randomstate=randomstate)
	check_datastream_shuffle(train_stream, rand_index=rand_index)

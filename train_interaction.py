import torch
from torch import optim
import argparse
from pathlib import Path

import numpy as np

from Models import CNNInteractionModel, EQScoringModel, EQInteraction, SidInteraction, EQInteractionF, SharpLoss, RankingLoss, EQRepresentationSid
from torchDataset import get_interaction_stream, get_interaction_stream_balanced
from tqdm import tqdm
import random

from SupervisedTrainer import SupervisedTrainer
from DockingTrainer import DockingTrainer

from DatasetGeneration import Protein, Complex
from Logger import Logger

def test(stream, trainer, epoch=0, theshold=0.5):
	TP, FP, TN, FN = 0, 0, 0, 0
	for data in tqdm(stream):
		tp, fp, tn, fn = trainer.eval_coef(data, theshold)
		TP += tp
		FP += fp
		TN += tn
		FN += fn
	
	Accuracy = float(TP + TN)/float(TP + TN + FP + FN)
	if (TP+FP)>0:
		Precision = float(TP)/float(TP + FP)
	else:
		Precision = 0.0
	if (TP + FN)>0:
		Recall = float(TP)/float(TP + FN)
	else:
		Recall = 0.0
	F1 = Precision*Recall/(Precision + Recall+1E-5)
	MCC = (TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1E-5)
	print(f'Epoch {epoch} Acc: {Accuracy} Prec: {Precision} Rec: {Recall} F1: {F1} MCC: {MCC}')
	return Accuracy, Precision, Recall

def get_threshold(stream, trainer):
	all_pred = []
	all_target = []
	for data in tqdm(stream):
		pred, target = trainer.eval(data)
		all_pred.append(pred.clone())
		all_target.append(target)

	all_pred = torch.cat(all_pred, dim=0)
	all_target = torch.cat(all_target, dim=0)
	sorted_pred, perm = torch.sort(all_pred)
	sorted_target = all_target[perm]
	target_true = (sorted_target == 1.0).to(dtype=torch.float32)
	target_false = (sorted_target == 0.0).to(dtype=torch.float32)
	cum_true = torch.cumsum(target_true, dim=0)
	cum_false = torch.cumsum(target_false.flip(dims=(0,)), dim=0).flip(dims=(0,))
	cum = cum_true + cum_false
	m, idx = torch.max(cum, dim=0)
	
	return sorted_pred[idx].item()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')	
	parser.add_argument('-experiment', default='Debug', type=str)
	
	parser.add_argument('-train', action='store_const', const=lambda:'train', dest='cmd')
	parser.add_argument('-test', action='store_const', const=lambda:'test', dest='cmd')

	parser.add_argument('-resnet', action='store_const', const=lambda:'resnet', dest='model')
	parser.add_argument('-docker', action='store_const', const=lambda:'docker', dest='model')

	parser.add_argument('-batch_size', default=8, type=int)
	parser.add_argument('-num_epochs', default=100, type=int)
	parser.add_argument('-pretrain', default=None, type=str)
	parser.add_argument('-gpu', default=1, type=int)



	args = parser.parse_args()

	if torch.cuda.device_count()>1:
		for i in range(torch.cuda.device_count()):
			if i == args.gpu:
				print('->', i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
			else:
				print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))	
		torch.cuda.set_device(args.gpu)

	train_stream = get_interaction_stream_balanced('DatasetGeneration/interaction_data_train.pkl', batch_size=args.batch_size, max_size=25, shuffle=True)
	valid_stream = get_interaction_stream('DatasetGeneration/interaction_data_valid.pkl', batch_size=args.batch_size, max_size=25)

	if args.model() == 'resnet':
		model = CNNInteractionModel().cuda()
		optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.0, 0.999))
		trainer = SupervisedTrainer(model, optimizer, type='int')

	elif args.model() == 'docker':
		scoring_model = EQScoringModel(bias=False)
		if not(args.pretrain is None):
			trainer = DockingTrainer(scoring_model, None, type='pos')
			trainer.load_checkpoint(Path('Log')/Path(args.pretrain)/Path('dock_ebm.th'))
		
		model = EQInteractionF(scoring_model).cuda()
		optimizer = optim.Adam([{'params': scoring_model.parameters(), 'lr':1e-4},
								{'params': model.F0, 'lr':1.0}])

		trainer = DockingTrainer(model, optimizer, type='int', omega=1e-5)

	logger = Logger.new(Path('Log')/Path(args.experiment))

	if args.cmd() == 'train':
		for epoch in range(args.num_epochs):
			losses = []
			for data in tqdm(train_stream):
				loss = trainer.step(data)
				logger.log_train(loss)
				losses.append(loss)
				
			print(f'Loss {np.mean(losses)}')
			print(model.F0.item())
			Accuracy, Precision, Recall = test(valid_stream, trainer, epoch=epoch, theshold=0.5)
			logger.log_valid_inter(Accuracy, Precision, Recall)
			
			
			# torch.save(model.state_dict(), logger.log_dir / Path('model.th'))
	
	elif args.cmd() == 'test':
		trainer.load_checkpoint(logger.log_dir / Path('model.th'))
		test_stream = get_interaction_stream('DatasetGeneration/interaction_data_test.pkl', batch_size=32, max_size=1000)
		print('Validation:')
		Accuracy, Precision, Recall = test(valid_stream, trainer, 0)
		print('Test:')
		Accuracy, Precision, Recall = test(test_stream, trainer, 0)

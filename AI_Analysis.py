# -*- coding: utf-8 -*-

"""
AI analysis via Siamese neural network

"""


from __future__ import print_function, division

import torch
import numpy as np
import pandas as pd
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import sys
import os
import copy

from dataset import CustomImageDataset
from torch.utils.data import DataLoader

######################################################################
from model import *
import utils
from network import *

######################################################################
# Parameters
# ---------
path = './Example_CSV/'
CSVBaseName = 'Data_Example_'
CSVFile_train = './Example_CSV/Data_Example_train.csv'
CSVFile_val = './Example_CSV/Data_Example_val.csv'

# Batch size
bs = 4 # default = 16
# Image size
sz1 = 15
sz2 = 1
# Learning rate
lr1 = 1e-3
lr2 = 1e-4
# Number Epochs
nb_epochs1 = 60
nb_epochs2 = 30


# --------
# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():


	plt.ion()   # interactive mode


	######################################################################
	# Load Data
	# ---------

	# Data augmentation and normalization for training
	# Just normalization for validation
	# Is normalization needed?
	#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

	data_transforms = {
		'train':{
			'input1': transforms.Compose([
				transforms.ToPILImage(),
				#transforms.RandomRotation(5),
				#transforms.ColorJitter(),
				#transforms.RandomHorizontalFlip(),
				#transforms.RandomVerticalFlip(),
				#transforms.RandomResizedCrop(sz1),
				transforms.CenterCrop(sz1),
				#transforms.RandomCrop(sz1),
				transforms.ToTensor(),
				#normalize
			]),
			'input2': transforms.Compose([
				transforms.ToPILImage(),
				#transforms.RandomHorizontalFlip(),
				transforms.CenterCrop(sz2),
				#transforms.RandomCrop(sz2),
				transforms.ToTensor(),
				#normalize
			]),
		},
		'val':{
			'input1': transforms.Compose([
				transforms.ToPILImage(),
				#transforms.Resize(256),
				transforms.CenterCrop(sz1),
				#transforms.RandomCrop(sz1),
				transforms.ToTensor(),
				#normalize
			]),
			'input2': transforms.Compose([
				transforms.ToPILImage(),
				#transforms.Resize(256),
				transforms.CenterCrop(sz2),
				#transforms.RandomCrop(sz2),
				transforms.ToTensor(),
				#normalize
			]),		
		},
		'GroundTruth': transforms.Compose([
			transforms.ToPILImage(),
			#transforms.Resize(256),
			transforms.CenterCrop(sz2),
			#transforms.RandomCrop(sz2),
			transforms.ToTensor(),
			#normalize
		]),
	}



	# ---------
	since = time.time()

	training_data = CustomImageDataset(CSVFile_train,transform=data_transforms['train'], target_transform=data_transforms['GroundTruth'])
	test_data = CustomImageDataset(CSVFile_val,transform=data_transforms['val'], target_transform=data_transforms['GroundTruth'])
	train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=bs, shuffle=True, num_workers=4)
	test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=4)

	dataloaders_dict1 = {}
	dataloaders_dict1['train'] = train_dataloader
	dataloaders_dict1['val'] = test_dataloader


	print('\nlen(training_data)',len(training_data))
	print('len(training_data[0])',len(training_data[0]))
	print('training_data[0][0].shape)',training_data[0][0].shape)

	print('\nlen(test_data)',len(test_data))
	print('len(test_data[0])',len(test_data[0]))
	print('test_data[0][0].shape)',test_data[0][0].shape)

	time_elapsed = time.time() - since
	print('--- Finish loading data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))

	# ---------


	# ---------
	# Initial implementation
	# data_dict = {}
	# dataloaders_dict = {}	

	# since = time.time()
	# for x in ['train', 'val']:

	# 	# CSV File
	# 	data_list = os.path.join( path, (CSVBaseName + x + '.csv'))
	# 	print(data_list)
	# 	data = utils.load_data(data_list, data_transforms[x], data_transforms['GroundTruth'])
	# 	print('\nlen(data)',len(data))
	# 	print('len(data[0])',len(data[0]))
	# 	#time.sleep(20)

	# 	data_dict[x] = data
	# 	dataloaders_dict[x] = torch.utils.data.DataLoader(data_dict[x], batch_size=bs, shuffle=True, num_workers=1)

	# time_elapsed = time.time() - since
	# print('--- Finish loading data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))



	# ----------------------
	# Visualize input data

	# default `log_dir` is "runs" - we'll be more specific here
	writer = SummaryWriter('tensorboard/MyNetwork')

	# Get a batch of training data
	inputs1, inputs2, GroundTruth = next(iter(dataloaders_dict1['train']))
	#print(inputs1)
	#print(classes)
	print('\n\n --- Check Input sizes ---')
	print('inputs1.type: ', inputs1.type())
	# torch.Size([16, 1, 15, 15])
	print('inputs1.shape: ', inputs1.shape)

	print('inputs2.type: ', inputs2.type())
	# torch.Size([16, 1, 1, 1])
	print('inputs2.shape: ', inputs2.shape)

	print('GroundTruth.type: ', GroundTruth.type())
	# torch.Size([16, 1, 1, 1])
	print('GroundTruth.shape: ', GroundTruth.shape)

	# # Make first grid from batch
	# Grid_2DCorr = torchvision.utils.make_grid(inputs1, nrow=4, normalize=True)
	# # shape [3, 70, 70] 4 rows with 2-pixel padding
	# print('\nimshow Grid_2DCorr shape: ', Grid_2DCorr.shape)
	# utils.imshow(Grid_2DCorr, title="Data batch - 2DCorr")

	# # Tensorboard - add grid image
	# writer.add_image('Grid_2DCorr', Grid_2DCorr)
	#plt.show()

	# # Make first grid from batch
	# Grid_TargetDisparity = torchvision.utils.make_grid(inputs2, nrow=4, normalize=True)
	# # shape [3, 14, 14] 4 rows with 2-pixel padding
	# print('\nimshow Grid_TargetDisparity shape: ', Grid_TargetDisparity.shape)
	# utils.imshow(Grid_TargetDisparity, title="Data batch - Target disparity")
	# #plt.show()

	# # Make second grid from batch
	# Grid_GroundTruth = torchvision.utils.make_grid(GroundTruth, nrow=4, normalize=True)
	# # shape [3, 14, 14] 4 rows with 2-pixel padding
	# print('\nimshow Grid_GroundTruth shape: ', Grid_GroundTruth.shape)
	# utils.imshow(Grid_GroundTruth, title="Data batch - Ground Truth")
	# #plt.show()


	######################################################################
	# Transfer learning - step 1: fixed features 
	# ----------------------
	#
	# Load a pretrained model and reset final fully connected layer.
	#

	# ----------------------
	# Create model
	model_ft = Model(writer)

	# Tensorboard - add graph
	writer.add_graph(model_ft.model, [inputs1.to(model_ft.device), inputs2.to(model_ft.device)])
	writer.close()
	# Tensorboard - log embedding
	# features = inputs1.view(-1, 15 * 15)
	# writer.add_embedding(features)
	# writer.close()

	# ----------------------
	# Train and evaluate
	print("\n")
	print('-' * 20)
	print("Training...")
	
	model_ft.train_model(dataloaders=dataloaders_dict1, lr=lr1, nb_epochs=nb_epochs1)
	
	# # ----------------------
	# # Evaluate on validation data
	model_ft.test_model(dataloaders_dict1)


	# # ----------------------
	# # Display predicted images
	# #visualize_model(model_ft, dataloaders_dict)

	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()

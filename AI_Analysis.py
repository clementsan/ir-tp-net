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

from dataset import CustomImageDataset, CustomImageDatasetTIO
from torch.utils.data import DataLoader
import torchio as tio  

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
bs = 400 # default = 16
# Image size
sz1 = 15
sz2 = 1
# Learning rate
lr1 = 5e-3
lr2 = 1e-4
# Number Epochs
nb_epochs1 = 40
nb_epochs2 = 30


# --------
# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():


	plt.ion()   # interactive mode


	######################################################################
	# Load Data
	# ---------

	# Data augmentation and normalization for training
	# Just normalization for validation
	# Is normalization needed?
	#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])



	# ------------------
	since = time.time()

	print('\n--- Loading data... ---')

	Subjects_list_train = utils.GenerateTIOSubjectsList(CSVFile_train)
	Subjects_list_test = utils.GenerateTIOSubjectsList(CSVFile_val)
	
	# torchIO transforms
	TIOtransforms = [
		tio.RandomFlip(axes=('lr')),
	]
	TIOtransform = tio.Compose(TIOtransforms)

	# TIO dataset
	Subjects_dataset_train = tio.SubjectsDataset(Subjects_list_train, transform=TIOtransform)
	Subjects_dataset_test = tio.SubjectsDataset(Subjects_list_test, transform=None)


	time_elapsed = time.time() - since
	print('--- Finish loading data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))
	# ------------------


	# ------------------
	# Patch-based pipeline...
	patch_size = (15,15,122)
	queue_length = 1200
	samples_per_volume = 200 # 62*78 # 4836
	sampler_train = tio.data.UniformSampler(patch_size=patch_size)
	sampler_test = tio.data.UniformSampler(patch_size=patch_size)

	patches_queue_train = tio.Queue(
		Subjects_dataset_train,
		queue_length,
		samples_per_volume,
		sampler_train,
		num_workers=4,
	)
	patches_queue_test = tio.Queue(
		Subjects_dataset_test,
		queue_length,
		samples_per_volume,
		sampler_test,
		num_workers=4,
	)
	
	patches_loader_train = DataLoader(
		patches_queue_train,
		batch_size=bs,
		num_workers=0,  # this must be 0
	)

	patches_loader_test = DataLoader(
		patches_queue_test,
		batch_size=bs,
		num_workers=0,  # this must be 0
	)

	patches_loader_dict = {}
	patches_loader_dict['train'] = patches_loader_train
	patches_loader_dict['val'] = patches_loader_test


	# ----------------------
	# Visualize input data

	# default `log_dir` is "runs" - we'll be more specific here
	writer = SummaryWriter('tensorboard/MyNetwork')

	# # Get a batch of training data
	#inputs1, inputs2, GroundTruth = next(iter(dataloaders_dict1['train']))
	print('\n\n --- Check Input sizes ---')
	patches_batch = next(iter(patches_loader_dict['train']))
	inputs = patches_batch['Combined'][tio.DATA]
	inputs1 = inputs[:,:,:,:,0:120]
	inputs2 = inputs[:,:,:,:,-2]
	GroundTruth = inputs[:,:,:,:,-1]
	
	# Max pooling
	m_MaxPool = nn.AdaptiveMaxPool2d((1,1))
	GroundTruth_MaxPool = m_MaxPool(GroundTruth)
					
	print('inputs.type: ', inputs.type())
	# torch.Size([16, 1, 15, 15])
	print('inputs.shape: ', inputs.shape)
	print('inputs1.shape: ', inputs1.shape)
	print('inputs2.shape: ', inputs2.shape)
	print('GroundTruth.shape: ', GroundTruth.shape)
	print('GroundTruth_MaxPool.shape: ', GroundTruth_MaxPool.shape)


	# # Make first grid from batch
	# Grid_2DCorr = torchvision.utils.make_grid(inputs1, nrow=80, normalize=True)
	# # # shape [3, 70, 70] 4 rows with 2-pixel padding
	# print('\n imshow Grid_2DCorr shape: ', Grid_2DCorr.shape)
	# utils.imshow(Grid_2DCorr, title="Data batch - 2DCorr")

	# # Tensorboard - add grid image
	# writer.add_image('Grid_2DCorr', Grid_2DCorr)
	#plt.show()

	# Make first grid from batch
	# Grid_TargetDisparity = torchvision.utils.make_grid(inputs2, nrow=80, normalize=True)
	# # shape [3, 14, 14] 4 rows with 2-pixel padding
	# print('\n imshow Grid_TargetDisparity shape: ', Grid_TargetDisparity.shape)
	# utils.imshow(Grid_TargetDisparity, title="Data batch - Target disparity")
	# #plt.show()

	# # Make second grid from batch
	# Grid_GroundTruth = torchvision.utils.make_grid(inputs3, nrow=80, normalize=True)
	# # shape [3, 14, 14] 4 rows with 2-pixel padding
	# print('\n imshow Grid_GroundTruth shape: ', Grid_GroundTruth.shape)
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
	model_ft.train_model(dataloaders=patches_loader_dict, lr=lr1, nb_epochs=nb_epochs1)
	
	# ----------------------
	# Evaluate on validation data
	model_ft.test_model(dataloaders=patches_loader_dict)


	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()

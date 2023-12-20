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
bs = 2000 
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

	print('Training set:', len(Subjects_dataset_train), 'subjects')
	print('Validation set:', len(Subjects_dataset_test), 'subjects')

	time_elapsed = time.time() - since
	print('--- Finish loading data in {:.0f}m {:.0f}s---'.format(time_elapsed // 60, time_elapsed % 60))
	# ------------------


	# ------------------
	# Subject visualization
	print('\n--- Subject Info... ---')
	MySubject = Subjects_dataset_train[0]
	print('MySubject: ',MySubject)
	print('MySubject.shape: ',MySubject.shape)
	print('MySubject.spacing: ',MySubject.spacing)
	print('MySubject.spatial_shape: ',MySubject.spatial_shape)
	print('MySubject history: ',MySubject.get_composed_history())

	# ------------------


	# ------------------
	# Patch-based pipeline...
	patch_size = (15,15,122)
	queue_length = 62*78*6 # 29016
	samples_per_volume = 62*78 # 62*78 # 4836
	sampler_train = tio.data.GridSampler(patch_size=patch_size)
	sampler_test = tio.data.GridSampler(patch_size=patch_size)

	patches_queue_train = tio.Queue(
		subjects_dataset = Subjects_dataset_train,
		max_length = queue_length,
		samples_per_volume = samples_per_volume,
		sampler = sampler_train,
		num_workers = 6, #4
		shuffle_subjects = True,
		shuffle_patches = True,
	)
	patches_queue_test = tio.Queue(
		subjects_dataset = Subjects_dataset_test,
		max_length = queue_length,
		samples_per_volume = samples_per_volume,
		sampler = sampler_test,
		num_workers = 6, #4
		shuffle_subjects = True,
		shuffle_patches = True,
	)
	
	patches_loader_train = DataLoader(
		patches_queue_train,
		batch_size = bs,
		#shuffle = False,
		num_workers = 0,  # this must be 0
	)

	patches_loader_test = DataLoader(
		patches_queue_test,
		batch_size = bs,
		#shuffle = False,
		num_workers = 0,  # this must be 0
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
	print('\n --- Check patch Input sizes ---')
	patches_batch = next(iter(patches_loader_dict['val']))
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


	# print('\n --- Plot first subject from batch ---')
	# FirstInputs2 = inputs2[0,:,:,:]
	# print('FirstInputs2.shape: ', FirstInputs2.shape)
	# utils.imshow(FirstInputs2, title="First Case - TargetDisp")
	# plt.show()

	# FirstGroundTruth = GroundTruth[0,:,:,:]
	# print('FirstGroundTruth.shape: ', FirstGroundTruth.shape)
	# utils.imshow(FirstGroundTruth, title="First Case - GroundTruth")
	# plt.show()


	# Make first grid from batch
	# print('\n --- Plot grid from batch ---')
	# GridSize = 100

	# Inputs2_ForGrid = inputs2[0:GridSize,:,:,:]
	# print('Inputs2_ForGrid.shape: ', Inputs2_ForGrid.shape)
	# print('Inputs2_ForGrid.type: ', Inputs2_ForGrid.dtype)
	# Grid_TargetDisparity = torchvision.utils.make_grid(Inputs2_ForGrid, nrow = 10, normalize=True)
	# # shape [3, 15, 15] 5x10 rows with 2-pixel padding
	# print('\n imshow Grid_TargetDisparity shape: ', Grid_TargetDisparity.shape)
	# utils.imshow(Grid_TargetDisparity, title="Data batch - Target disparity")
	# plt.show()

	# # Make second grid from batch
	# GroundTruth_ForGrid = GroundTruth[0:GridSize,:,:,:]
	# Grid_GroundTruth = torchvision.utils.make_grid(GroundTruth_ForGrid, nrow = 10, normalize=True)
	# # shape [3, 15, 15] 5x10 rows with 2-pixel padding
	# print('\n imshow Grid_GroundTruth shape: ', Grid_GroundTruth.shape)
	# utils.imshow(Grid_GroundTruth, title="Data batch - Ground Truth")
	# plt.show()


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
	
	# # ----------------------
	# # Evaluate on validation data
	# model_ft.test_model(dataloaders=patches_loader_dict)


	plt.ioff()
	plt.show()


if __name__ == "__main__":
	main()

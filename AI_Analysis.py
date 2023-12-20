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

# -----------
from model import *
import utils
import dataset
from network import *

######################################################################
# Parameters

# Input files
path = './Example_CSV/'
CSVBaseName = 'Data_Example_'
CSVFile_train = './Example_CSV/Data_Example_train.csv'
CSVFile_val = './Example_CSV/Data_Example_val.csv'

# Data parameters
NeighboringTiles = 5 # for 3x3, or 5x5 adjacent tiles
TileSize = 15
NbImageLayers = 122
Neighboringrid = str(NeighboringTiles) + 'x' + str(NeighboringTiles)
ModelName = './pytorch_model_Tiles' + Neighboringrid + '.h5'

# Data sampling parameters
queue_length = 62*78*NeighboringTiles # 29016
#samples_per_volume = round(62*78/NeighboringTiles) # 62*78 # 4836
samples_per_volume = 62*78 # 62*78 # 4836
num_workers = 6

# Neural network parameters
# Batch size
bs = 2000 
# Learning rate
lr = 5e-3
# Number Epochs
nb_epochs = 40

# Device for CUDA 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
######################################################################


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

	Subjects_list_train = dataset.GenerateTIOSubjectsList(CSVFile_train)
	Subjects_list_test = dataset.GenerateTIOSubjectsList(CSVFile_val)
	
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
	patch_size = (NeighboringTiles * TileSize, NeighboringTiles * TileSize, NbImageLayers)
	sampler_train = tio.data.GridSampler(patch_size=patch_size)
	sampler_test = tio.data.GridSampler(patch_size=patch_size)
	
	patches_queue_train = tio.Queue(
		subjects_dataset = Subjects_dataset_train,
		max_length = queue_length,
		samples_per_volume = samples_per_volume,
		sampler = sampler_train,
		num_workers = num_workers,
		shuffle_subjects = True,
		shuffle_patches = True,
	)
	patches_queue_test = tio.Queue(
		subjects_dataset = Subjects_dataset_test,
		max_length = queue_length,
		samples_per_volume = samples_per_volume,
		sampler = sampler_test,
		num_workers = num_workers,
		shuffle_subjects = True,
		shuffle_patches = True,
	)
	
	patches_loader_train = DataLoader(
		patches_queue_train,
		batch_size = bs,
		shuffle = True,
		num_workers = 0,  # this must be 0
	)

	patches_loader_test = DataLoader(
		patches_queue_test,
		batch_size = bs,
		shuffle = True,
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


	input1_tiles, input2_tiles_real, GroundTruth_real = utils.prepare_data(inputs,NbImageLayers,TileSize,NeighboringTiles)

	print('inputs.shape: ', inputs.shape)
	print('input1_tiles.shape: ', input1_tiles.shape)
	print('input2_tiles_real.shape: ', input2_tiles_real.shape)
	print('GroundTruth_real.shape: ', GroundTruth_real.shape)
	

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
	# Neural network - training 
	# ----------------------
	#
	# Create a neural network model and start training / testing.
	#

	# ----------------------
	# Create model
	model_ft = Model(writer, NbImageLayers, TileSize, NeighboringTiles, ModelName)

	# Tensorboard - add graph
	writer.add_graph(model_ft.model, [input1_tiles.to(model_ft.device), input2_tiles_real.to(model_ft.device)])
	writer.close()


	# # ----------------------
	# # Train and evaluate
	print("\n")
	print('-' * 20)
	print("Training...")
	model_ft.train_model(dataloaders=patches_loader_dict, lr=lr, nb_epochs=nb_epochs)
	
	# # ----------------------
	# # Evaluate on validation data
	# model_ft.test_model(dataloaders=patches_loader_dict)

	plt.ioff()
	plt.show()

if __name__ == "__main__":
	main()

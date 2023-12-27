# -*- coding: utf-8 -*-

"""
AI analysis via parallel neural networks

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import copy
import yaml
import argparse

# Device for CUDA 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchio as tio  

# -----------
from model import *
import utils
import dataset
import sampler


# Manual seed
torch.manual_seed(42)
######################################################################


def arg_parser():
	parser = argparse.ArgumentParser(description='AI analysis - DNN training')
	required = parser.add_argument_group('Required')
	required.add_argument('--config', type=str, required=True,
		help='YAML configuration / parameter file')
	options = parser.add_argument_group('Options')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser


def main(args=None):
	args = arg_parser().parse_args(args)
	config_filename = args.config
	

	# plt.ion()   # interactive mode


	######################################################################
	# Loading parameter file

	print('\n--- Loading configuration file --- ')
	with open(config_filename,'r') as yaml_file:
		config_file = yaml.safe_load(yaml_file)

	if args.verbose:
		print('config_file', config_file)
	
	# Defining parameters
	CSVFile_train = config_file['CSVFile_train']
	CSVFile_val = config_file['CSVFile_val']

	model_filename = config_file['ModelName']
	loss_filename = config_file['LossName']

	nb_image_layers = config_file['NbImageLayers']
	nb_corr_layers = config_file['NbCorrLayers']
	tile_size = config_file['TileSize']
	adjacent_tiles_dim = config_file['AdjacentTilesDim']
	
	num_workers = config_file['num_workers'] 
	samples_per_volume = config_file['samples_per_volume']
	queue_length = config_file['queue_length']
	
	data_filtering = config_file['DataFiltering']
	confidence_threshold = config_file['ConfidenceThreshold']

	dict_fc_features = config_file['dict_fc_features']
	bs = config_file['bs']
	lr = config_file['lr']
	nb_epochs = config_file['nb_epochs']
	# ------------------


	print('\n--- Generating torchIO dataset ---')

	File_list_train, TIOSubjects_list_train = dataset.GenerateTIOSubjectsList(CSVFile_train)
	File_list_test, TIOSubjects_list_test = dataset.GenerateTIOSubjectsList(CSVFile_val)
	
	# torchIO transforms
	TIOtransforms = [
		tio.RandomFlip(axes=('lr')),
	]
	TIOtransform = tio.Compose(TIOtransforms)

	# TIO dataset
	TIOSubjects_dataset_train = tio.SubjectsDataset(TIOSubjects_list_train, transform=TIOtransform)
	TIOSubjects_dataset_test = tio.SubjectsDataset(TIOSubjects_list_test, transform=None)

	print('Training set: ', len(TIOSubjects_dataset_train), 'subjects')
	print('Validation set: ', len(TIOSubjects_dataset_test), 'subjects')
	# ------------------


	# ------------------
	# Subject visualization
	if args.verbose:
		print('\n--- Quality control: TIOSubject Info ---')
		MyTIOSubject = TIOSubjects_dataset_train[0]
		print('MySubject: ', MyTIOSubject)
		print('MySubject.shape: ', MyTIOSubject.shape)
		print('MySubject.spacing: ', MyTIOSubject.spacing)
		print('MySubject.spatial_shape: ', MyTIOSubject.spatial_shape)
		print('MySubject.spatial_shape.type: ', type(MyTIOSubject.spatial_shape))
		print('MySubject history: ', MyTIOSubject.get_composed_history())
	# ------------------

	
	# - - - - - - - - - - - - - -
	# Training with GridSampler

	# patch_size, patch_overlap, padding_mode = dataset.initialize_gridsampler_variables(nb_image_layers, tile_size, adjacent_tiles_dim, padding_mode=None)
	# print('patch_size: ',patch_size)
	# print('patch_overlap: ',patch_overlap)
	# print('padding_mode: ',padding_mode)

	# example_grid_sampler = tio.data.GridSampler(
	# 	subject = MyTIOSubject,
	# 	patch_size = patch_size,
	# 	patch_overlap = patch_overlap,
	# 	padding_mode = padding_mode,
	# )
	# samples_per_volume = len(example_grid_sampler)
	# queue_length = samples_per_volume * num_workers
	# print('samples_per_volume', samples_per_volume)
	# print('queue_length', queue_length)

	# sampler_train = tio.data.GridSampler(
	# 	patch_size = patch_size,
	# 	patch_overlap = patch_overlap,
	# 	padding_mode = padding_mode,
	# )
	# sampler_test = tio.data.GridSampler(
	# 	patch_size = patch_size,
	# 	patch_overlap = patch_overlap,
	# 	padding_mode = padding_mode,
	# )
	# - - - - - - - - - - - - - -
	
	# - - - - - - - - - - - - - -
	# Training with UniformSampler
	print('\n--- Initializing patch sampling variables ---')
	
	patch_size, patch_overlap, padding_mode = dataset.initialize_uniformsampler_variables(nb_image_layers, tile_size, adjacent_tiles_dim, padding_mode=None)
	if args.verbose:
		print('patch_size: ',patch_size)
		print('patch_overlap: ',patch_overlap)
		print('padding_mode: ',padding_mode)
		print('samples_per_volume', samples_per_volume)
		print('queue_length', queue_length)

	sampler_train = sampler.MyUniformSampler(
		patch_size = patch_size,
		tile_size = tile_size,
	)
	sampler_test = sampler.MyUniformSampler(
		patch_size = patch_size,
		tile_size = tile_size,
	)

	patches_queue_train = tio.Queue(
		subjects_dataset = TIOSubjects_dataset_train,
		max_length = queue_length,
		samples_per_volume = samples_per_volume,
		sampler = sampler_train,
		num_workers = num_workers,
		shuffle_subjects = True, 
		shuffle_patches = True, 
	)
	patches_queue_test = tio.Queue(
		subjects_dataset = TIOSubjects_dataset_test,
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
		shuffle = False,
		num_workers = 0,  # this must be 0
	)

	# Dictionary for patch data loaders
	patches_loader_dict = {}
	patches_loader_dict['train'] = patches_loader_train
	patches_loader_dict['val'] = patches_loader_test


	# ----------------------
	# Visualize input data

	writer = SummaryWriter('tensorboard/MyNetwork')

	# # Get a batch of training data
	print('\n--- Quality control: patch inputs ---')
	patches_batch = next(iter(patches_loader_dict['val']))
	inputs = patches_batch['Combined'][tio.DATA]
	locations = patches_batch[tio.LOCATION]


	# Variable initialization needed for TensorBoard
	input_Corr_tiles, input_TargetDisp_tiles_real, GroundTruth_real = dataset.prepare_data_withfiltering(inputs, nb_image_layers, nb_corr_layers, tile_size, adjacent_tiles_dim, data_filtering, confidence_threshold)
	if args.verbose:
		print('\ninput_Corr_tiles.shape: ', input_Corr_tiles.shape)
		print('input_TargetDisp_tiles_real.shape: ', input_TargetDisp_tiles_real.shape)
		print('GroundTruth_real.shape: ', GroundTruth_real.shape)
	

	######################################################################
	# Neural network - training 
	# ----------------------
	#
	# Create a neural network model and start training / testing.
	#

	# ----------------------
	# Create model
	print('\n--- Creating neural network architecture ---')
	model_ft = Model(writer, nb_image_layers, nb_corr_layers, tile_size, adjacent_tiles_dim, model_filename, dict_fc_features, loss_filename, data_filtering, confidence_threshold)

	# Tensorboard - add graph
	writer.add_graph(model_ft.model, [input_Corr_tiles.to(model_ft.device), input_TargetDisp_tiles_real.to(model_ft.device)])
	writer.close()


	# ----------------------
	# Train and evaluate
	print('\n--- DNN training ---')
	model_ft.train_model(dataloaders=patches_loader_dict, lr=lr, nb_epochs=nb_epochs)
	
	# ----------------------
	# Evaluate on validation data
	print('\n--- DNN testing ---')
	model_ft.test_model(dataloaders=patches_loader_dict)

	# plt.ioff()
	# plt.show()

if __name__ == "__main__":
	main()

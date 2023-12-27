#!/usr/bin/env python3

import os
import sys
import argparse
import time 

# Device for CUDA 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
import numpy as np
import pandas as pd
import torchio as tio  
import imageio
import yaml

from network import *
import utils
import dataset


# --------------------

def arg_parser():
	parser = argparse.ArgumentParser(description='AI analysis - DNN inference')
	required = parser.add_argument_group('Required')
	required.add_argument('--config', type=str, required=True,
		help='YAML configuration / parameter file')
	options = parser.add_argument_group('Options')
	options.add_argument('--verbose', action="store_true",
						  help='verbose mode')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	config_filename = args.config

	# InputCSVFile = args.inputCSV
	# ModelName = args.network
	# TileSize = args.tile_size
	# AdjacentTilesDim = args.adjacent_tiles_dim
	# bs = args.bs
	# OutputSuffix = args.outsuffix

	# ----------
	# Loading parameter file
	print('\n--- Loading configuration file --- ')
	with open(config_filename,'r') as yaml_file:
		config_file = yaml.safe_load(yaml_file)

	if args.verbose:
		print('config_file', config_file)
	
	# Defining parameters
	csv_filename = config_file['CSVFile']
	model_filename = config_file['ModelName']

	output_folder = config_file['OutputFolder']
	output_suffix = config_file['OutputSuffix']

	nb_image_layers = config_file['NbImageLayers']
	nb_corr_layers = config_file['NbCorrLayers']
	
	tile_size = config_file['TileSize']
	adjacent_tiles_dim = config_file['AdjacentTilesDim']

	dict_fc_features = config_file['dict_fc_features']
	bs = config_file['bs']
	# ----------
	
	# Device for CUDA 
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# ----------
	
	# print('\n--------------------')
	since1 = time.time()
	time_inference_list = []

	# Creating Subject list from CSV file
	print('\n--- Generating TIO subject list --- ')
	File_list, TIOSubjects_list = dataset.GenerateTIOSubjectsList(csv_filename)

	# Initializing variables
	print('\n--- Initializing data variables --- ')
	TIOSubjectFirst = TIOSubjects_list[0]
	InputFile_Shape = TIOSubjectFirst['Combined'].shape
	NbTiles_H = InputFile_Shape[1] // tile_size
	NbTiles_W = InputFile_Shape[2] // tile_size
	input_depth = nb_corr_layers 

	if args.verbose:
		print('InputFile_Shape: ', InputFile_Shape)
		print('NbTiles_H: ', NbTiles_H)
		print('NbTiles_W: ', NbTiles_W)
		print('nb_image_layers: ', nb_image_layers)
		print('input_depth: ', input_depth)

	print('\n--- Initializing GridSampler variables --- ')
	patch_size, patch_overlap, padding_mode = dataset.initialize_gridsampler_variables(nb_image_layers, tile_size, adjacent_tiles_dim, padding_mode=None)
	if args.verbose:
		print('patch_size: ',patch_size)
		print('patch_overlap: ',patch_overlap)
		print('padding_mode: ',padding_mode)

	# Loading DNN model
	print('\n--- Loading DNN model --- ')
	model = MyParallelNetwork(input_depth, tile_size, adjacent_tiles_dim, dict_fc_features)
	model.load_state_dict(torch.load(model_filename))
	model.to(device)
	model.eval()
	if args.verbose:
		print(model)

	# Patch-based inference
	print('\n--- Patch-based inference --- ')
	for i, (File, TIOSubject) in enumerate(zip(File_list, TIOSubjects_list)):
		since2 = time.time()
		
		# Output filename
		dirname, basename = os.path.split(File)
		basename_without_ext = os.path.splitext(basename)
		output_dir = os.path.join(os.path.dirname(dirname),output_folder)
		os.makedirs(output_dir,exist_ok = True)
		Prediction_basename = basename.replace('_Combined.tiff', '_' + output_suffix + '.tiff')
		PredictionFile = os.path.join(output_dir,Prediction_basename)

		print('\n\t SubjectNb: ', i)
		print('\t FileName: ', basename)
		if args.verbose:
			File_Shape = TIOSubject['Combined'].shape
			print('\t File_Shape: ', File_Shape)
		print('\t\t Subject inference...')
		print('\t\t PredictionFile: ', PredictionFile)
		
		# GridSampler
		grid_sampler = tio.inference.GridSampler(
			subject = TIOSubject,
			patch_size = patch_size,
			patch_overlap = patch_overlap,
			padding_mode = padding_mode,
		)
		len_grid_sampler = len(grid_sampler)
		#print('length grid_sampler', len(grid_sampler))

		patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=bs)
		aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode = 'average')

		with torch.no_grad():
			for patch_idx, patches_batch in enumerate(patch_loader):
				# print('\n\t\t patch_idx: ', patch_idx)

				#print('\t\t Preparing data...')
				inputs = patches_batch['Combined'][tio.DATA]
				# print('\t\t inputs shape: ', inputs.shape)
				input1_tiles, input2_tiles_real, GroundTruth_real = dataset.prepare_data_withfiltering(inputs, nb_image_layers, nb_corr_layers, tile_size, adjacent_tiles_dim)
				#print('\t\t Preparing data - done -')

				input1_tiles = input1_tiles.to(device)
				input2_tiles_real = input2_tiles_real.to(device)
				#GroundTruth_real = GroundTruth_real.to(self.device)
				# Reducing last dimension to compute loss
				#GroundTruth_real = torch.squeeze(GroundTruth_real, dim=2)

				# print('\t\t input1_tiles shape: ', input1_tiles.shape)
				# print('\t\t input2_tiles_real shape:', input2_tiles_real.shape)

				outputs = model(input1_tiles, input2_tiles_real)
				# print('\t\t outputs shape: ',outputs.shape)
				# print('outputs device', outputs.device)

				# Reshape outputs
				outputs_reshape = torch.reshape(outputs,[outputs.shape[0],1,1,1,1])
				# print('\t\t outputs_reshape shape: ',outputs_reshape.shape)

				input_location = patches_batch[tio.LOCATION]
				# print('\t\t input_location shape: ', input_location.shape)
				# print('\t\t input_location: ', input_location)

				# Reshape input_location to prediction_location, to fit output image size (78,62,1)
				pred_location = dataset.prediction_patch_location(input_location, tile_size, adjacent_tiles_dim)
				# print('\t\t pred_location shape: ', pred_location.shape)
				# print('\t\t pred_location: ', pred_location)

				# Add batch with location to TorchIO aggregator
				aggregator.add_batch(outputs_reshape, pred_location)

		# output_tensor shape [1170, 930, 122]
		output_tensor = aggregator.get_output_tensor()
		# print('output_tensor type: ', output_tensor.dtype)
		# print('output_tensor device', output_tensor.device)
		# print('output_tensor shape: ', output_tensor.shape)

		# Extract real information of interest [78,62]
		output_tensor_real = output_tensor[0,:NbTiles_H,:NbTiles_W,0]
		# print('output_tensor_real shape: ', output_tensor_real.shape)

		output_np = output_tensor_real.numpy().squeeze()
		# print('output_np type', output_np.dtype)
		# print('output_np shape', output_np.shape)

		imageio_output = np.moveaxis(output_np, 0,1)
		# print('imageio_output shape', imageio_output.shape)

		time_elapsed2 = time.time() - since2
		time_inference_list.append(time_elapsed2)

		print('\t\t Writing output image - imageio...')
		imageio.imwrite(PredictionFile, imageio_output)

		print('\t\t Inference in {:.2f}s---'.format(time_elapsed2))
		

	time_elapsed1 = time.time() - since1
	# Compute average inference time
	time_inference_np = np.asarray(time_inference_list)
	avg_time_inference = np.mean(time_inference_np)

	print('--- Average inference time in {:.2f}s---'.format(avg_time_inference))
	print('--- Total time in {:.2f}s---'.format(time_elapsed1))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))



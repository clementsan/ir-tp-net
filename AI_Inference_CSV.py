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

def apply_dropout(m):
	if m.__class__.__name__.startswith('Dropout'):
		print('\t\t Enabling MC dropout!')
		m.train()

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	config_filename = args.config

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

	mc_dropout = config_file['MCDropout']
	mc_passes = config_file['MCPasses']
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
	if mc_dropout:
		print('\t MC Dropout')
		model.apply(apply_dropout)

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

		if mc_dropout:
			# Option - MC Dropout: Mean, Median and CV output files (Coefficient of Variation)
			Prediction_basename_mean = basename.replace('_Combined_WithDispLMA.tiff', '_' + output_suffix + '_Mean.tiff')
			PredictionFile_mean = os.path.join(output_dir,Prediction_basename_mean)
			Prediction_basename_median = basename.replace('_Combined_WithDispLMA.tiff', '_' + output_suffix + '_Median.tiff')
			PredictionFile_median = os.path.join(output_dir,Prediction_basename_median)
			Prediction_basename_cv = basename.replace('_Combined_WithDispLMA.tiff', '_' + output_suffix + '_CV.tiff')
			PredictionFile_cv = os.path.join(output_dir,Prediction_basename_cv)
		else:
			# Option - Direct prediction
			Prediction_basename = basename.replace('_Combined_WithDispLMA.tiff', '_' + output_suffix + '.tiff')
			PredictionFile = os.path.join(output_dir,Prediction_basename)
		
		print('\n\t SubjectNb: ', i)
		print('\t FileName: ', basename)
		if args.verbose:
			File_Shape = TIOSubject['Combined'].shape
			print('\t File_Shape: ', File_Shape)
		print('\t\t Subject inference...')
		if mc_dropout:
			print('\t\t PredictionFile_mean: ', PredictionFile_mean)
		else:
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

				if mc_dropout:
					# Perform multiple inference (mc_passes)
					outputs_all = torch.empty(size=(mc_passes, input1_tiles.shape[0])).to(device)
					for i in range(0, mc_passes):			
						outputs = model(input1_tiles, input2_tiles_real)
						outputs_all[i] = torch.squeeze(outputs)

					# Compute mean, median, std, CV (coefficient of variation), SE (standard error)
					outputs_mean = torch.mean(outputs_all,0)
					outputs_median = torch.median(outputs_all,0)[0]
					outputs_std = torch.std(outputs_all,0)
					outputs_cv = torch.div(outputs_std, torch.abs(outputs_mean))
					# outputs_se = torch.div(outputs_std, math.sqrt(mc_passes))
					outputs_combined = torch.stack((outputs_mean, outputs_median, outputs_cv), dim=1)
				else:
					outputs_combined = model(input1_tiles, input2_tiles_real)
				# print('\t\t outputs_combined shape: ', outputs_combined.shape)
				# print('outputs_combined device', outputs_combined.device)

				# Reshape outputs
				outputs_combined_reshape = torch.reshape(outputs_combined,[outputs_combined.shape[0],outputs_combined.shape[1],1,1,1])
				print('\t\t outputs_combined_reshape shape: ', outputs_combined_reshape.shape)
				
				input_location = patches_batch[tio.LOCATION]
				# print('\t\t input_location shape: ', input_location.shape)
				# print('\t\t input_location: ', input_location)

				# Reshape input_location to prediction_location, to fit output image size (78,62,1)
				pred_location = dataset.prediction_patch_location(input_location, tile_size, adjacent_tiles_dim)
				# print('\t\t pred_location shape: ', pred_location.shape)
				# print('\t\t pred_location: ', pred_location)

				# Add batch with location to TorchIO aggregator
				aggregator.add_batch(outputs_combined_reshape, pred_location)

		# output_tensor shape [3, 1170, 930, 122]
		output_tensor_combined = aggregator.get_output_tensor()
		# print('output_tensor_combined type: ', output_tensor_combined.dtype)
		# print('output_tensor_combined device', output_tensor_combined.device)
		# print('output_tensor_combined shape: ', output_tensor_combined.shape)

		# Extract real information of interest [3,78,62]
		output_tensor_combined_real = output_tensor_combined[:,:NbTiles_H,:NbTiles_W,0]
		# print('output_tensor_combined_real shape: ', output_tensor_combined_real.shape)

		output_combined_np = output_tensor_combined_real.numpy().squeeze()
		# print('output_combined_np type', output_combined_np.dtype)
		# print('output_combined_np shape', output_combined_np.shape)

		if mc_dropout:
			output_mean_np = output_combined_np[0,...]
			output_median_np = output_combined_np[1,...]
			output_cv_np = output_combined_np[2,...]
			
			imageio_output_mean = np.moveaxis(output_mean_np, 0,1)
			imageio_output_median = np.moveaxis(output_median_np, 0,1)
			imageio_output_cv = np.moveaxis(output_cv_np, 0,1)
			# print('imageio_output_mean shape', imageio_output_mean.shape)
			# print('imageio_output_median shape', imageio_output_median.shape)
			# print('imageio_output_cv shape', imageio_output_cv.shape)
		else:
			output_np = output_combined_np
			imageio_output = np.moveaxis(output_np, 0,1)
			# print('imageio_output shape', imageio_output.shape)

		time_elapsed2 = time.time() - since2
		time_inference_list.append(time_elapsed2)

		if mc_dropout:
			# print('Writing output mean image via imageio...')
			imageio.imwrite(PredictionFile_mean, imageio_output_mean)
			# print('Writing output median image via imageio...')
			imageio.imwrite(PredictionFile_median, imageio_output_median)
			# print('Writing output CV image via imageio...')
			imageio.imwrite(PredictionFile_cv, imageio_output_cv)
		else:
			# print('Writing output image via imageio...')
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



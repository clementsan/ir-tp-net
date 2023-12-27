#!/usr/bin/env python3

import os
import sys
import argparse
import time 

# Device for CUDA 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchio as tio  
import imageio
import math 

from network import *
import utils
import dataset

# --------------------
# Model - FC layers
dict_fc_features = {
	# Phase1- concatenation on 3rd layer
	'Phase1': [2048,512,256,64],
	'Phase2': [128,64,32],
}
# MC Dropout
mc_dropout = True
mc_passes = 50
# --------------------

def arg_parser():
	parser = argparse.ArgumentParser(description='Inference - ')
	required = parser.add_argument_group('Required')
	required.add_argument('--input', type=str, required=True,
						  help='Combined TIFF file (multi-layer)')
	required.add_argument('--network', type=str, required=True,
						  help='pytorch neural network')
	required.add_argument('--output', type=str, required=True,
						  help='Image prediction (2D TIFF file)')
	options = parser.add_argument_group('Options')
	options.add_argument('--tile_size', type=int, default=15,
						  help='tile size')
	options.add_argument('--adjacent_tiles_dim', type=int, default=1,
						  help='adjacent tiles dim (e.g. 3, 5)')
	options.add_argument('--bs', type=int, default=5000,
						  help='Batch size (default 5000)')
	options.add_argument('--output_median', type=str,
						  help='Image output - median for MCDropout (2D TIFF file)')
	options.add_argument('--output_cv', type=str,
						  help='Image output - Coefficient of Variation for MCDropout (2D TIFF file)')
	return parser


def apply_dropout(m):
	if m.__class__.__name__.startswith('Dropout'):
		print('\t\t Enabling MC dropout!')
		m.train()

#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	InputFile = args.input
	ModelName = args.network
	OutputFile = args.output
	OutputFile_median = args.output_median
	OutputFile_CV = args.output_cv
	TileSize = args.tile_size
	AdjacentTilesDim = args.adjacent_tiles_dim
	bs = args.bs

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print('\n--------------------')
	since1 = time.time()


	# TorchIO subject
	print('\nGenerating TIO subject...')
	Subject = tio.Subject(
		Combined = tio.ScalarImage(InputFile),
	)

	# Initialize variables
	InputFile_Shape = Subject['Combined'].shape
	NbTiles_H = InputFile_Shape[1] // TileSize
	NbTiles_W = InputFile_Shape[2] // TileSize
	NbImageLayers = InputFile_Shape[3]
	NbCorrLayers = NbImageLayers -4
	InputDepth = NbCorrLayers
	print('InputFile_Shape: ', InputFile_Shape)
	print('NbTiles_H: ', NbTiles_H)
	print('NbTiles_W: ', NbTiles_W)
	print('NbImageLayers: ', NbImageLayers)
	print('InputDepth: ', InputDepth)


	# GridSampler
	print('\nGenerating Grid Sampler...')
	patch_size, patch_overlap, padding_mode = dataset.initialize_gridsampler_variables(NbImageLayers, TileSize, AdjacentTilesDim, padding_mode=None)
	print('patch_size: ',patch_size)
	print('patch_overlap: ',patch_overlap)
	print('padding_mode: ',padding_mode)
	

	grid_sampler = tio.data.GridSampler(
		subject = Subject,
		patch_size = patch_size,
		patch_overlap = patch_overlap,
		padding_mode = padding_mode,
	)
	len_grid_sampler = len(grid_sampler)
	print('length grid_sampler', len(grid_sampler))

	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=bs)
	aggregator = tio.data.GridAggregator(grid_sampler, overlap_mode = 'average')
	
	print('\nLoading DNN model...')
	model = MyParallelNetwork(InputDepth, TileSize, AdjacentTilesDim, dict_fc_features)
	model.load_state_dict(torch.load(ModelName))
	print(model)
	model.to(device)
	model.eval()
	if mc_dropout:
		print('\t MC Dropout')
		model.apply(apply_dropout)

	print('\nPatch-based inference...')
	since2 = time.time()
	#model = nn.Identity().eval()
	with torch.no_grad():
		for patch_idx, patches_batch in enumerate(patch_loader):
			print('\t patch_idx: ', patch_idx)

			#print('\t\t Preparing data...')
			inputs = patches_batch['Combined'][tio.DATA]
			print('\t\t inputs shape: ', inputs.shape)
			input1_tiles, input2_tiles_real, GroundTruth_real = dataset.prepare_data_withfiltering(inputs, NbImageLayers, NbCorrLayers, TileSize, AdjacentTilesDim)
			#print('\t\t Preparing data - done -')

			input1_tiles = input1_tiles.to(device)
			input2_tiles_real = input2_tiles_real.to(device)
			#GroundTruth_real = GroundTruth_real.to(device)
			# Reducing last dimension to compute loss
			#GroundTruth_real = torch.squeeze(GroundTruth_real, dim=2)

			print('\t\t input1_tiles shape: ', input1_tiles.shape)
			print('\t\t input2_tiles_real shape:', input2_tiles_real.shape)
			
			if mc_dropout:
				# Perform multiple inference (mc_passes)
				outputs_all = torch.empty(size=(mc_passes, input1_tiles.shape[0])).to(device)
				for i in range(0, mc_passes):			
					outputs = model(input1_tiles, input2_tiles_real)
					outputs_all[i] = torch.squeeze(outputs)

				# Compute mean, std, CV (coefficient of variation), SE (standard error)
				outputs_mean = torch.mean(outputs_all,0)
				outputs_median = torch.median(outputs_all,0)[0]
				outputs_std = torch.std(outputs_all,0)
				outputs_cv = torch.div(outputs_std, torch.abs(outputs_mean))
				# outputs_se = torch.div(outputs_std, math.sqrt(mc_passes))
				outputs_combined = torch.stack((outputs_mean, outputs_median, outputs_cv), dim=1)
				
				print('\t\t outputs shape: ',outputs.shape)
				print('\t\t outputs device', outputs.device)
				print('\t\t outputs_all shape: ', outputs_all.shape)
				print('\t\t outputs_all device', outputs_all.device)
				print('\t\t outputs_mean shape: ', outputs_mean.shape)
				print('\t\t outputs_median shape: ', outputs_median.shape)
				print('\t\t outputs_median type: ', outputs_median.type())
				print('\t\t outputs_combined shape: ', outputs_combined.shape)

				print('\t\t outputs_mean[:20]',outputs_mean[:20])
				print('\t\t outputs_median[:20]',outputs_median[:20])
				print('\t\t outputs_std[:20]',outputs_std[:20])
				print('\t\t outputs_cv[:20]',outputs_cv[:20])
			else:
				outputs_combined = model(input1_tiles, input2_tiles_real)
			
			print('\t\t outputs_combined device', outputs_combined.device)
			print('\t\t outputs_combined shape: ', outputs_combined.shape)

			# Reshape outputs to match location dimensions
			outputs_combined_reshape = torch.reshape(outputs_combined,[outputs_combined.shape[0],outputs_combined.shape[1],1,1,1])
			print('\t\t outputs_combined_reshape shape: ', outputs_combined_reshape.shape)
			
			input_location = patches_batch[tio.LOCATION]
			print('\t\t input_location shape: ', input_location.shape)
			print('\t\t input_location type: ', input_location.dtype)
			print('\t\t input_location[:20]: ', input_location[:20])

			# Reshape input_location to prediction_location, to fit output image size (78,62,1)
			pred_location = dataset.prediction_patch_location(input_location, TileSize, AdjacentTilesDim)
			print('\t\t pred_location shape: ', pred_location.shape)
			print('\t\t pred_location[:20]: ', pred_location[:20])

			# Add batch with location to TorchIO aggregator
			aggregator.add_batch(outputs_combined_reshape, pred_location)

	# output_tensor shape [3, 1170, 930, 124]
	output_tensor_combined = aggregator.get_output_tensor()
	print('output_tensor_combined type: ', output_tensor_combined.dtype)
	print('output_tensor_combined shape: ', output_tensor_combined.shape)

	# Extract real information of interest [3, 78,62]
	output_tensor_combined_real = output_tensor_combined[:,:NbTiles_H,:NbTiles_W,0]
	print('output_tensor_combined_real shape: ', output_tensor_combined_real.shape)

	output_combined_np = output_tensor_combined_real.numpy().squeeze()
	print('output_combined_np type', output_combined_np.dtype)
	print('output_combined_np shape', output_combined_np.shape)

	if mc_dropout:
		output_mean_np = output_combined_np[0,...]
		output_median_np = output_combined_np[1,...]
		output_cv_np = output_combined_np[2,...]
		
		imageio_output_mean = np.moveaxis(output_mean_np, 0,1)
		imageio_output_median = np.moveaxis(output_median_np, 0,1)
		imageio_output_cv = np.moveaxis(output_cv_np, 0,1)
		print('imageio_output_mean shape', imageio_output_mean.shape)
		print('imageio_output_median shape', imageio_output_median.shape)
		print('imageio_output_cv shape', imageio_output_cv.shape)

	else:
		output_np = output_combined_np
		imageio_output = np.moveaxis(output_np, 0,1)
		print('imageio_output shape', imageio_output.shape)
		
	
	time_elapsed2 = time.time() - since2
	
	if mc_dropout:
		print('Writing output mean image via imageio...')
		imageio.imwrite(OutputFile, imageio_output_mean)
		print('Writing output median image via imageio...')
		imageio.imwrite(OutputFile_median, imageio_output_median)
		print('Writing output CV image via imageio...')
		imageio.imwrite(OutputFile_CV, imageio_output_cv)
	else:
		print('Writing output image via imageio...')
		imageio.imwrite(OutputFile, imageio_output)
		

	time_elapsed3 = time.time() - since2
	time_elapsed1 = time.time() - since1
	print('--- Inference in {:.2f}s---'.format(time_elapsed2))
	print('--- Inference and saving in {:.2f}s---'.format(time_elapsed3))
	print('--- Total time in {:.2f}s---'.format(time_elapsed1))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))



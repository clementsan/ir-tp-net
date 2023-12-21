#!/usr/bin/env python3

import os
import sys
import argparse
import time 

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchio as tio  
import imageio

from network import *
import utils

# --------------------
# Model - FC layers
dict_fc_features = {
	# Phase1- concatenation on 3rd layer
	'Phase1': [2048,512,256,64],
	'Phase2': [128,64,32],
}

# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
	options.add_argument('--bs', type=int, default=2000,
						  help='Batch size (default 2000)')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	InputFile = args.input
	ModelName = args.network
	OutputFile = args.output
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
	print('InputFile_Shape: ',InputFile_Shape)

	NbTiles_H = InputFile_Shape[1] // TileSize
	NbTiles_W = InputFile_Shape[2] // TileSize
	NbImageLayers = InputFile_Shape[3]
	InputDepth = NbImageLayers -2
	print('NbTiles_H: ',NbTiles_H)
	print('NbTiles_W: ',NbTiles_W)
	print('NbImageLayers: ',NbImageLayers)


	# GridSampler
	print('\nGenerating Grid Sampler...')
	patch_size = (AdjacentTilesDim * TileSize, AdjacentTilesDim * TileSize, NbImageLayers)
	patch_overlap = (0,0,0)

	

	grid_sampler = tio.inference.GridSampler(
		subject = Subject,
		patch_size = patch_size,
		patch_overlap = patch_overlap,
	)
	len_grid_sampler = len(grid_sampler)
	print('length grid_sampler', len(grid_sampler))

	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=len_grid_sampler)
	aggregator = tio.inference.GridAggregator(grid_sampler)

	print('Loading DNN model...')
	model = MyParallelNetwork(InputDepth, TileSize, AdjacentTilesDim, dict_fc_features)
	model.load_state_dict(torch.load(ModelName))
	#print(model)
	model.to(device)
	model.eval()

	print('Patch-based inference...')
	since2 = time.time()
	#model = nn.Identity().eval()
	with torch.no_grad():
		for patch_idx, patches_batch in enumerate(patch_loader):
			print('\t patch_idx: ', patch_idx)

			#print('\t\t Preparing data...')
			inputs = patches_batch['Combined'][tio.DATA]
			print('\t\t inputs shape: ', inputs.shape)
			input1_tiles, input2_tiles_real, GroundTruth_real = utils.prepare_data(inputs,NbImageLayers,TileSize, AdjacentTilesDim)
			#print('\t\t Preparing data - done -')

			input1_tiles = input1_tiles.to(device)
			input2_tiles_real = input2_tiles_real.to(device)
			#GroundTruth_real = GroundTruth_real.to(self.device)
			# Reducing last dimension to compute loss
			#GroundTruth_real = torch.squeeze(GroundTruth_real, dim=2)

			print('\t\t input1_tiles shape: ', input1_tiles.shape)
			print('\t\t input2_tiles_real shape:', input2_tiles_real.shape)

			outputs = model(input1_tiles, input2_tiles_real)
			print('\t\t outputs shape: ',outputs.shape)
			print('outputs device', outputs.device)

			# Reshape outputs
			outputs_reshape = torch.reshape(outputs,[outputs.shape[0],1,1,1,1])
			print('\t\t outputs_reshape shape: ',outputs_reshape.shape)

			locations = patches_batch[tio.LOCATION]
			print('\t\t location shape: ', locations.shape)
			print('\t\t location: ', locations)

			# Reshape locations to fit output image size (78,62,1)
			# - Divide by TileSize
			# - Depth = 1
			locations_new = torch.div(locations, TileSize, rounding_mode='floor')
			locations_new[:,-1]=1
			print('\t\t location_new shape: ', locations_new.shape)
			print('\t\t location_new: ', locations_new)

			aggregator.add_batch(outputs_reshape, locations_new)

	output_tensor = aggregator.get_output_tensor()
	print('output_tensor type: ', output_tensor.dtype)
	print('output_tensor device', output_tensor.device)
	print('output_tensor shape: ', output_tensor.shape)

	output_tensor_real = output_tensor[0,:NbTiles_H,:NbTiles_W,0]
	print('output_tensor_real shape: ', output_tensor_real.shape)

	output_np = output_tensor_real.numpy().squeeze()

	print('output_np type', output_np.dtype)
	print('output_np shape', output_np.shape)

	imageio_output = np.moveaxis(output_np, 0,1)
	print('imageio_output shape', imageio_output.shape)

	time_elapsed2 = time.time() - since2
	
	print('Writing output image - imageio...')
	imageio.imwrite(OutputFile, imageio_output)

	time_elapsed3 = time.time() - since2
	time_elapsed1 = time.time() - since1
	print('--- Inference in {:.2f}s---'.format(time_elapsed2))
	print('--- Inference and saving in {:.2f}s---'.format(time_elapsed3))
	print('--- Total time in {:.2f}s---'.format(time_elapsed1))

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))



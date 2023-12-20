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

from network import *

# --------
# Device for CUDA (pytorch 0.4.0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
	options.add_argument('--bs', type=int, default=100,
						  help='Batch size (default 100)')
	return parser


#MAIN
def main(args=None):
	args = arg_parser().parse_args(args)
	InputFile = args.input
	InputNetwork = args.network
	OutputFile = args.output
	bs = args.bs


	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print('\n ----------')
	print('Patch-based Inference')
	Time1 = time.time()
	#print(bcolors.BOLDWHITE+"Time1: "+str(Time1)+bcolors.ENDC)

	# TorchIO subject
	print('Generating TIO subject...')
	Subject = tio.Subject(
		Combined = tio.ScalarImage(InputFile),
	)

	# GridSampler
	print('Generating Grid Sampler...')
	patch_size = (15,15,122)
	samples_per_volume = 62*78 # 4836
	patch_overlap = 0

	grid_sampler = tio.inference.GridSampler(
		subject = Subject,
		patch_size = patch_size,
		patch_overlap = patch_overlap,
	)
	print('length grid_sampler', len(grid_sampler))

	patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=bs)
	aggregator = tio.inference.GridAggregator(grid_sampler)

	print('Loading DNN model...')
	# model = MyNetwork2()
	# model.load_state_dict(torch.load(InputNetwork))
	# model.eval()

	print('Patch-based inference...')
	model = nn.Identity().eval()
	with torch.no_grad():
		for patch_idx, patches_batch in enumerate(patch_loader):
			print('\t patch_idx: ', patch_idx)

			input_tensor = patches_batch['Combined'][tio.DATA]
			#input_tensor = patches_batch['Combined'][tio.DATA].to(device)
			locations = patches_batch[tio.LOCATION]
			#print('\t location: ', locations)

			# # Generate individual inputs / outputs			
			# inputs1 = input_tensor[:,:,:,:,0:120]
			# inputs2 = input_tensor[:,:,:,:,-2]
			# GroundTruth = input_tensor[:,:,:,:,-1]

			# print('inputs2 shape: ', inputs2.shape)
			# #logits = model(input_tensor)
			# logits = model(inputs2)

			logits = model(input_tensor)

			#print('logits shape: ', logits.shape)
			labels = logits.argmax(dim=tio.CHANNELS_DIMENSION, keepdim=True)
			#print('labels shape: ', labels.shape)
			outputs = labels
			aggregator.add_batch(outputs, locations)
	
	output_tensor = aggregator.get_output_tensor()

	print('output_tensor type: ', output_tensor.dtype)
	print('output_tensor shape: ', output_tensor.shape)



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))



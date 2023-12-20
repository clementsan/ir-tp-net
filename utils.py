from __future__ import print_function, division

import numpy as np 
import matplotlib.pyplot as plt
import torch

def imshow(inp, title=None):
	"""Imshow for Tensor."""
	# inp = inp.numpy().transpose((1, 2, 0))
	# mean = np.array([0.485, 0.456, 0.406])
	# std = np.array([0.229, 0.224, 0.225])
	# inp = std * inp + mean
	# inp = np.clip(inp, 0, 1)

	print('imshow inp.shape: ',inp.shape)
	# Transpose axes for matplotlib from HWC to CHW (Channel, Height, Width)
	inp = inp.numpy().transpose((1, 2, 0))
	print('imshow inp.shape: ',inp.shape)

	fig = plt.subplots()
	plt.imshow(inp,cmap='gray')
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated


# split ROI into 3x3 tiles (nrows x ncols)
def roi_split3x3(t_roi, tile_size):
	t_roi_shape = t_roi.shape

	t_roi_TopLeft = t_roi[:, :, 0:tile_size, 0:tile_size]
	t_roi_TopMiddle = t_roi[:, :, 0:tile_size, tile_size:2*tile_size]
	t_roi_TopRight = t_roi[:, :, 0:tile_size, 2*tile_size:3*tile_size]

	t_roi_MiddleLeft = t_roi[:, :, tile_size:2*tile_size, 0:tile_size]
	t_roi_MiddleMiddle = t_roi[:, :, tile_size:2*tile_size, tile_size:2*tile_size]
	t_roi_MiddleRight = t_roi[:, :, tile_size:2*tile_size, 2*tile_size:3*tile_size]

	t_roi_BottomLeft = t_roi[:, :, 2*tile_size:3*tile_size, 0:tile_size]
	t_roi_BottomMiddle = t_roi[:, :, 2*tile_size:3*tile_size, tile_size:2*tile_size]
	t_roi_BottomRight = t_roi[:, :, 2*tile_size:3*tile_size, 2*tile_size:3*tile_size]

	stack_t_roi = torch.cat((t_roi_TopLeft, t_roi_TopMiddle, t_roi_TopRight, t_roi_MiddleLeft, t_roi_MiddleMiddle , t_roi_MiddleRight, t_roi_BottomLeft, t_roi_BottomMiddle, t_roi_BottomRight),dim = 1)
	return stack_t_roi

# Prepare data as multiple inputs to network (tensors)
def prepare_data3x3(t_input, nb_image_layers, tile_size):
	t_input1 = t_input[:,:,:,:,0:nb_image_layers-2]
	t_input2 = t_input[:,:,:,:,-2]
	t_GroundTruth = t_input[:,:,:,:,-1]
	
	# print('t_input.type: ', t_input.type())
	# # torch.Size([2000, 1, 45, 45, 122])
	# print('t_input.shape: ', t_input.shape)
	# print('t_input1.shape: ', t_input1.shape)
	# print('t_input2.shape: ', t_input2.shape)

	# Split t_input into neighboring tiles
	t_input1_tiles = roi_split3x3(t_input1, tile_size)
	# # torch.Size([2000, 9, 15, 15, 120])
	# print('t_input1_tiles.shape: ', t_input1_tiles.shape)

	t_input2_tiles = roi_split3x3(t_input2, tile_size)
	# # torch.Size([2000, 9, 15, 15])
	t_input2_tiles_real = t_input2_tiles[:,:,::tile_size,::tile_size]
	# # torch.Size([2000, 9, 1, 1])
	# print('t_input2_tiles.shape: ', t_input2_tiles.shape)
	# print('t_input2_tiles_real.shape: ', t_input2_tiles_real.shape)

	t_GroundTruth_tiles = roi_split3x3(t_GroundTruth, tile_size)
	t_GroundTruth_tilescenter = t_GroundTruth_tiles[:,4,...]
	# torch.Size([2000, 15, 15])
	t_GroundTruth_real = t_GroundTruth_tilescenter[:,::tile_size,::tile_size]
	# torch.Size([2000, 1, 1])

	# print('t_GroundTruth_tiles.shape: ', t_GroundTruth_tiles.shape)
	# print('t_GroundTruth_tilescenter.shape: ', t_GroundTruth_tilescenter.shape)
	# print('t_GroundTruth_real.shape: ', t_GroundTruth_real.shape)

	return t_input1_tiles, t_input2_tiles_real, t_GroundTruth_real

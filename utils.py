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


# split ROI into tiles (3x3, or 5x5)
# initial shape [bs=2000,1,45,45,120]
# new shape for 3x3: [bs=2000,9,15,15,120]
def roi_split(t_roi, tile_size, neighboring_tiles):
	# t_roi_shape = [B,C,H,W,D]
	t_roi_shape = t_roi.shape
	#print('\t t_roi_shape: ',t_roi_shape)
	# Remove channel layer
	a = torch.squeeze(t_roi)
	# Reshape tensor [bs=2000,3,15,3,15,120]
	b = a.reshape(t_roi_shape[0],neighboring_tiles,tile_size,neighboring_tiles,tile_size,-1) 
	# Swap axes [bs=2000,3,3,15,15,120]
	c = b.swapaxes(2,3)
	# Combine channels [bs=2000,3x3,15,15,120]
	d = c.reshape(t_roi_shape[0],neighboring_tiles*neighboring_tiles,tile_size,tile_size,-1)
	# Remove last dimension of size 1 when needed (D=1 for TargetDisparity)
	e = torch.squeeze(d,axis=4)
	return e


# Prepare data as multiple inputs to network (tensors)
def prepare_dataInit(t_input, nb_image_layers, tile_size, neighboring_tiles):
	t_input1 = t_input[:,:,:,:,0:nb_image_layers-2]
	t_input2 = t_input[:,:,:,:,-2]
	t_GroundTruth = t_input[:,:,:,:,-1]
	
	# print('t_input.type: ', t_input.type())
	# # torch.Size([2000, 1, 45, 45, 122])
	# print('t_input.shape: ', t_input.shape)
	# print('t_input1.shape: ', t_input1.shape)
	# print('t_input2.shape: ', t_input2.shape)
	# print('t_GroundTruth.shape: ', t_GroundTruth.shape)

	# Split t_input into neighboring tiles
	t_input1_tiles = roi_split(t_input1, tile_size, neighboring_tiles)
	# # torch.Size([2000, 9, 15, 15, 120])
	# print('t_input1_tiles.shape: ', t_input1_tiles.shape)

	t_input2_tiles = roi_split(t_input2, tile_size, neighboring_tiles)
	# # torch.Size([2000, 9, 15, 15])
	t_input2_tiles_real = t_input2_tiles[:,:,::tile_size,::tile_size]
	# # torch.Size([2000, 9, 1, 1])
	# print('t_input2_tiles.shape: ', t_input2_tiles.shape)
	# print('t_input2_tiles_real.shape: ', t_input2_tiles_real.shape)

	t_GroundTruth_tiles = roi_split(t_GroundTruth, tile_size, neighboring_tiles)
	if (neighboring_tiles == 3):
		IndexCenterTile = 4
	elif (neighboring_tiles == 5):
		IndexCenterTile = 12
	else:
		IndexCenterTile = 0
	t_GroundTruth_tilescenter = t_GroundTruth_tiles[:,IndexCenterTile,...]
	# torch.Size([2000, 15, 15])
	t_GroundTruth_real = t_GroundTruth_tilescenter[:,::tile_size,::tile_size]
	# torch.Size([2000, 1, 1])

	# print('t_GroundTruth_tiles.shape: ', t_GroundTruth_tiles.shape)
	# print('t_GroundTruth_tilescenter.shape: ', t_GroundTruth_tilescenter.shape)
	# print('t_GroundTruth_real.shape: ', t_GroundTruth_real.shape)

	return t_input1_tiles, t_input2_tiles_real, t_GroundTruth_real


# Prepare data as multiple inputs to network (tensors)
def prepare_data(t_input, nb_image_layers, tile_size, neighboring_tiles):
	t_input1 = t_input[:,:,:,:,0:nb_image_layers-2]
	t_input2 = t_input[:,:,:,:,-2]
	t_GroundTruth = t_input[:,:,:,:,-1]
	
	# print('t_input.type: ', t_input.type())
	# # torch.Size([2000, 1, 45, 45, 122])
	# print('t_input.shape: ', t_input.shape)
	# print('t_input1.shape: ', t_input1.shape)
	# print('t_input2.shape: ', t_input2.shape)
	# print('t_GroundTruth.shape: ', t_GroundTruth.shape)

	# Generate tiles when needed
	if (neighboring_tiles == 1):
		t_input1_tiles = t_input1
		t_input2_tiles = t_input2
		t_GroundTruth_tiles = t_GroundTruth
	else:
		# Split t_input into neighboring tiles
		t_input1_tiles = roi_split(t_input1, tile_size, neighboring_tiles)
		# # torch.Size([2000, 9, 15, 15, 120])
		# print('t_input1_tiles.shape: ', t_input1_tiles.shape)

		t_input2_tiles = roi_split(t_input2, tile_size, neighboring_tiles)
		# print('t_input2_tiles.shape: ', t_input2_tiles.shape)
		# # torch.Size([2000, 9, 15, 15])
		t_GroundTruth_tiles = roi_split(t_GroundTruth, tile_size, neighboring_tiles)

	# Generate input2_tiles_real, scaling back to 62x78 pixels
	t_input2_tiles_real = t_input2_tiles[:,:,::tile_size,::tile_size]
	# # torch.Size([2000, 9, 1, 1])
	# print('t_input2_tiles_real.shape: ', t_input2_tiles_real.shape)
		
	# Generate t_GroundTruth_real, scaling back to 62x78 pixels
	if (neighboring_tiles == 3):
		IndexCenterTile = 4
	elif (neighboring_tiles == 5):
		IndexCenterTile = 12
	else:
		IndexCenterTile = 0
	t_GroundTruth_tilescenter = t_GroundTruth_tiles[:,IndexCenterTile,...]
	# torch.Size([2000, 15, 15])
	t_GroundTruth_real = t_GroundTruth_tilescenter[:,::tile_size,::tile_size]
	# torch.Size([2000, 1, 1])

	# print('t_GroundTruth_tiles.shape: ', t_GroundTruth_tiles.shape)
	# print('t_GroundTruth_tilescenter.shape: ', t_GroundTruth_tilescenter.shape)
	# print('t_GroundTruth_real.shape: ', t_GroundTruth_real.shape)

	return t_input1_tiles, t_input2_tiles_real, t_GroundTruth_real

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
def roi_split(t_roi, tile_size, adjacent_tiles_dim):
	# t_roi_shape = [B,C,H,W,D]
	t_roi_shape = t_roi.shape
	#print('\t t_roi_shape: ',t_roi_shape)
	# Remove channel layer
	a = torch.squeeze(t_roi)
	# Reshape tensor [bs=2000,3,15,3,15,120]
	b = a.reshape(t_roi_shape[0],adjacent_tiles_dim,tile_size,adjacent_tiles_dim,tile_size,-1) 
	# Swap axes [bs=2000,3,3,15,15,120]
	c = b.swapaxes(2,3)
	# Combine channels [bs=2000,3x3,15,15,120]
	d = c.reshape(t_roi_shape[0],adjacent_tiles_dim*adjacent_tiles_dim,tile_size,tile_size,-1)
	# Remove last dimension of size 1 when needed (D=1 for TargetDisparity)
	e = torch.squeeze(d,axis=4)
	return e


# Prepare data as multiple inputs to network (tensors)
def prepare_data(t_input, nb_image_layers, tile_size, adjacent_tiles_dim):
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
	if (adjacent_tiles_dim == 1):
		t_input1_tiles = t_input1
		t_input2_tiles = t_input2
		t_GroundTruth_tiles = t_GroundTruth
	else:
		# Split t_input into neighboring tiles
		t_input1_tiles = roi_split(t_input1, tile_size, adjacent_tiles_dim)
		# # torch.Size([2000, 9, 15, 15, 120])
		# print('t_input1_tiles.shape: ', t_input1_tiles.shape)

		t_input2_tiles = roi_split(t_input2, tile_size, adjacent_tiles_dim)
		# print('t_input2_tiles.shape: ', t_input2_tiles.shape)
		# # torch.Size([2000, 9, 15, 15])
		t_GroundTruth_tiles = roi_split(t_GroundTruth, tile_size, adjacent_tiles_dim)

	# Generate input2_tiles_real, t_GroundTruth_tiles_real, scaling back to 62x78 pixels
	t_input2_tiles_real = t_input2_tiles[:,:,::tile_size,::tile_size]
	t_GroundTruth_tiles_real = t_GroundTruth_tiles[:,:,::tile_size,::tile_size]
	# # torch.Size([2000, 9, 1, 1])
	# print('t_input2_tiles_real.shape: ', t_input2_tiles_real.shape)
		
	# Generate t_GroundTruth_real, scaling back to 62x78 pixels
	if (adjacent_tiles_dim == 3):
		IndexCenterTile = 4
	elif (adjacent_tiles_dim == 5):
		IndexCenterTile = 12
	else:
		IndexCenterTile = 0
	t_GroundTruth_real_center = t_GroundTruth_tiles_real[:,IndexCenterTile,...]
	#t_input2_real_center = t_input2_tiles_real[:,IndexCenterTile,...]

	# print('t_GroundTruth_tiles.shape: ', t_GroundTruth_tiles.shape)
	# print('t_GroundTruth_tiles_real.shape: ', t_GroundTruth_tiles_real.shape)
	# print('t_GroundTruth_real_center.shape: ', t_GroundTruth_real_center.shape)

	# print('t_input2_tiles.shape: ', t_input2_tiles.shape)
	# print('t_input2_tiles_real.shape: ', t_input2_tiles_real.shape)
	# print('t_input2_real_center.shape: ', t_input2_real_center.shape)

	return t_input1_tiles, t_input2_tiles_real, t_GroundTruth_real_center

# Initialize TorchIO GridSampler variables
# Generate patch_overlap based on adjacent_tiles_dim
def initialize_gridsampler_variables(nb_image_layers, tile_size, adjacent_tiles_dim, padding_mode=None):
	# Define patch_size
	patch_size = (adjacent_tiles_dim * tile_size, adjacent_tiles_dim * tile_size, nb_image_layers)

	# Define padding_mode
	#padding_mode = 'symmetric'

	# Define patch_overlap
	if (adjacent_tiles_dim == 1):
		patch_overlap = (0,0,0)
	elif (adjacent_tiles_dim == 3):
		# patch_overlap = (30,30,0)
		patch_overlap = (2*tile_size,2*tile_size,0)
	elif (adjacent_tiles_dim == 5):
		# patch_overlap = (60,60,0)
		patch_overlap = (4*tile_size,4*tile_size,0)
	else:
		print("Error initialize_gridsampler_variables - adjacent_tiles_dim")
		sys.exit()	
	# print('patch_size: ',patch_size)
	# print('patch_overlap: ',patch_overlap)
	# print('padding_mode: ',padding_mode)

	padding_mode = padding_mode

	return patch_size, patch_overlap, padding_mode


# Initialize TorchIO uniform Sampler variables
# patch_overlap = (0,0,0) # Not directly used
# patch overlap is generated by the random locations
def initialize_uniformsampler_variables(nb_image_layers, tile_size, adjacent_tiles_dim, padding_mode=None):
	# Define patch_size
	patch_size = (adjacent_tiles_dim * tile_size, adjacent_tiles_dim * tile_size, nb_image_layers)

	# Define patch_overlap
	patch_overlap = (0,0,0)

	# Define padding_mode
	#padding_mode = 'symmetric'
	padding_mode = padding_mode
	
	# print('patch_size: ',patch_size)
	# print('patch_overlap: ',patch_overlap)
	# print('padding_mode: ',padding_mode)

	return patch_size, patch_overlap, padding_mode



# Generate TorchIO aggregator patch_location for prediction
# Example - Input patch location for Tiles 5x5 = [   0,    0,    0,   75,   75,  122]
# Example - Output patch location for Tiles5x5 = [ 2,  2,  0,  3,  3,  1]
# - Use CenterTile location
# - Divide by TileSize
# - Depth = 1
def prediction_patch_location(input_location, tile_size, adjacent_tiles_dim):

	if (adjacent_tiles_dim == 1):
		output_location = input_location
	elif (adjacent_tiles_dim == 3):
		#CenterTile_Update = torch.tensor([15,15,0,-15,-15,0], dtype=torch.int64)
		CenterTile_Update = torch.tensor([tile_size,tile_size,0,-tile_size,-tile_size,0], dtype=torch.int64)
		output_location = input_location + CenterTile_Update[None,:]
	elif (adjacent_tiles_dim == 5):
		#CenterTile_Update = torch.tensor([30,30,0,-30,-30,0], dtype=torch.int64)
		CenterTile_Update = torch.tensor([2*tile_size,2*tile_size,0,-2*tile_size,-2*tile_size,0], dtype=torch.int64)
		output_location = input_location + CenterTile_Update[None,:]
	else:
		print("Error prediction_patch_location - adjacent_tiles_dim")
		sys.exit()	
	
	# print('\t\t output_location shape: ', output_location.shape)
	# print('\t\t output_location: ', output_location)

	# Divide by tile_size
	output_location = torch.div(output_location, tile_size, rounding_mode='floor')

	# Update depth to 1 (from 3D volume to 2D image)
	output_location[:,-1]=1

	# print('\t\t output_location shape: ', output_location.shape)
	# print('\t\t output_location: ', output_location)

	return output_location

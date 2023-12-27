
import pandas as pd
import torchio as tio
import torch
import numpy as np 

# Generate list of torchIO subjects from CSV file
def GenerateTIOSubjectsList(CSVFile):

	df = pd.read_csv(CSVFile, sep=',')
	File_list = df['Combined'].tolist()
	TIOSubjects_list = []

	for idx in range(len(File_list)):		
		TIOSubject = tio.Subject(
			Combined = tio.ScalarImage(File_list[idx]),
			)
		TIOSubjects_list.append(TIOSubject)
	return File_list, TIOSubjects_list


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
# Perform data filtering if enabled, using Confidence and DispLMA maps
def prepare_data_withfiltering(t_input, nb_image_layers, nb_corr_layers, tile_size, adjacent_tiles_dim, is_filtering=0, confidence_threshold=0.0):
	t_input_Corr = t_input[:,:,:,:,0:nb_corr_layers]
	t_input_TargetDisp = t_input[:,:,:,:,-4]
	t_GroundTruth = t_input[:,:,:,:,-3]
	t_Confidence = t_input[:,:,:,:,-2]
	t_DispLMA = t_input[:,:,:,:,-1]
	
	# print('t_input.type: ', t_input.type())
	# # torch.Size([2000, 1, 45, 45, 122])
	# print('t_input.shape: ', t_input.shape)
	# print('t_input_Corr.shape: ', t_input_Corr.shape)
	# print('t_input_TargetDisp.shape: ', t_input_TargetDisp.shape)
	# print('t_GroundTruth.shape: ', t_GroundTruth.shape)

	# Generate tiles when needed
	if (adjacent_tiles_dim == 1):
		t_input_Corr_tiles = t_input_Corr
		t_input_TargetDisp_tiles = t_input_TargetDisp
		t_GroundTruth_tiles = t_GroundTruth
		t_Confidence_tiles = t_Confidence
		t_DispLMA_tiles = t_DispLMA
	else:
		# Split t_input into neighboring tiles
		t_input_Corr_tiles = roi_split(t_input_Corr, tile_size, adjacent_tiles_dim)
		# # torch.Size([2000, 9, 15, 15, 120])
		# print('t_input_Corr_tiles.shape: ', t_input_Corr_tiles.shape)

		t_input_TargetDisp_tiles = roi_split(t_input_TargetDisp, tile_size, adjacent_tiles_dim)
		# print('t_input_TargetDisp_tiles.shape: ', t_input_TargetDisp_tiles.shape)
		# # torch.Size([2000, 9, 15, 15])
		t_GroundTruth_tiles = roi_split(t_GroundTruth, tile_size, adjacent_tiles_dim)
		t_Confidence_tiles = roi_split(t_Confidence, tile_size, adjacent_tiles_dim)
		t_DispLMA_tiles = roi_split(t_DispLMA, tile_size, adjacent_tiles_dim)


	# Generate input_TargetDisp_tiles_real, t_GroundTruth_tiles_real, scaling back to 62x78 pixels
	t_input_TargetDisp_tiles_real = t_input_TargetDisp_tiles[:,:,::tile_size,::tile_size]
	t_GroundTruth_tiles_real = t_GroundTruth_tiles[:,:,::tile_size,::tile_size]
	t_Confidence_tiles_real = t_Confidence_tiles[:,:,::tile_size,::tile_size]
	t_DispLMA_tiles_real = t_DispLMA_tiles[:,:,::tile_size,::tile_size]
	
	# # torch.Size([2000, 9, 1, 1])
	# print('\nt_input_TargetDisp_tiles_real.shape: ', t_input_TargetDisp_tiles_real.shape)
	# print('t_GroundTruth_tiles_real.shape: ', t_GroundTruth_tiles_real.shape)
	# print('t_Confidence_tiles_real.shape: ', t_Confidence_tiles_real.shape)
	# print('t_DispLMA_tiles_real.shape: ', t_DispLMA_tiles_real.shape)


	# - - - - - - - - 
	# Data filtering
	# print('Data filtering...')
	if is_filtering:
		t_DispLMA_tiles_real_Mask = ~torch.isnan(t_DispLMA_tiles_real)
		t_Confidence_tiles_real_Mask = torch.where(t_Confidence_tiles_real >= confidence_threshold, 1, 0)
		t_Mask = torch.logical_and(t_DispLMA_tiles_real_Mask, t_Confidence_tiles_real_Mask)
		t_Mask = torch.squeeze(t_Mask)
		if (adjacent_tiles_dim != 1):
			t_Mask = torch.all(t_Mask, axis=1)
		# print('t_Mask.shape: ', t_Mask.shape)
		# print('t_Mask[:20]: ', t_Mask[:20,...])

		t_input_Corr_tiles_filtered = t_input_Corr_tiles[t_Mask]
		t_input_TargetDisp_tiles_real_filtered = t_input_TargetDisp_tiles_real[t_Mask]
		t_GroundTruth_tiles_real_filtered = t_GroundTruth_tiles_real[t_Mask]
		# print('t_input_Corr_tiles_filtered.shape: ', t_input_Corr_tiles_filtered.shape)
		# print('t_input_TargetDisp_tiles_real_filtered.shape: ', t_input_TargetDisp_tiles_real_filtered.shape)
	else:
		t_input_Corr_tiles_filtered = t_input_Corr_tiles
		t_input_TargetDisp_tiles_real_filtered = t_input_TargetDisp_tiles_real
		t_GroundTruth_tiles_real_filtered = t_GroundTruth_tiles_real		

	# Define center tile for GroundTruth and TargetDisp maps
	if (adjacent_tiles_dim == 3):
		IndexCenterTile = 4
	elif (adjacent_tiles_dim == 5):
		IndexCenterTile = 12
	else:
		IndexCenterTile = 0
	t_input_TargetDisp_real_filtered_center = t_input_TargetDisp_tiles_real_filtered[:,IndexCenterTile,...]
	t_GroundTruth_real_filtered_center = t_GroundTruth_tiles_real_filtered[:,IndexCenterTile,...]
	
	# print('t_GroundTruth_tiles.shape: ', t_GroundTruth_tiles.shape)
	# print('t_GroundTruth_tiles_real.shape: ', t_GroundTruth_tiles_real.shape)
	# print('t_GroundTruth_tiles_real_filtered.shape: ', t_GroundTruth_tiles_real_filtered.shape)
	# print('t_GroundTruth_real_filtered_center.shape: ', t_GroundTruth_real_filtered_center.shape)

	# print('t_input_TargetDisp_tiles.shape: ', t_input_TargetDisp_tiles.shape)
	# print('t_input_TargetDisp_tiles_real.shape: ', t_input_TargetDisp_tiles_real.shape)
	# print('t_input_TargetDisp_tiles_real_filtered.shape: ', t_input_TargetDisp_tiles_real_filtered.shape)
	# print('t_input_TargetDisp_real_filtered_center.shape: ', t_input_TargetDisp_real_filtered_center.shape)

	return t_input_Corr_tiles_filtered, t_input_TargetDisp_real_filtered_center, t_GroundTruth_real_filtered_center

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


from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage
from torch.utils.data import Dataset
from torchio.data import SubjectsDataset, Subject

import pandas as pd
import imageio 
import numpy as np


import torch
import torchio as tio


# Custom data class
# - read 120 layers from 2dcorr image
# - apply manual transform - CropCenter
class CustomImageDataset(Dataset):
	def __init__(self, CSVFile, transform=None, target_transform=None):

		df = pd.read_csv(CSVFile, sep=',')
		im2DCorr_list = df['2DCorr'].tolist()
		imTargetDisp_list = df['TargetDisparity'].tolist()
		imGT_list = df['GroundTruth'].tolist()

		num_file = df.shape[0]
		
		self.im2DCorr_list = im2DCorr_list
		self.imTargetDisp_list = imTargetDisp_list
		self.imGT_list = imGT_list
		self.num_file = num_file
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.imGT_list)

	def CropCenter(self,img, cropx=15, cropy=15):
		z,y,x = img.shape
		startx = x//2-(cropx//2)
		starty = y//2-(cropy//2)   
		
		return img[:,starty:starty+cropy,startx:startx+cropx]

	def __getitem__(self, idx):
		img2DCorr_name = self.im2DCorr_list[idx]
		imgTargetDisp_name = self.imTargetDisp_list[idx]
		imgGT_name = self.imGT_list[idx]
	
		# img2DCorr = PILimage.open(img2DCorr_name, 'r')
		# imgTargetDisp = PILimage.open(imgTargetDisp_name, 'r')
		# imgGT = PILimage.open(imgGT_name, 'r')
		
		# Read one-layer only - 2dcorr file
		#img2DCorr = imageio.imread(img2DCorr_name)

		# Read all layers - 2dcorr file
		img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
		img2DCorr = np.array(img2DCorr)
		print('\nimg2DCorr type: ', img2DCorr.dtype)
		print('img2DCorr shape: ', img2DCorr.shape)

		imgTargetDisp = imageio.imread(imgTargetDisp_name)
		print('imgTargetDisp type: ', imgTargetDisp.dtype)
		print('imgTargetDisp shape: ', imgTargetDisp.shape)

		imgGT = imageio.imread(imgGT_name)
		print('imgGT type: ', imgGT.dtype)
		print('imgGT shape: ', imgGT.shape)


		#Special transform for input1 (multi-channel)
		img2DCorr = self.CropCenter(img2DCorr, cropx=15, cropy=15)
		#print(img2DCorr.shape)
		img2DCorr = torch.from_numpy(img2DCorr)

		if self.transform:
			#img2DCorr = self.transform['input1'](img2DCorr)
			imgTargetDisp = self.transform['input2'](imgTargetDisp)

		if self.target_transform:
			imgGT = self.target_transform(imgGT)

		return img2DCorr, imgTargetDisp, imgGT


# Custom data class
# - read 120 layers from 2dcorr image
# - Use of TorchIO transforms
class CustomImageDatasetTIO(Dataset):
	def __init__(self, CSVFile, transform=None, target_transform=None):

		df = pd.read_csv(CSVFile, sep=',')
		im2DCorr_list = df['2DCorr'].tolist()
		imTargetDisp_list = df['TargetDisparity'].tolist()
		imGT_list = df['GroundTruth'].tolist()

		num_file = df.shape[0]
		
		self.im2DCorr_list = im2DCorr_list
		self.imTargetDisp_list = imTargetDisp_list
		self.imGT_list = imGT_list
		self.num_file = num_file
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.imGT_list)

	def __getitem__(self, idx):

		#Retrieve file name 
		img2DCorr_name = self.im2DCorr_list[idx]
		imgTargetDisp_name = self.imTargetDisp_list[idx]
		imgGT_name = self.imGT_list[idx]

		# Read 2dcorr image - all layers
		img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
		img2DCorr = np.array(img2DCorr)
		print('\nimg2DCorr type: ', img2DCorr.dtype)
		print('img2DCorr shape: ', img2DCorr.shape)

		# Read TargetDisp and GroundTruth images
		imgTargetDisp = imageio.imread(imgTargetDisp_name)
		print('imgTargetDisp type: ', imgTargetDisp.dtype)
		print('imgTargetDisp shape: ', imgTargetDisp.shape)

		imgGT = imageio.imread(imgGT_name)
		print('imgGT type: ', imgGT.dtype)
		print('imgGT shape: ', imgGT.shape)

		# - - - - - - - - - - -
		# Resample TargetDisp and imgTG (repeating values, to match img2DCor size)
		imgTargetDisp_repeat0 = np.repeat(imgTargetDisp, 15, axis=0)
		imgTargetDisp_repeat = np.repeat(imgTargetDisp_repeat0, 15, axis=1)
		imgTargetDisp_repeat = np.expand_dims(imgTargetDisp_repeat, axis=0)

		imgGT_repeat0 = np.repeat(imgGT, 15, axis=0)
		imgGT_repeat = np.repeat(imgGT_repeat0, 15, axis=1)
		imgGT_repeat = np.expand_dims(imgGT_repeat, axis=0)
		print('imgTargetDisp_repeat shape: ', imgTargetDisp_repeat.shape)
		print('imgGT_repeat shape: ', imgGT_repeat.shape)

		# Stack TargetDisp and GT to img2dCorr, to generate one 3D volume
		imgAll = np.concatenate((img2DCorr,imgTargetDisp_repeat), axis = 0)
		imgAll = np.concatenate((imgAll,imgGT_repeat), axis = 0)
		print('imgAll shape: ', imgAll.shape)

		# - - - - - - - - - - -
		# Use of torchio transforms
		# Create 4d tensor - (C, W, H, D) Channel, Weight, Height, Dimensions / layer)
		imgAll_tio = np.moveaxis(imgAll, 0, 2)
		imgAll_tio = np.expand_dims(imgAll_tio,axis=0)
		print('imgAll_tio shape: ', imgAll_tio.shape)

		# Step 1 - convert np arrays to tensors
		imgAll_t = torch.from_numpy(imgAll_tio)

		# TorchIO - Create subject
		Subject = tio.Subject(
			All = tio.ScalarImage(tensor=imgAll_t),
		)

		# TorchIO - Create transform
		transform_Subject = tio.Compose([
			tio.RandomFlip(axes=('lr')),
		])

		# TorchIO - apply transform
		print('torchIO Subject transformation...')
		transformed = transform_Subject(Subject)
		# Subject transform
		imgAll_tio_transformed = transformed['All']
		print('torchIO Subject transformation - done -')

		return imgAll_tio_transformed.data




# Custom TIO Subject class
# - read 120 layers from 2dcorr image

class CustomSubjectTIO():
	def __init__(self, CSVFile):

		df = pd.read_csv(CSVFile, sep=',')
		im2DCorr_list = df['2DCorr'].tolist()
		imTargetDisp_list = df['TargetDisparity'].tolist()
		imGT_list = df['GroundTruth'].tolist()

		num_file = df.shape[0]
		
		self.im2DCorr_list = im2DCorr_list
		self.imTargetDisp_list = imTargetDisp_list
		self.imGT_list = imGT_list
		self.num_file = num_file

	def __len__(self):
		return len(self.imGT_list)

	def __getitem__(self, idx):

		#Retrieve file name 
		img2DCorr_name = self.im2DCorr_list[idx]
		imgTargetDisp_name = self.imTargetDisp_list[idx]
		imgGT_name = self.imGT_list[idx]

		# Read 2dcorr image - all layers
		img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
		img2DCorr = np.array(img2DCorr)
		# print('\nimg2DCorr type: ', img2DCorr.dtype)
		# print('img2DCorr shape: ', img2DCorr.shape)

		# Read TargetDisp and GroundTruth images
		imgTargetDisp = imageio.imread(imgTargetDisp_name)
		# print('imgTargetDisp type: ', imgTargetDisp.dtype)
		# print('imgTargetDisp shape: ', imgTargetDisp.shape)

		imgGT = imageio.imread(imgGT_name)
		# print('imgGT type: ', imgGT.dtype)
		# print('imgGT shape: ', imgGT.shape)

		# - - - - - - - - - - -
		# Resample TargetDisp and imgTG (repeating values, to match img2DCor size)
		imgTargetDisp_repeat0 = np.repeat(imgTargetDisp, 15, axis=0)
		imgTargetDisp_repeat = np.repeat(imgTargetDisp_repeat0, 15, axis=1)
		imgTargetDisp_repeat = np.expand_dims(imgTargetDisp_repeat, axis=0)

		imgGT_repeat0 = np.repeat(imgGT, 15, axis=0)
		imgGT_repeat = np.repeat(imgGT_repeat0, 15, axis=1)
		imgGT_repeat = np.expand_dims(imgGT_repeat, axis=0)
		# print('imgTargetDisp_repeat shape: ', imgTargetDisp_repeat.shape)
		# print('imgGT_repeat shape: ', imgGT_repeat.shape)

		# Stack TargetDisp and GT to img2dCorr, to generate one 3D volume
		imgAll = np.concatenate((img2DCorr,imgTargetDisp_repeat), axis = 0)
		imgAll = np.concatenate((imgAll,imgGT_repeat), axis = 0)
		# print('imgAll shape: ', imgAll.shape)

		# - - - - - - - - - - -
		# Use of torchio transforms
		# Create 4d tensor - (C, W, H, D) Channel, Weight, Height, Dimensions / layer)
		imgAll_tio = np.moveaxis(imgAll, 0, 2)
		imgAll_tio = np.expand_dims(imgAll_tio,axis=0)
		# print('imgAll_tio shape: ', imgAll_tio.shape)

		# Step 1 - convert np arrays to tensors
		imgAll_t = torch.from_numpy(imgAll_tio)

		# TorchIO - Create subject
		Subject = tio.Subject(
			All = tio.ScalarImage(tensor=imgAll_t),
		)

		return Subject


# Custom TIO Subject class
# - read 122 layers from Combined image
class MySubjectTIO():
	def __init__(self, CSVFile):

		df = pd.read_csv(CSVFile, sep=',')
		imCombined_list = df['Combined'].tolist()

		num_file = df.shape[0]
		
		self.imCombined_list = imCombined_list
		self.num_file = num_file

	def __len__(self):
		return len(self.imCombined_list)

	def __getitem__(self, idx):

		#Retrieve file name 
		imgCombined_name = self.imCombined_list[idx]

		# Read 2dcorr image - all layers
		imgCombined = imageio.mimread(imgCombined_name,memtest=False)
		imgCombined = np.array(imgCombined)
		# print('\nimgCombined type: ', imgCombined.dtype)
		# print('imgCombined shape: ', imgCombined.shape)

		# Read TargetDisp and GroundTruth images
		imgTargetDisp = imageio.imread(imgTargetDisp_name)
		# print('imgTargetDisp type: ', imgTargetDisp.dtype)
		# print('imgTargetDisp shape: ', imgTargetDisp.shape)

		imgGT = imageio.imread(imgGT_name)
		# print('imgGT type: ', imgGT.dtype)
		# print('imgGT shape: ', imgGT.shape)

		# - - - - - - - - - - -
		# Resample TargetDisp and imgTG (repeating values, to match img2DCor size)
		imgTargetDisp_repeat0 = np.repeat(imgTargetDisp, 15, axis=0)
		imgTargetDisp_repeat = np.repeat(imgTargetDisp_repeat0, 15, axis=1)
		imgTargetDisp_repeat = np.expand_dims(imgTargetDisp_repeat, axis=0)

		imgGT_repeat0 = np.repeat(imgGT, 15, axis=0)
		imgGT_repeat = np.repeat(imgGT_repeat0, 15, axis=1)
		imgGT_repeat = np.expand_dims(imgGT_repeat, axis=0)
		# print('imgTargetDisp_repeat shape: ', imgTargetDisp_repeat.shape)
		# print('imgGT_repeat shape: ', imgGT_repeat.shape)

		# Stack TargetDisp and GT to img2dCorr, to generate one 3D volume
		imgAll = np.concatenate((img2DCorr,imgTargetDisp_repeat), axis = 0)
		imgAll = np.concatenate((imgAll,imgGT_repeat), axis = 0)
		# print('imgAll shape: ', imgAll.shape)

		# - - - - - - - - - - -
		# Use of torchio transforms
		# Create 4d tensor - (C, W, H, D) Channel, Weight, Height, Dimensions / layer)
		imgAll_tio = np.moveaxis(imgAll, 0, 2)
		imgAll_tio = np.expand_dims(imgAll_tio,axis=0)
		# print('imgAll_tio shape: ', imgAll_tio.shape)

		# Step 1 - convert np arrays to tensors
		imgAll_t = torch.from_numpy(imgAll_tio)

		# TorchIO - Create subject
		Subject = tio.Subject(
			All = tio.ScalarImage(tensor=imgAll_t),
		)

		return Subject

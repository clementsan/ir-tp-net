from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage
from torch.utils.data import Dataset
import pandas as pd
import imageio 
import numpy as np


import torch
import torchio as tio



class MyData(Dataset):
	def __init__(self, file_list):

		df = pd.read_csv(file_list, sep=',')
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
		
		img2DCorr_name = self.im2DCorr_list[idx]
		imgTargetDisp_name = self.imTargetDisp_list[idx]
		imgGT_name = self.imGT_list[idx]
	
		# img2DCorr = PILimage.open(img2DCorr_name, 'r')
		# imgTargetDisp = PILimage.open(imgTargetDisp_name, 'r')
		# imgGT = PILimage.open(imgGT_name, 'r')
		
		# Read multi-layer file
		#img2DCorr = imageio.mimread(img2DCorr_name,memtest=False)
		#print(len(img2DCorr))
		img2DCorr = imageio.imread(img2DCorr_name)
		# Convert list to ndarray
		#img2DCorr = np.array(img2DCorr)
		#print(img2DCorr.shape)
		#img2DCorr = np.expand_dims(img2DCorr, axis=0)
		#img2DCorr = np.moveaxis(img2DCorr, 0, 1)
		#print(img2DCorr.shape)
		
		# Change order of axes
		#img2DCorr = np.moveaxis(img2DCorr, 0, 2)

		#print('\nimg2DCorr type: ', img2DCorr.dtype)
		#print('img2DCorr shape: ', img2DCorr.shape)
		imgTargetDisp = imageio.imread(imgTargetDisp_name)
		#print('imgTargetDisp type: ', imgTargetDisp.dtype)
		#print('imgTargetDisp shape: ', imgTargetDisp.shape)
		imgGT = imageio.imread(imgGT_name)
		#print('imgGT type: ', imgGT.dtype)
		#print('imgGT shape: ', imgGT.shape)
		
		return img2DCorr, imgTargetDisp, imgGT



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


# Use of TorchIO transforms
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

		# - - - - - - - - - - -
		# Use of torchio transforms
		# Create 4d tensor - (C, W, H, D) channel, Weight, Height, Dimensions / layer)
		img2DCorr_tio = np.moveaxis(img2DCorr, 0, 2)
		img2DCorr_tio = np.expand_dims(img2DCorr_tio,axis=0)
		print('img2DCorr_tio shape: ', img2DCorr_tio.shape)

		imgTargetDisp_tio = np.expand_dims(imgTargetDisp,axis=0)
		imgTargetDisp_tio = np.expand_dims(imgTargetDisp_tio,axis=-1)
		print('imgTargetDisp_tio shape: ', imgTargetDisp_tio.shape)

		imgGT_tio = np.expand_dims(imgGT,axis=0)
		imgGT_tio = np.expand_dims(imgGT_tio,axis=-1)
		print('imgGT_tio shape: ', imgGT_tio.shape)

		# Step 1 - convert np arrays to tensors
		img2DCorr_t = torch.from_numpy(img2DCorr_tio)
		imgTargetDisp_t = torch.from_numpy(imgTargetDisp_tio)
		imgGT_t = torch.from_numpy(imgGT_tio)
		
		# Affine matrix for world coordinates
		# Aff_matrix = np.array([[15,0,0,0],[0,15,0,0],[0,0,15,0],[0,0,0,1]])
		# print('Aff_matrix: ')
		# print(Aff_matrix)

		# Subject = tio.Subject(
		# 	Corr = tio.ScalarImage(tensor=img2DCorr_t),
		# 	TargetDisp = tio.ScalarImage(tensor=imgTargetDisp_t, affine=Aff_matrix),
		# 	GT = tio.ScalarImage(tensor=imgGT_t, affine=Aff_matrix),
		# )

		imgTemplate = np.zeros((930,1170))
		imgTemplate_tio = np.expand_dims(imgTemplate,axis=0)
		imgTemplate_tio = np.expand_dims(imgTemplate_tio,axis=-1)
		print('imgTemplate_tio shape: ', imgTemplate_tio.shape)


		Subject = tio.Subject(
			Corr = tio.ScalarImage(tensor=img2DCorr_t),
			TargetDisp = tio.ScalarImage(tensor=imgTargetDisp_t),
			GT = tio.ScalarImage(tensor=imgGT_t),
			Template = tio.ScalarImage(tensor=imgTemplate_tio),
		)

		transform_Subject = tio.Compose([
			#tio.ToCanonical(),
			#tio.Resample(target='Template', include=['TargetDisp','GT'], image_interpolation='nearest'),
			#tio.ZNormalization(),
			tio.RandomFlip(axes=('lr')),
			#tio.CropOrPad((15,15,120), include='Corr'),
			#tio.CropOrPad((15,15,120), include='Corr'),
			#tio.CropOrPad((15,15,1), include=['TargetDisp','GT']),
		])

		# Perform individual cropping
		transform_Corr = tio.Compose([
			tio.ZNormalization(),
			tio.CropOrPad((15,15,120)),
		])
		transform_Other = tio.Compose([
			tio.ZNormalization(),
			tio.CropOrPad((1,1,1)),
		])

		
		print('torchIO Subject transformation...')
		transformed = transform_Subject(Subject)
		# Subject transform
		img2DCorr_tio_transformed = transformed['Corr']
		imgTargetDisp_tio_transformed = transformed['TargetDisp']
		imgGT_tio_transformed = transformed['GT']
		print('torchIO Subject transformation - done -')

		# Individual transforms
		print('torchIO Indiv transformation...')
		img2DCorr_tio_transformed2 = transform_Corr(img2DCorr_tio_transformed)
		imgTargetDisp_tio_transformed2 = transform_Other(imgTargetDisp_tio_transformed)
		imgGT_tio_transformed2 = transform_Other(imgGT_tio_transformed)
		print('torchIO Indiv transformation - done -')


		print('img2DCorr_tio_transformed type: ', type(img2DCorr_tio_transformed))
		print('img2DCorr_tio_transformed2 type: ', type(img2DCorr_tio_transformed2))
		print('img2DCorr_tio_transformed shape: ', img2DCorr_tio_transformed2.shape)
		print('imgTargetDisp_tio_transformed shape: ', imgTargetDisp_tio_transformed2.shape)
		print('imgGT_tio_transformed shape: ', imgGT_tio_transformed2.shape)

		#img2DCorr_tio_transformed2_t = torch.from_numpy(img2DCorr_tio_transformed2.numpy()))
		# Return torchTensor instead of tio ScalarImage
		return img2DCorr_tio_transformed2.data, imgTargetDisp_tio_transformed2.data, imgGT_tio_transformed2.data


from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage
from torch.utils.data import Dataset
import pandas as pd
import imageio 
import numpy as np

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

		if self.transform:
			img2DCorr = self.transform['input1'](img2DCorr)
			imgTargetDisp = self.transform['input2'](imgTargetDisp)

		if self.target_transform:
			imgGT = self.target_transform(imgGT)

		return img2DCorr, imgTargetDisp, imgGT

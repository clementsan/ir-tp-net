from __future__ import print_function, division, absolute_import, unicode_literals
from PIL import Image as PILimage
from torch.utils.data import Dataset
from torchio.data import SubjectsDataset, Subject

import pandas as pd
import imageio 
import numpy as np


import torch
import torchio as tio


# Generate list of images from CSV file
def GenerateFileList(CSVFile):
	df = pd.read_csv(CSVFile, sep=',')
	file_list = df['Combined'].tolist()

	return file_list

# Initialize variables, reading first image file
def RetrieveImageInfo(image_file):
	imageio_in = imageio.mimread(image_file,memtest=False)
	imageio_in = np.array(imageio_in)
	# image_shape (122, 930, 1170)
	image_shape = imageio_in.shape
	nb_image_layers = image_shape[0]
	input_depth = nb_image_layers -2
	
	return image_shape, nb_image_layers, input_depth

# Generate label template as numpy array
# Numpy array size (H,W)
# - TIO TensorSize: (C, W, H, D)
def GenerateTIOLabelTemplate(image_shape, tile_size):
	nb_tiles_H = image_shape[1] // tile_size
	nb_tiles_W = image_shape[2] // tile_size

	# Generate Template Tile (15,15)
	template_tile = np.zeros((tile_size,tile_size), dtype=np.int32)
	template_tile[7,7] = 1
	print('\t template_tile shape',template_tile.shape)

	# Generate label template (2d image)
	print("\nGenerating label template...")
	label_template = np.tile(template_tile, (nb_tiles_H,nb_tiles_W))
	print('\t label_template shape',label_template.shape)

	# Update axes for torchIO format
	label_template = np.moveaxis(label_template,0,1)
	print('\t label_template shape',label_template.shape)
	label_template = np.expand_dims(label_template,0)
	label_template = np.expand_dims(label_template,-1)
	print('\t label_template shape',label_template.shape)

	return label_template

# Generate list of tio subjects from CSV file - used for inference
def GenerateTIOSubjectsListFromCSV(CSVFile):
	df = pd.read_csv(CSVFile, sep=',')
	file_list = df['Combined'].tolist()
	TIOSubjects_list = []

	for idx in range(len(file_list)):		
		TIOSubject = tio.Subject(
			Combined = tio.ScalarImage(file_list[idx]),
			)
		TIOSubjects_list.append(TIOSubject)
	return file_list, TIOSubjects_list

# Generate list of tio subjects with label template - used for training
def GenerateTIOSubjectsListWithLabelMap(file_list, tio_label_template):
	TIOSubjects_list = []

	for idx in range(len(file_list)):		
		TIOSubject = tio.Subject(
			Combined = tio.ScalarImage(file_list[idx]),
			#Label = tio.LabelMap(tensor=tio_label_template, orientation=('L', 'P', 'S')),
			Mask = tio.LabelMap(tio_label_template),
		)
		TIOSubjects_list.append(TIOSubject)
	return TIOSubjects_list

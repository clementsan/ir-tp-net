from __future__ import print_function, division

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torchio as tio 

from dataset import CustomSubjectTIO


# Generate list of tio subjects from tensors
def GenerateTIOSubjects(CSVFile):
	MyCustomSubjectTIO = CustomSubjectTIO(CSVFile)
	Subjects = []

	for x in range(MyCustomSubjectTIO.num_file):		
		Subjects.append(MyCustomSubjectTIO[x])
	return Subjects


# Generate list of tio subjects from CSV file
def GenerateTIOSubjectsList(CSVFile):

	df = pd.read_csv(CSVFile, sep=',')
	File_list = df['Combined'].tolist()
	Subjects = []

	for idx in range(len(File_list)):		
		Subject = tio.Subject(
			Combined = tio.ScalarImage(File_list[idx]),
			)
		Subjects.append(Subject)
	return Subjects



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

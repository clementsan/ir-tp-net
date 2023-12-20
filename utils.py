from __future__ import print_function, division

import numpy as np 
import matplotlib.pyplot as plt

from dataset import MyData

def load_data(data_list, data_transforms_input, data_transforms_output):
	dataclass = MyData(data_list)
	datasets = []

	for x in range(dataclass.num_file):		
		datasets.append((data_transforms_input['input1'](dataclass[x][0]), data_transforms_input['input2'](dataclass[x][1]), \
			 data_transforms_output(dataclass[x][2])))

	return datasets


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
	#plt.imshow(inp, cmap='gray')
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)  # pause a bit so that plots are updated

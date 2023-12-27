
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

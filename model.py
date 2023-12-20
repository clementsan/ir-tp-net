from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np 
import scipy
import sys
import os
import copy
import time
import matplotlib.pyplot as plt
import torchio as tio
#from torchsummmary import summary

#from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from layers import *
from network import *
import utils

class Model(object):
	def __init__(self, writer):

		self.writer = writer
		# Criterion MSE -> loss RMSE
		self.criterion = nn.MSELoss()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# Create ResNet50 model
		self.create_model()

	def create_model(self):
		
		# Fully Connected network - 5 layers
		#self.model = MySingleNetwork()
		self.model = MyParallelNetwork()
		print(self.model)

		# Attach to device
		self.model = self.model.to(self.device)
		


	# Need to udpate: step1 vs step2
	def train_model(self, dataloaders, lr, nb_epochs=25, nb_image_layers=120, tile_size=15):
		since = time.time()

		# Unfreeze all layers
		# for param in self.model.parameters():
		# 	param.requires_grad = True

		# Observe that all parameters are being optimized
		optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

		# Decay LR by a factor of 0.1 every 7 epochs
		scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

		best_model_wts = copy.deepcopy(self.model.state_dict())
		#best_acc = 0.0
		best_loss = 100000

		train_loss = []
		val_loss = []
		#train_acc = []
		#val_acc = []

		for epoch in range(nb_epochs):
			print('Epoch {}/{}'.format(epoch, nb_epochs - 1))
			print('-' * 10)

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
					scheduler.step()
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				#running_corrects = 0

				# Iterate over data.
				#for inputs1, inputs2, GroundTruth in dataloaders[phase]:
				for patch_idx, patches_batch in enumerate(dataloaders[phase]):
					print('\t patch_idx: ', patch_idx)
					inputs = patches_batch['Combined'][tio.DATA]

					print('\t\t Preparing data...')
					input1_tiles, input2_tiles_real, GroundTruth_real = utils.prepare_data3x3(inputs,nb_image_layers,tile_size)
					print('\t\t Preparing data - done -')
					
					input1_tiles = input1_tiles.to(self.device)
					input2_tiles_real = input2_tiles_real.to(self.device)
					GroundTruth_real = GroundTruth_real.to(self.device)
					#print('\t\t GroundTruth.shape: ', GroundTruth.shape)

					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):

						# Provide two inputs to model
						outputs = self.model(input1_tiles, input2_tiles_real)
						#_, preds = torch.max(outputs, 1)
						loss = torch.sqrt(self.criterion(outputs, GroundTruth_real))

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							optimizer.step()

					# statistics
					running_loss += loss.item() * input1_tiles.size(0)
					#running_corrects += torch.sum(preds == labels.data)

				epoch_loss = running_loss / len(dataloaders[phase].dataset) 
				#epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) 

				curr_lr = optimizer.param_groups[0]['lr']

				print('{} Loss: {:.4f} Lr: {:.6f}'.format(
					phase, epoch_loss, curr_lr))

				# Append values for plots
				if phase == 'train':
					train_loss.append(epoch_loss)
					#train_acc.append(epoch_acc)
					self.writer.add_scalar('Loss/train', epoch_loss, epoch)
				else:
					val_loss.append(epoch_loss)
					#val_acc.append(epoch_acc)
					self.writer.add_scalar('Loss/val', epoch_loss, epoch)

				# deep copy the model
				#if phase == 'val' and epoch_acc >= best_acc:
				if phase == 'val' and epoch_loss <= best_loss:
					#best_acc = epoch_acc
					best_loss = epoch_loss
					best_model_wts = copy.deepcopy(self.model.state_dict())
					# Save trained model
					torch.save(self.model.state_dict(),'pytorch_model.h5')

			print()

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))
		#print('Best val Acc: {:4f}'.format(best_acc))

		# Generate plots
		plt.figure(); plt.plot(range(1,nb_epochs+1),train_loss,'k', range(1,nb_epochs+1), val_loss, 'r')
		plt.legend(['Train Loss','Val Loss'])
		plt.savefig(os.getcwd()+ '/loss.png')

		# plt.figure(); plt.plot(range(1,nb_epochs+1),train_acc,'k', range(1,nb_epochs+1), val_acc, 'r')
		# plt.legend(['Train Accuracy','Val Accuracy'])
		# plt.savefig(os.getcwd()+ '/acc.png')

		# load best model weights
		self.model.load_state_dict(best_model_wts)

		# Save trained model
		torch.save(self.model.state_dict(),'pytorch_model.h5')


	def test_model(self, dataloaders, nb_image_layers=120, tile_size=15):
		print("\nPrediction on validation data")
		was_training = self.model.training
		self.model.eval()
		#self.model.load_state_dict(torch.load('pytorch_model.h5'))
		#self.model.eval()
		total_labels = []
		total_preds = []
		running_loss = 0.0

		with torch.no_grad():
			#for i, (inputs1, inputs2, GroundTruth) in enumerate(dataloaders['val']):
			for patch_idx, patches_batch in enumerate(dataloaders['val']):
				print('\t patch_idx: ', patch_idx)
				inputs = patches_batch['Combined'][tio.DATA]

				print('\t\t Preparing data...')
				input1_tiles, input2_tiles_real, GroundTruth_real = utils.prepare_data3x3(inputs,nb_image_layers,tile_size)
				print('\t\t Preparing data - done -')

				#print("DataLoader iteration: %d" % i)
				input1_tiles = input1_tiles.to(self.device)
				input2_tiles_real = input2_tiles_real.to(self.device)
				GroundTruth_real = GroundTruth_real.to(self.device)
					
				outputs = self.model(input1_tiles, input2_tiles_real)

				loss = torch.sqrt(self.criterion(outputs, GroundTruth_real))

				# statistics
				running_loss += loss.item() * input1_tiles.size(0)

			total_loss = running_loss / len(dataloaders['val'].dataset)


		# Total loss
		print("len(dataloaders['val'].dataset)",len(dataloaders['val'].dataset))
		print("len(dataloaders['val'])",len(dataloaders['val']))
		print('Loss: ', total_loss)

		self.model.train(mode=was_training)




	def majority_vote(self, preds, labels):

		num_per_img = self.num_img_split
		maj_vec = np.zeros((labels.shape[0]//num_per_img,))
		maj_labels = np.copy(maj_vec)
		
		for i in range(0,labels.shape[0],num_per_img):
			curr_mode,_ = scipy.stats.mode(preds[i:i+num_per_img])
			
			maj_vec[i//num_per_img] = curr_mode[0]
			maj_labels[i//num_per_img] = labels[i]

		acc = float(len(np.where(maj_vec == maj_labels)[0])) / len(maj_vec)
		print('Majority vote Acc = {:.6f}'.format(acc))


	def cov(self, attr, ftrs):
		mu_attr = np.mean(attr)
		mu_ftrs = np.mean(ftrs)

		vec = ftrs - mu_ftrs
		overall = 0.0
		for i in range(len(attr)):
			overall += np.sum((attr[i]-mu_attr) * vec)

		cov = overall / (len(attr) * len(ftrs))

		return cov


	def save_img(self, files, preds):

		p = preds.cpu().numpy()
		for i in range(len(files)):
			curr_name = os.getcwd() + '/predicted_img/' + files[i].split('/')[9][:-4] + '_pred.tif'
			curr_img = p[i,...]			
			#print(curr_img)
			scipy.misc.imsave(curr_name, curr_img)


	def plot_confusion_matrix(self, ma, classes,
						  normalize=False,
						  title='Confusion matrix',
						  cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.figure()
		if normalize:
			ma = ma.astype('float') / ma.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')
		

		plt.imshow(ma, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		fmt = '.2f' if normalize else 'd'
		thresh = ma.max() / 2.
		for i, j in itertools.product(range(ma.shape[0]), range(ma.shape[1])):
			plt.text(j, i, format(ma[i, j], fmt),
					 horizontalalignment="center",
					 color="white" if ma[i, j] > thresh else "black")

		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.tight_layout()
		plt.savefig(os.getcwd() + '/' + title + '.tif')


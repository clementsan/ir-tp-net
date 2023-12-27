
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

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

# from layers import *
from network import *
import utils
import dataset 

class Model(object):
	def __init__(self, writer, nb_image_layers, nb_corr_layers, tile_size, adjacent_tiles_dim, model_name, dict_fc_features, loss_name, data_filtering, confidence_threshold):

		self.writer = writer
		self.nb_image_layers = nb_image_layers
		self.nb_corr_layers = nb_corr_layers
		self.tile_size = tile_size
		self.adjacent_tiles_dim = adjacent_tiles_dim
		self.InputDepth = self.nb_corr_layers
		# Model name - for saving
		self.model_name = model_name
		self.dict_fc_features = dict_fc_features
		self.loss_name = loss_name
		self.data_filtering = data_filtering
		self.confidence_threshold = confidence_threshold

		# Criterion MSE -> loss RMSE
		self.criterion = nn.MSELoss()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		# Create ResNet50 model
		self.create_model()


	def create_model(self):
				
		# Dynamic network with parallel subnets (e.g. for 3x3, 5x5 neighboring tiles)
		self.model = MyParallelNetwork(self.InputDepth, self.tile_size, self.adjacent_tiles_dim, self.dict_fc_features)
		
		print(self.model)

		# Attach to device
		self.model = self.model.to(self.device)
		


	# Need to udpate: step1 vs step2
	def train_model(self, dataloaders, lr, nb_epochs=25):
		since = time.time()

		# Unfreeze all layers
		# for param in self.model.parameters():
		# 	param.requires_grad = True

		# Observe that all parameters are being optimized
		#optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
		optimizer = optim.AdamW(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)

		# Decay LR by a factor of 0.1 every 7 epochs
		#scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

		best_model_wts = copy.deepcopy(self.model.state_dict())
		#best_acc = 0.0
		best_loss = 100000

		train_loss = []
		val_loss = []
		#train_acc = []
		#val_acc = []
		train_PercentFiltering = []
		val_PercentFiltering = []

		for epoch in range(nb_epochs):
			print('-' * 10)
			print('Epoch {}/{}'.format(epoch, nb_epochs - 1))
			
			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				time_beginepoch = time.time()
				if phase == 'train':
					#scheduler.step()
					self.model.train()  # Set model to training mode
				else:
					self.model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				running_patch_size = [] 
				#running_corrects = 0

				# Iterate over data.
				#for inputs1, inputs2, GroundTruth in dataloaders[phase]:
				for patch_idx, patches_batch in enumerate(dataloaders[phase]):
					#print('\t patch_idx: ', patch_idx)
					inputs = patches_batch['Combined'][tio.DATA]
					locations = patches_batch[tio.LOCATION]

					inputs = inputs.to(self.device)
					# locations = locations.to(self.device)

					# Data filtering: exclude patches based on DispLMA and confidence maps
					# if self.data_filtering:
					# 	inputs_filtered, locations_filtered = dataset.data_filtering(inputs, locations, self.confidence_threshold)
					# else:
					# 	inputs_filtered, locations_filtered = inputs, locations

					# #print('\t\t Preparing data...')
					# input_Corr_tiles, input_TargetDisp_tiles_real, GroundTruth_real, Confidence_real, DispLMA_real = dataset.prepare_data(inputs_filtered, self.nb_image_layers, self.nb_corr_layers, self.tile_size, self.adjacent_tiles_dim)
					# #print('\t\t Preparing data - done -')
					
					input_Corr_tiles, input_TargetDisp_tiles_real, GroundTruth_real = dataset.prepare_data_withfiltering(inputs, self.nb_image_layers, self.nb_corr_layers, self.tile_size, self.adjacent_tiles_dim, self.data_filtering, self.confidence_threshold)
					


					#input_Corr_tiles = input_Corr_tiles.to(self.device)
					#input_TargetDisp_tiles_real = input_TargetDisp_tiles_real.to(self.device)
					#GroundTruth_real = GroundTruth_real.to(self.device)
					# Reducing last dimension to compute loss
					GroundTruth_real = torch.squeeze(GroundTruth_real, dim=2)
					
					# zero the parameter gradients
					optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						#print('\t\t DNN - forward...')
						# Provide two inputs to model
						outputs = self.model(input_Corr_tiles, input_TargetDisp_tiles_real)
						#print('\t\t DNN - computing loss...')
						loss = torch.sqrt(self.criterion(outputs, GroundTruth_real))
						
						# backward + optimize only if in training phase
						if phase == 'train':
							#print('\t\t DNN - backward...')
							loss.backward()
							optimizer.step()
						#print('\t patch - done -')
					
					# statistics
					#print('\t running_loss...')
					running_loss += loss.item() * input_Corr_tiles.size(0)
					running_patch_size.append(input_Corr_tiles.size(0))
					#print('\t\t patch_size: ', input_Corr_tiles.size(0))
					#print('\t running_loss - done -')
					#running_corrects += torch.sum(preds == labels.data)
					

				if self.data_filtering:
					epoch_loss = running_loss / sum(running_patch_size)
				else:
					epoch_loss = running_loss / len(dataloaders[phase].dataset) 
				#epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) 
				PercentFiltering = sum(running_patch_size) / len(dataloaders[phase].dataset)

				curr_lr = optimizer.param_groups[0]['lr']
				print('{} Loss: {:.4f} Lr: {:.6f}'.format(
					phase, epoch_loss, curr_lr))
				print('{} Data Filtering: {:.4f}'.format(
					phase, PercentFiltering))
				

				# Append values for plots
				if phase == 'train':
					train_loss.append(epoch_loss)
					#train_acc.append(epoch_acc)
					self.writer.add_scalar('Loss/train', epoch_loss, epoch)
					train_PercentFiltering.append(PercentFiltering)
				else:
					val_loss.append(epoch_loss)
					#val_acc.append(epoch_acc)
					self.writer.add_scalar('Loss/val', epoch_loss, epoch)
					val_PercentFiltering.append(PercentFiltering)

				# deep copy the model
				#if phase == 'val' and epoch_acc >= best_acc:
				if phase == 'val' and epoch_loss <= best_loss:
					#best_acc = epoch_acc
					best_loss = epoch_loss
					best_model_wts = copy.deepcopy(self.model.state_dict())
					# Save trained model
					torch.save(self.model.state_dict(),self.model_name)

				time_epoch = time.time() - time_beginepoch
				print('--- {} epoch in {:.2f}s---'.format(phase, time_epoch))

		print('-' * 10)
		print('Best val loss: {:.4f}'.format(best_loss))
		print('Average data filtering - train: {:.4f}'.format(sum(train_PercentFiltering) / nb_epochs))
		print('Average data filtering - val: {:.4f}'.format(sum(val_PercentFiltering) / nb_epochs))

		time_elapsed = time.time() - since
		print('\nTraining complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60))
		#print('Best val Acc: {:4f}'.format(best_acc))

		# Generate plots
		plt.figure(); plt.plot(range(1,nb_epochs+1),train_loss,'k', range(1,nb_epochs+1), val_loss, 'r')
		plt.legend(['Train Loss','Val Loss'])
		plt.savefig(self.loss_name)

		# plt.figure(); plt.plot(range(1,nb_epochs+1),train_acc,'k', range(1,nb_epochs+1), val_acc, 'r')
		# plt.legend(['Train Accuracy','Val Accuracy'])
		# plt.savefig(os.getcwd()+ '/acc.png')

		# load best model weights
		self.model.load_state_dict(best_model_wts)

		# Save trained model
		torch.save(self.model.state_dict(), self.model_name)


	def test_model(self, dataloaders):
		print("\nPrediction on validation data")
		was_training = self.model.training
		self.model.eval()
		#self.model.load_state_dict(torch.load(self.model_name))
		#self.model.eval()
		total_labels = []
		total_preds = []
		running_loss = 0.0
		running_patch_size = [] 

		with torch.no_grad():
			#for i, (inputs1, inputs2, GroundTruth) in enumerate(dataloaders['val']):
			for patch_idx, patches_batch in enumerate(dataloaders['val']):
				print('\t patch_idx: ', patch_idx)
				inputs = patches_batch['Combined'][tio.DATA]

				#print('\t\t Preparing data...')
				input_Corr_tiles, input_TargetDisp_tiles_real, GroundTruth_real = dataset.prepare_data_withfiltering(inputs, self.nb_image_layers, self.nb_corr_layers, self.tile_size, self.adjacent_tiles_dim, self.data_filtering, self.confidence_threshold)
				#print('\t\t Preparing data - done -')

				#print("DataLoader iteration: %d" % i)
				input_Corr_tiles = input_Corr_tiles.to(self.device)
				input_TargetDisp_tiles_real = input_TargetDisp_tiles_real.to(self.device)
				GroundTruth_real = GroundTruth_real.to(self.device)
				# Reducing last dimension to compute loss
				GroundTruth_real = torch.squeeze(GroundTruth_real, dim=2)
						
				outputs = self.model(input_Corr_tiles, input_TargetDisp_tiles_real)

				loss = torch.sqrt(self.criterion(outputs, GroundTruth_real))

				# statistics
				running_loss += loss.item() * input_Corr_tiles.size(0)
				running_patch_size.append(input_Corr_tiles.size(0))
			
			if self.data_filtering:
				total_loss = running_loss / sum(running_patch_size)
			else:
				total_loss = running_loss / len(dataloaders['val'].dataset)
			PercentFiltering = sum(running_patch_size) / len(dataloaders['val'].dataset)

		# Total loss
		print('Validation Loss: {:.4f}'.format(total_loss))
		print('Validation Data Filtering: {:.4f}'.format(PercentFiltering))
				
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


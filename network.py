from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms


# FC network - 4 hidden layers
#  - Max pool on input 2
#  - Flatten inputs
#  - Concatenate inputs
#  - 4 hidden layers 
class MySingleNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		
		# Initial parameters
		tile_size = 15
		nb_image_pairs = 120

		# Need to add batch normalization?
		self.flatten1 = nn.Flatten()
		self.flatten2 = nn.Flatten()
		self.MaxPool2d = nn.MaxPool2d(tile_size)

		nb_input_features = nb_image_pairs * tile_size * tile_size + 1 # 120*15*15+1 #27001
		self.fc1 = nn.Linear(nb_input_features, 10000)
		self.BN1 = torch.nn.BatchNorm1d(10000)
		#self.drop1 = nn.Dropout(p=0.25)

		self.fc2 = nn.Linear(10000, 5000)
		self.BN2 = torch.nn.BatchNorm1d(5000)
		#self.drop2 = nn.Dropout(p=0.25)

		self.fc3 = nn.Linear(5000, 1000)
		self.BN3 = torch.nn.BatchNorm1d(1000)
		#self.drop3 = nn.Dropout(p=0.25)

		self.fc4 = nn.Linear(1000, 500)
		self.BN4 = torch.nn.BatchNorm1d(500)
		#self.drop4 = nn.Dropout(p=0.25)

		self.fc5 = nn.Linear(500, 100)
		self.BN5 = torch.nn.BatchNorm1d(100)
		self.drop5 = nn.Dropout(p=0.25)

		self.fc6 = nn.Linear(100, 1)


	def forward(self, x1, x2):
		x1 = self.flatten1(x1)
		x2 = self.MaxPool2d(x2)
		x2 = self.flatten2(x2)
		x = torch.cat((x1,x2),1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.BN1(x)
		#x = self.drop1(x)

		x = self.fc2(x)
		x = F.relu(x)
		x = self.BN2(x)
		#x = self.drop2(x)

		x = self.fc3(x)
		x = F.relu(x)
		x = self.BN3(x)
		#x = self.drop3(x)

		x = self.fc4(x)
		x = F.relu(x)
		x = self.BN4(x)
		#x = self.drop4(x)

		x = self.fc5(x)
		x = F.relu(x)
		x = self.BN5(x)
		x = self.drop5(x)

		out = self.fc6(x)
	
		return out



# Phase 1 - Sub-network
class MySubNetworkPhase1(nn.Module):
	def __init__(self, nb_image_layers, tile_size, neighboring_tiles):
		super().__init__()

		self.nb_image_layers = nb_image_layers
		self.tile_size = tile_size
		self.neighboring_tiles = neighboring_tiles

		#print('SUBNETPHASE1 - nb_image_layers',self.nb_image_layers)
		#print('SUBNETPHASE1 - neighboring_tiles',self.neighboring_tiles)
		

		list_nbfeatures = [2048,512,256,64]

		self.flatten1 = nn.Flatten()
		self.flatten2 = nn.Flatten()

		nb_input_features = self.nb_image_layers * self.tile_size * self.tile_size + 1 # 120*15*15+1 #27001
		#print('SUBNETPHASE1 - nb_input_features',nb_input_features)

		self.fc1 = nn.Linear(nb_input_features, list_nbfeatures[0])
		self.BN1 = torch.nn.BatchNorm1d(list_nbfeatures[0])
		#self.drop1 = nn.Dropout(p=0.25)

		self.fc2 = nn.Linear(list_nbfeatures[0], list_nbfeatures[1])
		self.BN2 = torch.nn.BatchNorm1d(list_nbfeatures[1])
		#self.drop2 = nn.Dropout(p=0.25)

		self.fc3 = nn.Linear(list_nbfeatures[1], list_nbfeatures[2])
		self.BN3 = torch.nn.BatchNorm1d(list_nbfeatures[2])
		#self.drop3 = nn.Dropout(p=0.25)

		self.fc4 = nn.Linear(list_nbfeatures[2], list_nbfeatures[3])
		self.BN4 = torch.nn.BatchNorm1d(list_nbfeatures[3])
		#self.drop4 = nn.Dropout(p=0.25)

	def forward(self, x1, x2):
		x1 = self.flatten1(x1)
		x2 = self.flatten2(x2)
		x = torch.cat((x1,x2),1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.BN1(x)
		#x = self.drop1(x)

		x = self.fc2(x)
		x = F.relu(x)
		x = self.BN2(x)
		#x = self.drop2(x)

		x = self.fc3(x)
		x = F.relu(x)
		x = self.BN3(x)
		#x = self.drop3(x)

		out = self.fc4(x)
	
		return out


# sub-Network - Phase2
class MySubNetworkPhase2(nn.Module):
	def __init__(self, neighboring_tiles, fc_inputfeatures):
		super().__init__()
		
		self.neighboring_tiles = neighboring_tiles
		self.fc_inputfeatures = fc_inputfeatures
		nb_input_features = self.neighboring_tiles * self.neighboring_tiles * self.fc_inputfeatures

		list_nbfeatures = [128,64,32]

		self.fc1 = nn.Linear(nb_input_features, list_nbfeatures[0])
		self.BN1 = torch.nn.BatchNorm1d(list_nbfeatures[0])
		#self.drop1 = nn.Dropout(p=0.25)

		self.fc2 = nn.Linear(list_nbfeatures[0], list_nbfeatures[1])
		self.BN2 = torch.nn.BatchNorm1d(list_nbfeatures[1])
		#self.drop2 = nn.Dropout(p=0.25)

		self.fc3 = nn.Linear(list_nbfeatures[1], list_nbfeatures[2])
		self.BN3 = torch.nn.BatchNorm1d(list_nbfeatures[2])
		#self.drop3 = nn.Dropout(p=0.25)

		self.fc4 = nn.Linear(list_nbfeatures[2], 1)


	def forward(self, x):

		x = self.fc1(x)
		x = F.relu(x)
		x = self.BN1(x)
		#x = self.drop1(x)

		x = self.fc2(x)
		x = F.relu(x)
		x = self.BN2(x)
		#x = self.drop2(x)

		x = self.fc3(x)
		x = F.relu(x)
		x = self.BN3(x)
		#x = self.drop3(x)

		out = self.fc4(x)
	
		return out


# Parallel network for neighboring tiles (e.g. 3x3, 5x5)
#  - Phase1 : 3x3 or 5x5 parallel sub-networks with FC layers
#  - Concatenating 3x3 or 5x5 features
#  - Phase 2 - single sub-network with FC layers
class MyParallelNetwork(nn.Module):
	def __init__(self, nb_image_layers, tile_size, neighboring_tiles):
		super().__init__()
		
		self.nb_image_layers = nb_image_layers
		self.tile_size = tile_size
		self.neighboring_tiles = neighboring_tiles
		
		# Number of sub-networks: 5x5
		self.nb_subnetworks = self.neighboring_tiles * self.neighboring_tiles
		
		#print('NETWORK - nb_image_layers',self.nb_image_layers)
		#print('NETWORK - neighboring_tiles',self.neighboring_tiles)
		#print('NETWORK - nb_subnetworks',self.nb_subnetworks)

		# define ModuleList of subnetworks
		self.subnetworks = nn.ModuleList([MySubNetworkPhase1(self.nb_image_layers, self.tile_size, self.neighboring_tiles) for i in range(self.nb_subnetworks)])
		
		self.Phase2 = MySubNetworkPhase2(self.neighboring_tiles, 64)

		# Need to add batch normalization?
		self.flatten1 = nn.Flatten()

	def forward(self, x1, x2):
		# x1 & x2 = list of 5x5 neighboring tiles
 
		outputs_subnetworks = [net(x1[:,i,...], x2[:,i,...]) for i, net in enumerate(self.subnetworks)]
		#print('NETWORK - len outputs_subnetworks',len(outputs_subnetworks))
		#print('NETWORK - outputs_subnetworks[0] shape',outputs_subnetworks[0].shape)
		out = torch.cat((outputs_subnetworks), dim = 1)
		#print('NETWORK - out shape',out.shape)
		out = self.Phase2(out)
		
		return out


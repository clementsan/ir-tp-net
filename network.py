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


class MySubNetworkPhase1(nn.Module):
	def __init__(self, nb_image_layers, tile_size, adjacent_tiles_dim, list_fc_features):
		super().__init__()

		self.nb_image_layers = nb_image_layers
		self.tile_size = tile_size
		self.adjacent_tiles_dim = adjacent_tiles_dim

		#print('SUBNETPHASE1 - nb_image_layers',self.nb_image_layers)
		#print('SUBNETPHASE1 - adjacent_tiles_dim',self.adjacent_tiles_dim)
		
		self.list_fc_features = list_fc_features

		self.flatten1 = nn.Flatten()
		self.flatten2 = nn.Flatten()

		nb_input_features = self.nb_image_layers * self.tile_size * self.tile_size # 120*15*15 #27000
		#print('SUBNETPHASE1 - nb_input_features',nb_input_features)

		self.fc1 = nn.Linear(nb_input_features, self.list_fc_features[0])
		self.BN1 = torch.nn.BatchNorm1d(self.list_fc_features[0])
		#self.drop1 = nn.Dropout(p=0.25)

		self.fc2 = nn.Linear(self.list_fc_features[0], self.list_fc_features[1])
		self.BN2 = torch.nn.BatchNorm1d(self.list_fc_features[1])
		#self.drop2 = nn.Dropout(p=0.25)

		self.fc3 = nn.Linear(self.list_fc_features[1], self.list_fc_features[2])
		self.BN3 = torch.nn.BatchNorm1d(self.list_fc_features[2])
		#self.drop3 = nn.Dropout(p=0.25)

		# Concatenation before FC4 (input features + 1)
		self.fc4 = nn.Linear(self.list_fc_features[2]+1, self.list_fc_features[3])
		self.BN4 = torch.nn.BatchNorm1d(self.list_fc_features[3])
		#self.drop4 = nn.Dropout(p=0.25)

	def forward(self, x1, x2):
		x1 = self.flatten1(x1)
		x2 = self.flatten2(x2)

		x = self.fc1(x1)
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

		x = torch.cat((x,x2),1)

		out = self.fc4(x)
	
		return out


# sub-Network - Phase2
class MySubNetworkPhase2(nn.Module):
	def __init__(self, fc_inputfeatures, list_fc_features):
		super().__init__()
		
		# Input features after concatenation e.g. [3x3x64] or [5x5x64] 
		self.fc_inputfeatures = fc_inputfeatures

		#list_nbfeatures = [128,64,32]
		self.list_fc_features = list_fc_features

		self.fc1 = nn.Linear(self.fc_inputfeatures, self.list_fc_features[0])
		self.BN1 = torch.nn.BatchNorm1d(self.list_fc_features[0])
		#self.drop1 = nn.Dropout(p=0.25)

		self.fc2 = nn.Linear(self.list_fc_features[0], self.list_fc_features[1])
		self.BN2 = torch.nn.BatchNorm1d(self.list_fc_features[1])
		#self.drop2 = nn.Dropout(p=0.25)

		self.fc3 = nn.Linear(self.list_fc_features[1], self.list_fc_features[2])
		self.BN3 = torch.nn.BatchNorm1d(self.list_fc_features[2])
		#self.drop3 = nn.Dropout(p=0.25)

		self.fc4 = nn.Linear(self.list_fc_features[2], 1)


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
#  - Phase1 : dynamic parallel sub-networks with FC layers (e.g. 3x3 or 5x5)
#  - Concatenating 3x3 or 5x5 features
#  - Phase 2 - single sub-network with FC layers
class MyParallelNetwork(nn.Module):
	def __init__(self, nb_image_layers, tile_size, adjacent_tiles_dim, dict_fc_features):
		super().__init__()
		
		self.nb_image_layers = nb_image_layers
		self.tile_size = tile_size
		self.adjacent_tiles_dim = adjacent_tiles_dim
		
		# Number of sub-networks: 5x5
		self.nb_subnetworks = self.adjacent_tiles_dim * self.adjacent_tiles_dim
		
		# FC features
		self.dict_fc_features = dict_fc_features
		self.Phase2_InputFeatures = self.adjacent_tiles_dim * self.adjacent_tiles_dim * self.dict_fc_features['Phase1'][-1]


		#print('NETWORK - nb_image_layers',self.nb_image_layers)
		#print('NETWORK - adjacent_tiles_dim',self.adjacent_tiles_dim)
		#print('NETWORK - nb_subnetworks',self.nb_subnetworks)

		# define ModuleList of subnetworks
		self.Phase1_subnetworks = nn.ModuleList([MySubNetworkPhase1(self.nb_image_layers, self.tile_size, self.adjacent_tiles_dim, self.dict_fc_features['Phase1']) for i in range(self.nb_subnetworks)])
		
		self.Phase2_net = MySubNetworkPhase2(self.Phase2_InputFeatures, self.dict_fc_features['Phase2'])

		# Need to add batch normalization?
		self.flatten1 = nn.Flatten()

	def forward(self, x1, x2):
		# Phase 1 - Parallel subnets
		# x1 & x2 = list of 5x5 neighboring tiles
		outputs_subnetworks = [Phase1_net(x1[:,i,...], x2[:,i,...]) for i, Phase1_net in enumerate(self.Phase1_subnetworks)]
		#print('NETWORK - len outputs_subnetworks',len(outputs_subnetworks))
		#print('NETWORK - outputs_subnetworks[0] shape',outputs_subnetworks[0].shape)

		# Concatenating outputs of subnets
		out_Phase1 = torch.cat((outputs_subnetworks), dim = 1)
		#print('NETWORK - out_Phase1 shape',out_Phase1.shape)

		# Phase 2 - FC layers
		out_Phase2 = self.Phase2_net(out_Phase1)
		
		return out_Phase2


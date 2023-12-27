
import torch
import torch.nn as nn
import torch.nn.functional as F

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

		self.fc4 = nn.Linear(self.list_fc_features[2], self.list_fc_features[3])
		#self.BN4 = torch.nn.BatchNorm1d(self.list_fc_features[3])
		#self.drop4 = nn.Dropout(p=0.25)

	def forward(self, x1):
		x1 = self.flatten1(x1)

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

		x = self.fc4(x)
		out = F.relu(x)
		#out = self.BN4(out)
		#out = self.drop4(out)

		return out


# sub-Network - Phase2
class MySubNetworkPhase2(nn.Module):
	def __init__(self, fc_inputfeatures, list_fc_features):
		super().__init__()
		
		# Input features after concatenation e.g. [3x3x64] or [5x5x64] 
		self.fc_inputfeatures = fc_inputfeatures

		#list_nbfeatures = [128,64,32]
		self.list_fc_features = list_fc_features

		self.BN0 = torch.nn.BatchNorm1d(self.fc_inputfeatures)
		self.drop0 = nn.Dropout(p=0.25)

		self.fc1 = nn.Linear(self.fc_inputfeatures, self.list_fc_features[0])
		self.BN1 = torch.nn.BatchNorm1d(self.list_fc_features[0])
		self.drop1 = nn.Dropout(p=0.20)

		self.fc2 = nn.Linear(self.list_fc_features[0], self.list_fc_features[1])
		self.BN2 = torch.nn.BatchNorm1d(self.list_fc_features[1])
		self.drop2 = nn.Dropout(p=0.10)

		self.fc3 = nn.Linear(self.list_fc_features[1], self.list_fc_features[2])
		self.BN3 = torch.nn.BatchNorm1d(self.list_fc_features[2])
		#self.drop3 = nn.Dropout(p=0.25)

		self.fc4 = nn.Linear(self.list_fc_features[2], 1)

		self.flatten2 = nn.Flatten()

	def forward(self, x, x2):

		x = self.BN0(x)
		x = self.drop0(x)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.BN1(x)
		x = self.drop1(x)

		x = self.fc2(x)
		x = F.relu(x)
		x = self.BN2(x)
		x = self.drop2(x)

		x = self.fc3(x)
		x = F.relu(x)
		x = self.BN3(x)
		#x = self.drop3(x)

		x = self.fc4(x)
	
		x2 = self.flatten2(x2)
		out = torch.add(x,x2)

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


	def forward(self, x1, x2):
		# Phase 1 - Parallel subnets
		# x1 & x2 = list of 5x5 neighboring tiles
		outputs_subnetworks = [Phase1_net(x1[:,i,...]) for i, Phase1_net in enumerate(self.Phase1_subnetworks)]
		#print('NETWORK - len outputs_subnetworks',len(outputs_subnetworks))
		#print('NETWORK - outputs_subnetworks[0] shape',outputs_subnetworks[0].shape)

		# Concatenating outputs of subnets
		out_Phase1 = torch.cat((outputs_subnetworks), dim = 1)
		#print('NETWORK - out_Phase1 shape',out_Phase1.shape)

		# Phase 2 - FC layers
		out_Phase2 = self.Phase2_net(out_Phase1, x2)
		
		return out_Phase2


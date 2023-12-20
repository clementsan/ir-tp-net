from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms


class MyNetwork(nn.Module):
	def __init__(self):
		super(MyNetwork, self).__init__()
		
		# Need to add batch normalization?
		self.flatten1 = nn.Flatten()
		self.flatten2 = nn.Flatten()
		nb_input_features = 15*15 + 1 #226
		self.fc1 = nn.Linear(nb_input_features, 100)
		self.BN2 = torch.nn.BatchNorm1d(100)
		self.drop2 = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(100, 50)
		self.BN3 = torch.nn.BatchNorm1d(50)
		self.drop3 = nn.Dropout(p=0.5)
		self.fc3 = nn.Linear(50, 1)


	def forward(self, x1, x2):
		x1 = self.flatten1(x1)
		x2 = self.flatten2(x2)
		x = torch.cat((x1,x2),1)

		x = self.fc1(x)
		x = F.relu(x)
		x = self.BN2(x)
		x = self.drop2(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.BN3(x)
		x = self.drop3(x)
		out = self.fc3(x)
	
		return out






# Network similar to fastAI (with shared weights)
class MyNetworkFastAI2(nn.Module):
	def __init__(self):
		super(MyNetworkFastAI2, self).__init__()

		num_ftrs = 4096
		num_classes = 4

		mymodel = models.resnet50(pretrained=True)
		self.model_class1 = nn.Sequential(*list(mymodel.children())[:-2])

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)

		# Change output to 512 features
		# self.fc1 = nn.Linear(num_ftrs*4, 4096)
		# self.fc2 = nn.Linear(4096, 512)
		# self.fc = nn.Linear(512, num_classes)
		
		# self.BN1 = torch.nn.BatchNorm1d(num_ftrs*4)
		# self.BN2 = torch.nn.BatchNorm1d(4096)
		# self.BN3 = torch.nn.BatchNorm1d(512)
		
		# self.drop1 = torch.nn.Dropout(p=0.25)
		# self.drop2 = torch.nn.Dropout(p=0.25)
		# self.drop3 = torch.nn.Dropout(p=0.5)

		self.BN1 = torch.nn.BatchNorm1d(num_ftrs*4)
		self.drop1 = torch.nn.Dropout(p=0.25)
		self.fc1 = nn.Linear(num_ftrs*4, 4096)
		self.BN2 = torch.nn.BatchNorm1d(4096)
		self.drop2 = torch.nn.Dropout(p=0.25)
		self.fc2 = nn.Linear(4096, 512)
		self.BN3 = torch.nn.BatchNorm1d(512)
		self.drop3 = torch.nn.Dropout(p=0.5)
		self.fc = nn.Linear(512, num_classes)


	def forward(self, x1, x2, x3, x4):	

		avg_pool1 = self.avg_pool( self.model_class1(x1) )
		max_pool1 = self.max_pool( self.model_class1(x1) )
		# Added dropout
		ftrs1 = torch.squeeze(torch.cat((avg_pool1,max_pool1),1))

		avg_pool2 = self.avg_pool( self.model_class1(x2) )
		max_pool2 = self.max_pool( self.model_class1(x2) )
		ftrs2 = torch.squeeze(torch.cat((avg_pool2,max_pool2),1))

		avg_pool3 = self.avg_pool( self.model_class1(x3) )
		max_pool3 = self.max_pool( self.model_class1(x3) )
		ftrs3 = torch.squeeze(torch.cat((avg_pool3,max_pool3),1))

		avg_pool4 = self.avg_pool( self.model_class1(x4) )
		max_pool4 = self.max_pool( self.model_class1(x4) )
		ftrs4 = torch.squeeze(torch.cat((avg_pool4,max_pool4),1))

		# Concatenation: ftrs size = 4096 * 4
		concat_ftrs = torch.cat( (torch.cat( (torch.cat( (ftrs1, ftrs2), 1), ftrs3), 1), ftrs4), 1)
		x = self.BN1(concat_ftrs)
		x = self.drop1(x)
		x = self.fc1(x) # 4096 output features
		x = F.relu(x)
		x = self.BN2(x)
		x = self.drop2(x)
		x = self.fc2(x) # 512 output features
		x = F.relu(x)
		x = self.BN3(x)
		ftrs = self.drop3(x)
		out = self.fc(ftrs) # 4 output features

		return out, ftrs


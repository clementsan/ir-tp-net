#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import imageio

# Generate numpy array
Input_FileName = './Image_001_Combined.tiff'
Output_FileName_2D = './LabelTemplate_2D.tiff'
Output_FileName_3D = './LabelTemplate_3D.tiff'
Output_FileName_3D_Full = './LabelTemplate_3D_Full.tiff'
TileSize = 15

# Read input via imageio
print("\nReading input file - imageio...")
imageio_in = imageio.mimread(Input_FileName,memtest=False)
imageio_in = np.array(imageio_in)
InputFile_Shape = imageio_in.shape
Depth = InputFile_Shape[0]
NbTiles_H = InputFile_Shape[1] // TileSize
NbTiles_W = InputFile_Shape[2] // TileSize
print('\t imageio_in shape',imageio_in.shape)

# Generate Template Tile
print("\nGenerating Template Tile...")
TemplateTileZeros = np.zeros((TileSize,TileSize), dtype=np.int32)
TemplateTile = np.zeros((TileSize,TileSize), dtype=np.int32)
TemplateTile[7,7] = 1
print('\t TemplateTile shape',TemplateTile.shape)
print('\t TemplateTile type',TemplateTile.dtype)
print('\t TemplateTile voxel0',TemplateTile[0,0])
print('\t TemplateTile voxelcenter',TemplateTile[7,7])


# Generate 2D label template
print("\nGenerating label template...")
LabelTemplate_2D = np.tile(TemplateTile, (NbTiles_H,NbTiles_W))
print('\t LabelTemplate_2D type',LabelTemplate_2D.dtype)
print('\t LabelTemplate_2D shape',LabelTemplate_2D.shape)
print('\t LabelTemplate_2D voxel0',LabelTemplate_2D[0,0])
print('\t LabelTemplate_2D voxelcenter',LabelTemplate_2D[7,7])

# Save via imageio
print('Writing 2D output image - imageio...')
imageio.imwrite(Output_FileName_2D, LabelTemplate_2D)


# Generate 3D label template (tiles in center of 3D volume)
Zeros_2D = np.tile(TemplateTileZeros, (NbTiles_H,NbTiles_W))
Zeros_3D = np.expand_dims(Zeros_2D,axis=0)
print('\t Zeros_2D shape',Zeros_2D.shape)
Zeros_3D_Frames = np.repeat(Zeros_2D[np.newaxis, :, :], 60, axis=0)
print('\t Zeros_3D_Frames shape',Zeros_3D_Frames.shape)

LabelTemplate_CenterFrame = np.expand_dims(LabelTemplate_2D,axis=0)
print('\t LabelTemplate_CenterFrame shape',LabelTemplate_CenterFrame.shape)

LabelTemplate_3D = np.append(Zeros_3D_Frames, LabelTemplate_CenterFrame, axis=0)
LabelTemplate_3D = np.append(LabelTemplate_3D, LabelTemplate_CenterFrame, axis=0)
print('\t LabelTemplate_3D shape',LabelTemplate_3D.shape)
LabelTemplate_3D = np.append(LabelTemplate_3D, Zeros_3D_Frames, axis=0)
print('\t LabelTemplate_3D shape',LabelTemplate_3D.shape)

# Save via imageio
print('Writing 3D output image - imageio...')
imageio.mimwrite(Output_FileName_3D, LabelTemplate_3D)


# Generate 3D label template (tiles across all layers)
LabelTemplate_3D_Full = np.repeat(LabelTemplate_2D[np.newaxis, :, :], Depth, axis=0)
print('\t LabelTemplate_3D_Full shape',LabelTemplate_3D_Full.shape)
# Save via imageio
print('Writing 3D output image - imageio...')
imageio.mimwrite(Output_FileName_3D_Full, LabelTemplate_3D_Full)


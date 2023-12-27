# AI-based 3D depth sensing from stereo vision systems

Multi-input neural network to predict disparity maps 

Network name: ir-tp-net (infrared tile processing network)

Warning: This python library works on specific multi-layer TIFF images, that were generated via a complex preprocessing stage for stereo-vision (including image correction and 2D phase correlation). 


## Installation and requirements 

Installation using conda
```
> conda create --name <env_name> --file <conda_environment.yaml>
> conda activate <env_name>
```
OR 
```
> source activate <env_name>
```

General libraries being used:
	-	python 3.9 
	-	pillow, numpy, pandas, matplotlib 
	-	pytorch, torchio, imageio, tensorboard
	-	scikit-learn, yaml


## Execution

### DNN training using configuration file

Command line:
```
> python3 AI_Training.py --config AI_Training_Config.yaml
```
 
### DNN inference using configuration file

Command line:
```
> python3 AI_Inference_CSV.py --config AI_Inference_Config.yaml --verbose
```
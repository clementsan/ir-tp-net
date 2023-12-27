# AI-based 3D depth sensing from stereo vision systems

Multi-input neural network to predict disparity maps 

Network name: ir-tp-net (infrared tile processing network)


## Installation and requirements 

Installation using conda
> conda create --name <env_name> --file <conda_environment.yaml>
> conda activate <env_name>
OR 
> source activate <env_name>

General libraries being used:
	-	python 3.9 
	-	pillow, numpy, pandas, matplotlib 
	-	pytorch, torchio, imageio, tensorboard
	-	scikit-learn, yaml


## Execution

  2.1 DNN training using configuration file

Command line:
> python3 AI_Training.py --config AI_Training_Config.yaml
 
 
  2.2 DNN inference using configuration file

Command line:
 > python3 AI_Inference_CSV.py --config AI_Inference_Config.yaml --verbose

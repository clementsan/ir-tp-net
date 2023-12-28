# AI-based 3D depth sensing from stereo vision systems

Multi-input neural network to predict disparity maps 

Network name: ir-tp-net (infrared tile processing network)


## Notes

### ir-tp-net

ir-tp-net is a two-stage deep neural network (DNN). Stage 1 includes dynamic parallel sub-networks, that are then concatenated to form a single sub-network in stage 2.

This python library works on specific multi-layer TIFF images, that were generated via a complex preprocessing 
stage for multi-sensor stereo-vision (including image correction and 2D phase correlation).

### Related github repo

The additional github repository ["disp-map-analysis"](https://github.com/clementsan/disp-map-analysis) provides additional pre-processing and post-processing python scripts.


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
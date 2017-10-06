# Unsupervised Image to Image Translation Networks
This is the TensorFlow Implementation of the NIPS 2017 paper "Unsupervised Image to Image Translation Networks". 

## Introduction

We wrote the code as a research project before knowing the paper. Later we found the model and hyper-parameters are mostly consistent with what the paper describes. In particular, we tried the 'pix2pix' model which is the auto-encoder model described in the paper, and also the 'resnet' model made up of 9 blocks of resnet (middle blocks are shared). We found that the 'resnet' model gives better result than auto-encoder and (slightly) better results than CycleGAN.

Below is a snapshot of our result at the ? epoch using 'resnet' model with default parameters on the horse-zebra dataset:

## Getting Started
### Prepare dataset

* Download horse-zebra dataset (or any other dataset) and create a csv file containing the paths of the images in the dataset. 
	* Download a CycleGAN dataset (e.g. horse2zebra):
	```bash
	bash ./download_datasets.sh horse2zebra
	```
	```bash
	python -m CycleGAN_TensorFlow.create_cyclegan_dataset --image_path_a=folder_a --image_path_b=folder_b --dataset_name="horse2zebra_train" --do_shuffle=0
	```
	* Modify config.py based on where your csv file is (the trained models will also be saved in the folder).

### Train CycleGAN

* Use resnet model with default parameters:
```
 python -m Img2ImgTrans.main --split_name='horse_zebra' --cycle_lambda=15 --rec_lambda=1 --num_separate_layers_g=2 --num_separate_layers_d=5 --num_no_skip_layers=0 --lsgan_lambda_a=1 --lsgan_lambda_b=1 --network_structure='resnet'
```
* Use autoencoder model with default parameters:
```
python -m chameleon.main --split_name='horse_zebra' --cycle_lambda=30 --rec_lambda=10 --num_separate_layers_g=3 --num_separate_layers_d=5 --num_no_skip_layers=0 --lsgan_lambda_a=2 --lsgan_lambda_b=2
```

### Restore from previous checkpoints
```
 python -m Img2ImgTrans.main --split_name='horse_zebra' --cycle_lambda=15 --rec_lambda=1 --num_separate_layers_g=2 --num_separate_layers_d=5 --num_no_skip_layers=0 --lsgan_lambda_a=1 --lsgan_lambda_b=1 --network_structure='resnet' --checkpoint_dir=path/to/saved/checkpoint
```

### TensorBoard Output

### Visualization
Each epoch saves an html file to config dir.
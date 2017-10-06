# Unsupervised Image to Image Translation Networks
This is the TensorFlow Implementation of the NIPS 2017 paper "Unsupervised Image to Image Translation Networks".

## Usage

### Create dataset

##### Create the p2_mturbo dataset 
```
python -m dl_research.projects.chameleon.create_p2_mturbo_dataset --split_name='train'
```
##### Create the p2_gain dataset
```
notebooks/create_gain_dataset_for_cyclegan.ipynb
```
##### Create your own dataset
Create a csv file at /data2/tmp/CycleGAN with N rows and two columns. N is the number of pairs, and the columns contain image names with full path. Each row of the file is a pair of images from the two domains for CycleGAN training. 

### Train CycleGAN

```
./qsub-run.sh python -m dl_research.projects.chameleon.main \
    --split_name=p2_mturbo_train \
	--to_train=1 \
	--log_dir=output/cyclegan_nd/exp_01 \
	--config_filename=dl_research/projects/chameleon/configs/cyclegan_default.json 
```
Note: split_name is the name of the csv file. to_train could be 0 (test), 1 (train) and 2 (resume training from previous checkpoint). If to_train is set to 2, we need to provide the checkpoint_dir which points to the previous training point folder.

### Set the hyper-parameters
The weight of the cycle consistency loss could be configured in the json file with '_LAMBDA_A' and '_LAMBDA_B'. The base learning rate and max_step for training could also be configured. 

### Convert dataset
Convert the real dataset to the synthetic dataset using
```
./qsub-run.sh python -m dl_research.projects.chameleon.convert_dataset \
	--split_name=p2_mturbo_train \
	--checkpoint_dir=output/cyclegan_nd/p2_mturbo_more_cycle_weight/20170801-234226 \
	--save_dir=/data2/tmp/CycleGAN/fake_weight_50_epoch_45 \
	--prefix='/data2/dl-data-partner/'
```
The prefix is the part of the path of the original file name. It will convert each pair using the trained CycleGAN and save it to the save_dir. 

### Train CycleGAN with classification loss
```
./qsub-run.sh python -m dl_research.projects.chameleon.main_classification \
	--split_name= p2_mturbo_labels_train \
	--to_train=1 \
	--log_dir=output/cyclegan_nd/p2_mturbo_class_weight_1 \
	--config_filename=dl_research/projects/chameleon/configs/p2_mturbo_class_weight_1.json 
```
Parameters including how often do we update the generator using classification loss, and the weight of the classification loss to the generator can be configured in the json file.


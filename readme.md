# FYP Report: CCDS24-0455
The instructions provided here are concise as this is merely a forked repository. For a better understanding of the how the code works, please visit the main [Github page](https://github.com/MIC-DKFZ/nnUNet) of nnUNet. Also, please check out the paper [here](https://arxiv.org/abs/1904.08128). This paper contains information on how nnU-net works, and more information on the datasets. 



Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNetv2_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the project.scripts in the [pyproject.toml](pyproject.toml) file.

All nnU-Net commands have a `-h` option which gives information on how to use them.



# Information
## Files Structure
The results of the trainings can be found in [nnUNet_results](nnUNet_results). Please refer to the folder names for the specific trainigs conducted for them. You can also check out [nnUNet_preprocessed](nnUNet_preprocessed) and [nnUNet_raw](nnUNet_raw) but the .json files generated there should have been copied over to [nnUNet_results](nnUNet_results).

## Edited Files

nnunetv2 contains the code for the repository. 

Under the folder [experiment_planners](nnunetv2/experiment_planning/experiment_planners), the planners can be found. The planners folder contains the files for changing the batch size/patch size and the target spacing for the FYP report. Afterwards, preprocessing can be done.

Under the folder [nnUNetTrainer](nnunetv2/training/nnUNetTrainer), the file [nnUnetTrainer.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) can be found. This file is the default file used for training. Since data augmentation is done on the fly during training, we simply need to edit this file. The edited versions of these files can be found under the folder [variants](nnunetv2/training/nnUNetTrainer/variants). The respective trainer files are used for different groups of data augmentations for the FYP report. 

## Datasets
1. [MSD dataset](http://medicaldecathlon.com/dataaws/)
2. [ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb)
3. [Brats2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
4. [Amos2022](https://zenodo.org/records/7262581)
5. [KiTs2023](https://github.com/neheller/kits23)

# Code Usage


## Environment Setup 

Use a virual environment to install the requirements.txt
```bash
pip install -r requirements.txt
```

## Starting 
Before using any of the codes in here, environment variables for nnUNet_raw, nnUNet_preprocessed and nnUNet_results need to be set up. Please refer to [here](documentation/set_environment_variables.md) on how to set up the environmental variables.

For more information on what these environmental variables are for, please refer to the guide from [here](documentation/setting_up_paths.md)

## Dataset conversion 
Please refer to the original page on [nnU-Net dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). 

nnU-Net requires the dataset to be in a certain format. Open the folder [dataset_conversion](nnunetv2/dataset_conversion) and run the curated python files for conversion. Please edit `if __name__ == "__main__":` and change the file paths accordingly. Alternatively if you are using the MSD dataset, you can use `nnUNetv2_convert_MSD_dataset` to convert the dataset. Read `nnUNetv2_convert_MSD_dataset -h` for usage instructions.





## Planning and Preprocessing/Training 
Please refer to [here](documentation/how_to_use_nnunet.md). 

This can be removed for subsequent trainings if it is not required(for eg, training the next fold) 
```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
Use `-overwrite_plans_name` if the default plan is not being used(for eg different target spacing/batch size). This will generate a new .json plans file. If not, the original .json plans file will be overwritten.

Use `--verify_dataset_integrity` to check datset validity. This is not strictly required.

For the training command 
```bash
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```
Use `-tr` to refer the trainer files. 

Use `-p` to refer to the planner files

## Example 
Here is an example using Windows. This does not include the data conversion step that should be done before these commands.
```bash
@echo off


call conda activate fyp

set nnUNet_results=C:\Users\linch\fyp\nnUNet_results
set nnUNet_preprocessed=C:\Users\linch\fyp\nnUNet_preprocessed
set nnUNet_raw=C:\Users\linch\fyp\nnUNet_raw

nnUNetv2_plan_and_preprocess -d 10 -overwrite_plans_name nnUNetPlans_batch_size_4 --verify_dataset_integrity
nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_probabilisticOversampling_050 -p nnUNetPlans_batch_size_4
```


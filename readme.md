# FYP Report: CCDS24-0455
The instructions provided here are concise as this is merely a forked repository.For a better understanding of the how the code works, please visit the main [Github page](https://github.com/MIC-DKFZ/nnUNet) of nnUNet. Also, please check out the paper [here](https://arxiv.org/abs/1904.08128). This paper contains information on how nnU-net works, and more information on the datasets. 

# Installation Instructions
It is recommend that you install nnU-Net in a virtual environment! Pip or anaconda are both fine. If you choose to 
compile PyTorch from source (see below), you will need to use conda instead of pip. 

Use a recent version of Python! 3.9 or newer is guaranteed to work!

**nnU-Net v2 can coexist with nnU-Net v1! Both can be installed at the same time.**

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**. For maximum speed, consider 
[compiling pytorch yourself](https://github.com/pytorch/pytorch#from-source) (experienced users only!). 
2) Install nnU-Net depending on your use case:
For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you can modify it as needed):

          ```bash
          git clone https://github.com/MIC-DKFZ/nnUNet.git
          cd nnUNet
          pip install -e .
          ```
3) nnU-Net needs to know where you intend to save raw data, preprocessed data and trained models. For this you need to
   set a few environment variables. Please follow the instructions [here](setting_up_paths.md).
4) (OPTIONAL) Install [hiddenlayer](https://github.com/waleedka/hiddenlayer). hiddenlayer enables nnU-net to generate
   plots of the network topologies it generates (see [Model training](how_to_use_nnunet.md#model-training)). 
To install hiddenlayer,
   run the following command:

    ```bash
    pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
    ```

Installing nnU-Net will add several new commands to your terminal. These commands are used to run the entire nnU-Net
pipeline. You can execute them from any location on your system. All nnU-Net commands have the prefix `nnUNetv2_` for
easy identification.

Note that these commands simply execute python scripts. If you installed nnU-Net in a virtual environment, this
environment must be activated when executing the commands. You can see what scripts/functions are executed by 
checking the project.scripts in the [pyproject.toml](pyproject.toml) file.

All nnU-Net commands have a `-h` option which gives information on how to use them.



# Information
## Edited Files

nnunetv2 contains the code for the repository. 

Under the folder [experiment_planners](nnunetv2/experiment_planning/experiment_planners), the planners can be found. The planners folder contains the files for changing the batch size/patch size and the target spacing for the FYP report. Afterwards, preprocessing can be done.

Under the folder [nnUNetTrainer](nnunetv2/training/nnUNetTrainer), the file [nnUnetTrainer.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) can be found. This file is the default file used for training. Since data augmentation is done on the fly during training, we simply need to edit this file. The edited versions of these files can be found under the folder [variants](nnunetv2/training/nnUNetTrainer/variants). The respective trainer files are used for different groups of data augmentations for the FYP report. 

# Code Usage

## Starting 
Before using any of the codes in here, environment variables for nnUNet_raw, nnUNet_preprocessed and nnUNet_results need to be set up. Please refer to [here](documentation/set_environment_variables.md) on how to set up the environmental variables.


For more information on what these environmental variables are for, please refer to the guide from [here](documentation/setting_up_paths.md)

## Dataset conversion 
Please refer to the original page on [nnU-Net dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md). 

nnU-Net requires the dataset to be in a certain format. Open the folder [dataset_conversion](nnunetv2/dataset_conversion) and run the curated python files for conversion. Please edit `if __name__ == "__main__":` and change the file paths accordingly. Alternatively if you are using the MSD dataset, you can use `nnUNetv2_convert_MSD_dataset` to convert the dataset. Read `nnUNetv2_convert_MSD_dataset -h` for usage instructions.





## Planning and Preprocessing/Training 
Please refer to [here](documentation/how_to_use_nnunet.md). 

For the training command `nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]`, please use -tr and -p to refer the the trainer and planner files respectively. 


## Example 
Here is an example using Windows. This does not include the data conversion step that should be done before these commands. Also, `nnUNetv2_plan_and_preprocess` can be removed for subsequent trainings if it is not required.

```bash
@echo off


call conda activate fyp

set nnUNet_results=C:\Users\linch\fyp\nnUNet_results
set nnUNet_preprocessed=C:\Users\linch\fyp\nnUNet_preprocessed
set nnUNet_raw=C:\Users\linch\fyp\nnUNet_raw

nnUNetv2_plan_and_preprocess -d 10 -overwrite_plans_name nnUNetPlans_batch_size_4 --verify_dataset_integrity
nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_probabilisticOversampling_050 -p nnUNetPlans_batch_size_4
```


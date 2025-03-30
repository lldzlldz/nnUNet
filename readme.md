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
For use as integrative **framework** (this will create a copy of the nnU-Net code on your computer so that you
   can modify it as needed):
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
checking the project.scripts in the [pyproject.toml](../pyproject.toml) file.

All nnU-Net commands have a `-h` option which gives information on how to use them.

#Training 


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

nnU-Net requires the dataset to be in a certain format. Use `nnUNetv2_convert_MSD_dataset` to convert the dataset. Read `nnUNetv2_convert_MSD_dataset -h` for usage instructions. Alternatively, just open the folder [dataset_conversion](nnunetv2/dataset_conversion) and use the curated python files for conversion.





## Planning and Preprocessing/Training 
Please refer to [here](documentation/how_to_use_nnunet.md). 

For the training command `nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]`, please use -tr and -p to refer the the trainer and planner files respectively. 


## Example 
Here is an example using Windows.

```bash
@echo off


call conda activate fyp

set nnUNet_results=C:\Users\linch\fyp\nnUNet_results
set nnUNet_preprocessed=C:\Users\linch\fyp\nnUNet_preprocessed
set nnUNet_raw=C:\Users\linch\fyp\nnUNet_raw

nnUNetv2_plan_and_preprocess -d 10 -overwrite_plans_name nnUNetPlans_batch_size_4 --verify_dataset_integrity
nnUNetv2_train 10 3d_fullres 0 -tr nnUNetTrainer_probabilisticOversampling_050 -p nnUNetPlans_batch_size_4
```






--------------------------------------------------------------------





# Welcome to the new nnU-Net!

Click [here](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) if you were looking for the old one instead.

Coming from V1? Check out the [TLDR Migration Guide](documentation/tldr_migration_guide_from_v1.md). Reading the rest of the documentation is still strongly recommended ;-)

## **2024-04-18 UPDATE: New residual encoder UNet presets available!**
Residual encoder UNet presets substantially improve segmentation performance.
They ship for a variety of GPU memory targets. It's all awesome stuff, promised! 
Read more :point_right: [here](documentation/resenc_presets.md) :point_left:

Also check out our [new paper](https://arxiv.org/pdf/2404.09556.pdf) on systematically benchmarking recent developments in medical image segmentation. You might be surprised!

# What is nnU-Net?
Image datasets are enormously diverse: image dimensionality (2D, 3D), modalities/input channels (RGB image, CT, MRI, microscopy, ...), 
image sizes, voxel sizes, class ratio, target structure properties and more change substantially between datasets. 
Traditionally, given a new problem, a tailored solution needs to be manually designed and optimized  - a process that 
is prone to errors, not scalable and where success is overwhelmingly determined by the skill of the experimenter. Even 
for experts, this process is anything but simple: there are not only many design choices and data properties that need to 
be considered, but they are also tightly interconnected, rendering reliable manual pipeline optimization all but impossible! 

![nnU-Net overview](documentation/assets/nnU-Net_overview.png)

**nnU-Net is a semantic segmentation method that automatically adapts to a given dataset. It will analyze the provided 
training cases and automatically configure a matching U-Net-based segmentation pipeline. No expertise required on your 
end! You can simply train the models and use them for your application**.

Upon release, nnU-Net was evaluated on 23 datasets belonging to competitions from the biomedical domain. Despite competing 
with handcrafted solutions for each respective dataset, nnU-Net's fully automated pipeline scored several first places on 
open leaderboards! Since then nnU-Net has stood the test of time: it continues to be used as a baseline and method 
development framework ([9 out of 10 challenge winners at MICCAI 2020](https://arxiv.org/abs/2101.00232) and 5 out of 7 
in MICCAI 2021 built their methods on top of nnU-Net, 
 [we won AMOS2022 with nnU-Net](https://amos22.grand-challenge.org/final-ranking/))!

Please cite the [following paper](https://www.google.com/url?q=https://www.nature.com/articles/s41592-020-01008-z&sa=D&source=docs&ust=1677235958581755&usg=AOvVaw3dWL0SrITLhCJUBiNIHCQO) when using nnU-Net:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
    method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.


## What can nnU-Net do for you?
If you are a **domain scientist** (biologist, radiologist, ...) looking to analyze your own images, nnU-Net provides 
an out-of-the-box solution that is all but guaranteed to provide excellent results on your individual dataset. Simply 
convert your dataset into the nnU-Net format and enjoy the power of AI - no expertise required!

If you are an **AI researcher** developing segmentation methods, nnU-Net:
- offers a fantastic out-of-the-box applicable baseline algorithm to compete against
- can act as a method development framework to test your contribution on a large number of datasets without having to 
tune individual pipelines (for example evaluating a new loss function)
- provides a strong starting point for further dataset-specific optimizations. This is particularly used when competing 
in segmentation challenges
- provides a new perspective on the design of segmentation methods: maybe you can find better connections between 
dataset properties and best-fitting segmentation pipelines?

## What is the scope of nnU-Net?
nnU-Net is built for semantic segmentation. It can handle 2D and 3D images with arbitrary 
input modalities/channels. It can understand voxel spacings, anisotropies and is robust even when classes are highly
imbalanced.

nnU-Net relies on supervised learning, which means that you need to provide training cases for your application. The number of 
required training cases varies heavily depending on the complexity of the segmentation problem. No 
one-fits-all number can be provided here! nnU-Net does not require more training cases than other solutions - maybe 
even less due to our extensive use of data augmentation. 

nnU-Net expects to be able to process entire images at once during preprocessing and postprocessing, so it cannot 
handle enormous images. As a reference: we tested images from 40x40x40 pixels all the way up to 1500x1500x1500 in 3D 
and 40x40 up to ~30000x30000 in 2D! If your RAM allows it, larger is always possible.

## How does nnU-Net work?
Given a new dataset, nnU-Net will systematically analyze the provided training cases and create a 'dataset fingerprint'. 
nnU-Net then creates several U-Net configurations for each dataset: 
- `2d`: a 2D U-Net (for 2D and 3D datasets)
- `3d_fullres`: a 3D U-Net that operates on a high image resolution (for 3D datasets only)
- `3d_lowres` → `3d_cascade_fullres`: a 3D U-Net cascade where first a 3D U-Net operates on low resolution images and 
then a second high-resolution 3D U-Net refined the predictions of the former (for 3D datasets with large image sizes only)

**Note that not all U-Net configurations are created for all datasets. In datasets with small image sizes, the 
U-Net cascade (and with it the 3d_lowres configuration) is omitted because the patch size of the full 
resolution U-Net already covers a large part of the input images.**

nnU-Net configures its segmentation pipelines based on a three-step recipe:
- **Fixed parameters** are not adapted. During development of nnU-Net we identified a robust configuration (that is, certain architecture and training properties) that can 
simply be used all the time. This includes, for example, nnU-Net's loss function, (most of the) data augmentation strategy and learning rate.
- **Rule-based parameters** use the dataset fingerprint to adapt certain segmentation pipeline properties by following 
hard-coded heuristic rules. For example, the network topology (pooling behavior and depth of the network architecture) 
are adapted to the patch size; the patch size, network topology and batch size are optimized jointly given some GPU 
memory constraint. 
- **Empirical parameters** are essentially trial-and-error. For example the selection of the best U-net configuration 
for the given dataset (2D, 3D full resolution, 3D low resolution, 3D cascade) and the optimization of the postprocessing strategy.

## How to get started?
Read these:
- [Installation instructions](documentation/installation_instructions.md)
- [Dataset conversion](documentation/dataset_format.md)
- [Usage instructions](documentation/how_to_use_nnunet.md)

Additional information:
- [Learning from sparse annotations (scribbles, slices)](documentation/ignore_label.md)
- [Region-based training](documentation/region_based_training.md)
- [Manual data splits](documentation/manual_data_splits.md)
- [Pretraining and finetuning](documentation/pretraining_and_finetuning.md)
- [Intensity Normalization in nnU-Net](documentation/explanation_normalization.md)
- [Manually editing nnU-Net configurations](documentation/explanation_plans_files.md)
- [Extending nnU-Net](documentation/extending_nnunet.md)
- [What is different in V2?](documentation/changelog.md)

Competitions:
- [AutoPET II](documentation/competitions/AutoPETII.md)

[//]: # (- [Ignore label]&#40;documentation/ignore_label.md&#41;)

## Where does nnU-Net perform well and where does it not perform?
nnU-Net excels in segmentation problems that need to be solved by training from scratch, 
for example: research applications that feature non-standard image modalities and input channels,
challenge datasets from the biomedical domain, majority of 3D segmentation problems, etc . We have yet to find a 
dataset for which nnU-Net's working principle fails!

Note: On standard segmentation 
problems, such as 2D RGB images in ADE20k and Cityscapes, fine-tuning a foundation model (that was pretrained on a large corpus of 
similar images, e.g. Imagenet 22k, JFT-300M) will provide better performance than nnU-Net! That is simply because these 
models allow much better initialization. Foundation models are not supported by nnU-Net as 
they 1) are not useful for segmentation problems that deviate from the standard setting (see above mentioned 
datasets), 2) would typically only support 2D architectures and 3) conflict with our core design principle of carefully adapting 
the network topology for each dataset (if the topology is changed one can no longer transfer pretrained weights!) 

## What happened to the old nnU-Net?
The core of the old nnU-Net was hacked together in a short time period while participating in the Medical Segmentation 
Decathlon challenge in 2018. Consequently, code structure and quality were not the best. Many features 
were added later on and didn't quite fit into the nnU-Net design principles. Overall quite messy, really. And annoying to work with.

nnU-Net V2 is a complete overhaul. The "delete everything and start again" kind. So everything is better 
(in the author's opinion haha). While the segmentation performance [remains the same](https://docs.google.com/spreadsheets/d/13gqjIKEMPFPyMMMwA1EML57IyoBjfC3-QCTn4zRN_Mg/edit?usp=sharing), a lot of cool stuff has been added. 
It is now also much easier to use it as a development framework and to manually fine-tune its configuration to new 
datasets. A big driver for the reimplementation was also the emergence of [Helmholtz Imaging](http://helmholtz-imaging.de), 
prompting us to extend nnU-Net to more image formats and domains. Take a look [here](documentation/changelog.md) for some highlights.

# Acknowledgements
<img src="documentation/assets/HI_Logo.png" height="100px" />

<img src="documentation/assets/dkfz_logo.png" height="100px" />

nnU-Net is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).

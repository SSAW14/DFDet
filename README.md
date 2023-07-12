# DFDet - DeepFakes Detection in the wild

This repository is for a journal submission:
> **Towards Sustainable Deepfake Recognition, Detection, and Segmentation by Re-Synthesis and Multi-Level Features** \
> Pengxiang Xu <sup>1,#</sup>, Yang He<sup>2,#</sup>, Ning Yu <sup>3</sup>, Margret Keuper<sup>4</sup>, Shanshan Zhang<sup>1</sup>, Jian Yang <sup>1</sup>, Mario Fritz <sup>2</sup> \
> <sup>1</sup> Nanjing University of Science and Technology, Nanjing, China \
> <sup>2</sup> CISPA Helmholtz Center for Information Security, Saarbrücken, Germany \
> <sup>3</sup> Salesforce AI Research, Palo Alto, United States \
> <sup>4</sup> University of Siegen, Germany \
> <sup>#</sup>Equal contribution.

## Introduction

This work is an extension of our previous conference paper "Beyond the Spectrum: Detecting Deepfakes via Re-synthesis" in IJCAI. In this repo, we release the code for the extension, i.e., Deepfake detection & segmentation. For the classification results, the source code is released at https://github.com/SSAW14/BeyondtheSpectrum.

## Updates
* **07/03/23**: ***Inference code and checkpoint open to public.***

## Setup
### 1. Environment
* CUDA: 11.1
* mmcv: 1.4.0
* apex: 0.1

### 2. Dataset
Please download [OpenForensics](https://sites:google:com/view/ltnghia/research/openforensics/) dataset.

### 3. Checkpoint
Please download the checkpoint of [Baseline](https://1drv.ms/u/s!Ak80-EOBRQsUiSRTkEkXMOWJhhfD) and [Ours+](https://1drv.ms/u/s!Ak80-EOBRQsUiR9c_uXJnGgzmy7k) (Fix SR, Pixel and Stage5) and please it in `./work_dirs/` folder.

### 4. Inference
We provide some samples from Openforensics in `./demo/sample/Test-Dev/` and `./demo/sample/Test-Challenge/` folders.
Evaluate on single image:
```
python demo/image_demo.py --img <IMAGE_PATH> --config <CONFIG_FILE> --checkpoint <CHECKPOINT_FILE> --output <OUTPUT_FILE>
```
For example, for baseline:
```
python demo/image_demo.py --img demo/sample/Test-Dev/4210dfb597.jpg --config configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_openforensics_baseline.py  --checkpoint work_dirs/Checkpoint_Baseline.pth --output demo/sample/output/4210dfb597_baseline.jpg
```
For Fix SR, Pixel and Stage5:
```
python demo/image_demo.py --img demo/sample/Test-Dev/4210dfb597.jpg --config configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_openforensics.py --checkpoint work_dirs/Checkpoint_Fix_SR_Pixel_Stage5_Aug_SR_Aug.pth --output demo/sample/output/4210dfb597.jpg
```


## Acknowledgement
*This project is built upon MMDetection and Swin-Transformer. Great thanks to them!*

MMDetection https://github.com/open-mmlab/mmdetection

Swin-Transformer https://github.com/microsoft/Swin-Transformer
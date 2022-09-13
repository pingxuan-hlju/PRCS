# CMPB: Convolutional bi-directional learning and spatial enhanced attentions for lung tumor segmentation
Built upon [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet), this repository provides the official PyTorch implementation of CMPB.

## How to use CMPB:
### 1. Requirements:
Linux, Python3.7+, Pytorch1.6+
### 2. Installation:
* Install nnU-Net as below
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
* Copy the python files in folder gru to nnunet
* Copy the python files in folder attention to nnunet
* Copy the python files in folder network_architecture to nnunet/network_architecture

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{
xuan2022cmpb,
title={Convolutional bi-directional learning and spatial enhanced attentions for lung tumor segmentation},
author={Ping Xuan and Bin Jiang and Hui Cui and Qiangguo Jin and Peng Cheng and Toshiya Nakaguchi and Tiangang Zhang and Changyang Li and Zhiyu Ning and Menghan Guo and Linlin Wang},
booktitle={Computer Methods and Programs in Biomedicine(under review)},
year={2022}
}
```

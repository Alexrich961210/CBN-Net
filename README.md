# Introduction

The source code of the proposed method is available online at Github.

## Abstract

This project contains the source code of the paper **Cellular Binary Neural Network for Accurate Classification**.

The demo consists of two parts. The first one contains the training implementation of  CBN-Net model on the CIFAR-10  and ImageNet dataset. The second one is about the evaluation of the CBN-Net on these two datasets. 

For easy evaluation, we also provide the pre-trained model parameters.

## Prerequisites

- Ubuntu 16.04
- Python 3.6
- Pytorch >1.7.0

## Structure

For both the CIFAR-10 and ImageNet dataset, the main folder should be in the following structure:

```python
CBN-Net	
  Evaluation
    Cifar10
    	ResNet20
        	eval.py
    	VGGSmall
        	eval.py
    ImageNet
    	eval.py
    
  Training
    Cifar10
    	ResNet20
        	train_kl.py
    	VGGSmall
        	train_kl.py
    ImageNet
    	train_kl.py
```

The datasets should be put into the **‘data’** folder or point to the place where the data is stored.  The log files are stored in the **‘log’** folder.

## Pre-trained Model

The pre-trained model file on the ImageNet dataset can be downloaded from [https://pan.baidu.com/s/1AhgGIBNFc4R-E15KeA-98w](https://pan.baidu.com/s/1AhgGIBNFc4R-E15KeA-98w) (extracted code: nips), and one should put it in the **‘save’** folder for easy evaluation.

## Training

Run the training script to train the CBN-Net with different backbones on both datasets:

```python
python train_kl.py
```

## Evaluation

Run the evaluating script to evaluate the trained models with different backbones on both datasets:

```python
python eval.py
```




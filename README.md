## Introduction

* This repository contains a UNet implementation in tensorflow to segment out features from Satellite Images.
* The data was provided by DSTL as a part of its challenge in Kaggle.

## Dependencies

* This code requires Python 2.7/3.5 , tensorflow (1.0+), OpenCV 3.0+
* This code has been tested on Ubuntu 16.04 LTS, macOS 10.11, Windows 10

## Pretrained Checkpoint:

* I have trained the UNet for the specific case of buildings.
* You can download the checkpoint at: [Checkpoint](https://drive.google.com/drive/folders/1T02s5ABQDATvnqdJUBOpO5PU8qsY5dxX?usp=sharing)
* You can train the model for any class you require by changing the class number from 1 to whatever number you desire.

#### For more information on the dataset click [Here](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)

# Content-based image retrieva with bag of words and vocabulary tree
## Introduction

This project is a simple system of image retrieva. 
The key points of the system is: 
* SIFT feature 
* bag of words 
* vocabulary tree 
 
The vocabulary tree is essentially a tree structure. You can apply a hierarchical k-means on the samples with it. 
It will make your CBIR be faster than apply a common k-means in training and testing. 
More details in the paper as follows.

## How to use 
  
### 0
First you should put your dataset in the `./data`, I've used the `The Oxford Buildings Dataset`.
 
### 1 

`build_feature_base.py` is for build vocabulary tree with image features. You can use the command to start:  

* `python build_feature_base.py --depth depth_param --branch branch_param --train_set_rate rate` 

depth_param is to set depth of tree, branch_param is to set branch of node, and rate is to set the percent of usage of your data set. 
 
### 2
`performance_testing.py` is for test the trained vocabulary tree. Before it, **you must finish running `build_feature_base.py`**  
You can use the command to start:  

* `python performance_testing.py --depth depth_param --branch branch_param --train_set_rate rate` 

These params is for locate the vocabulary tree file. 
 
### 3
`search_picture.py` is for retrieav similar picture in the image data base with given image file. 
You can use the command to start: 

* `python search_picture.py --image_file image_file_path --depth depth_param --branch branch_param --train_set_rate rate` 


## Important 

The system is proposed by the paper: Scalable Recognition with a Vocabulary Tree, publiced on cvpr 2006. 

## Notes

* The SIFT feature funtion is only in the opencv-contrib. 
So you should use the command to install `pip install opencv-contrib-python==3.4.2.16` 
* please make sure the depth and branch is proper for your samples, maybe your samples are too few to generate the leaf nodes. 

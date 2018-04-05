# DeepSight

### Introduction 

DeepSight is an application which helps to recognize text on images in real time. 
This academic project utilizes latest trends and discoveries in Deep Learning and Computer Vision. 
DeepSight uses two Convolutional Neural Networks as two main components of the processing pipeline. 
Former is a cheap but precise text agnostic detector. Later is a strong word classificator. 

### Abstract

The whole pipeline is implemented as a combination of two convolutional neural networks, the first of which is
an agnostic text detector and the other is a strong word classificator. 
The learning process of the second network was carried out in an incremental manner on 
a data set containing more than 9 million synthetically generated images. 
Architectures of both neural networks are selected in terms of short inference time, so that processing time can be reduced to a minimum. 
The resulting model, working on the NVIDIA GeForce GTX 1080 graphics card, performs detection and 
classification across 89k-word dictionary in 0.0434 seconds at average.

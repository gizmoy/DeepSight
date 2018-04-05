import os
import sys
import cv2
import json
import time
import skimage.transform as skim
import scipy
import numpy as np
import tensorflow as tf

from darkflow.net.build import TFNet
from darkflow.defaults import argHandler  #Import the default arguments
from resnet.resnet2 import ResNet18
import resnet.imagenet_input as data_input



class Model:
    def __init__(self, args=[]):
        # handle arguments
        self.FLAGS = argHandler()
        self.FLAGS.setDefaults()
        self.FLAGS.parseArgs(args)

        # Load file with names and indexes, create dict
        with open('./resnet/build/shuffled.txt','r') as f:
            lines = f.read().splitlines()
            words = [line.split() for line in lines]
            self.name_dict = {int(word[0]): word[2] for word in words}

        # fix FLAGS.load to appropriate type
        try: self.FLAGS.load = int(self.FLAGS.load)
        except: pass

        # create pipeline and run prediction
        self.yolo = TFNet(self.FLAGS)
        self.resnet = ResNet18()
        

    def predict(self, imgs):
        # Preprocess an image
        input = [self.yolo.framework.preprocess(img) for img in imgs]
        input = np.array(input)

        # Feed to the detection net
        feed_dict = {self.yolo.inp : input}    
        out = self.yolo.sess.run(self.yolo.out, feed_dict)

        results = []
        tuples = []
        for j, img in enumerate(imgs):
            results.append([])

            # Find boxes
            boxes = self.yolo.framework.findboxes(out[j])   
        
            # Prepare an image
            h, w, _ = img.shape
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = np.reshape(img_gray, [h, w, 1])
    
		    # Get cropped part of images according to bounding boxes
            threshold = self.yolo.meta['thresh']
            for i, b in enumerate(boxes):
                box_results = self.yolo.framework.process_box(b, h, w, threshold)
                if box_results is not None:
                    left, right, top, bot, mess, max_indx, confidence = box_results
                    cropped = img_gray[top:bot, left:right, :]
                    resized = skim.resize(cropped, [64, 224, 1], mode='reflect')
                    tuples.append((resized, box_results, j))

        for chunk in chunkitize(tuples, 100):          
            # Prepare input
            crops = [x[0] for x in chunk]
            imgs = np.array(crops)
            imgs = np.expand_dims(imgs, axis=0)

            # Feed to the classification net
            feed_dict = {
                self.resnet.input: imgs, 
                self.resnet.network.is_train: False
            }

            # Save time
            start = time.time()

            preds = self.resnet.sess.run([self.resnet.network.preds], feed_dict=feed_dict) 

            # Print total time
            stop = time.time() - start
            print('Total total processing time : {0: .3f}s'.format(stop))    

            # Make results from prediction
            for i, tuple in enumerate(chunk):
                _, br, img_index = tuple
                left, right, top, bot, mess, max_indx, confidence = br
                id = preds[0][i]
                label = self.name_dict[id]
                results[img_index].append({'id': int(id), 'label': label.upper(), 'confidence': float('%.2f' % confidence), 
                    'bbox': {'x': left, 'y': top, 'w': right - left, 'h': bot - top}})
      
        return results


def chunkitize(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
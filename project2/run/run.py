import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import fire
import models

import matplotlib.image as mpimg
from PIL import Image
from mask_to_submission import *

import models


class Predict(object):

    def __init__(self,  logdir='./logs'):

        self.__logdir = logdir
        
    def img_float_to_uint8(img):
        rimg = img - np.min(img)
        rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
        return rimg

    def predict(self, checkpoint='./best_model/model-12',test_data_filename = "../test_set_images",pred_size=50):
    
        def img_float_to_uint8(img):
            PIXEL_DEPTH = 255

            rimg = img - np.min(img)
            rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
            return rimg
        
        def get_prediction_without_concat(filename, image_idx):
            IMAGE_SIZE=224
            PRED_SIZE_IMG=608
            
            imageid = "/test_{:1d}/test_{:1d}".format(image_idx,image_idx)
            image_filename = filename + imageid + ".png"
            img = mpimg.imread(image_filename)

            img_input = misc.imresize(img, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
            pred = anotation_prediction.eval(feed_dict={model_in: [img_input]})[0]
            pred_resize = misc.imresize(pred, [PRED_SIZE_IMG, PRED_SIZE_IMG]) / 255.0
            img_8 = img_float_to_uint8(pred_resize)

            return img_8
    
        segmentation_model = models.SegnetConnectedGate(training_mode = False, num_classes=2) # we use the segnet connected version 

        # Get Placeholders
        model_in = segmentation_model.input
        model_out = segmentation_model.output
        anotation_prediction = segmentation_model.anotation_prediction

        # Load the pretrained tensorflow model
        print("Loading model: %s" % checkpoint)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        # Use the pretrained Segnet Gate architecture to make the prediction on the set
        print ("Running prediction on training set")
        prediction_training_dir = "predictions_segnet_gate/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, pred_size+1):
            pimg = get_prediction_without_concat(test_data_filename, i)
            Image.fromarray(pimg).save(prediction_training_dir + "prediction_" + str(i) + ".png") 
        
        # Generate the .csv file
        submission_filename = 'submission.csv'
        image_filenames = []
        for i in range(1, pred_size+1):
            image_filename = prediction_training_dir+'prediction_' + str(i) + '.png'
            print(image_filename)
            image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)#use the provided code to generate the csv file
        
        print(submission_filename + " generated")


if __name__ == '__main__':
  fire.Fire(Predict)
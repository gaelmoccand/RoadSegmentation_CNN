import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os
import fire
import models
from handle_data import HandleData
import matplotlib.image as mpimg
from PIL import Image
from mask_to_submission import *
import models


class Run(object):

    def __init__(self,  logdir='./logs',lmdb='../training/lmdb', input_val='',mem_frac=0.8):

        self.__logdir = logdir
        self.__memfrac = mem_frac
        self.__input = lmdb
        self.__input_val = input_val
        
    # Used to predict the lmdb file from an image data set
    # cmd: 
    def create_lmdb(self,train_set = "../test_set_images"):  
        
   

    # Used to predict the roads on  data sata images
    # cmd: python run.py predict --pred_size=3 --test_set='../test_set_images'  --trained_model='./best_model/model-12' 
    def predict(self, trained_model='./best_model/model-12',test_set = "../test_set_images",pred_size=50):
    
        def img_float_to_uint8(img):
            PIXEL_DEPTH = 255

            rimg = img - np.min(img)
            rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
            return rimg
        
        def get_prediction_without_concat(filename, image_idx):
            IMAGE_SIZE=224
            PRED_SIZE_IMG=608 # the test images size are different do not why
            
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
        print("Loading model: %s" % trained_model)
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, trained_model)

        # Use the pretrained Segnet Gate architecture to make the prediction on the set
        print ("Running prediction on training set")
        prediction_training_dir = "predictions_segnet_gate/"
        if not os.path.isdir(prediction_training_dir):
            os.mkdir(prediction_training_dir)
        for i in range(1, pred_size+1):
            pimg = get_prediction_without_concat(test_set, i)
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

    
    # train the SegNet model
    # cmd: python run.py train --lmdb='lmdb' --epochs=5 --batch_size=10
    #code from https://github.com/leonardoaraujosantos/LearnSegmentation
    def train(self , epochs=4, learning_rate_init=0.001, checkpoint='', batch_size=50, l2_reg=0.0001, nclass=2, do_resize=False):

        # Avoid allocating the whole memory
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.__memfrac)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # Regularization value
        L2NormConst = l2_reg
        mode='segnet_connected_gate'
        print('Train segmentation model:', mode)

        # Build model
        segmentation_model = models.SegnetConnectedGate(num_classes=nclass)

        # Get Placeholders
        model_in = segmentation_model.input
        model_out = segmentation_model.output
        labels_in = segmentation_model.label_in
        anotation_prediction = segmentation_model.anotation_prediction

        # Add input image on summary
        tf.summary.image("input_image", model_in, 2)
        tf.summary.image("ground_truth", tf.cast(labels_in, tf.uint8), max_outputs=2)
        # Expand dimension before asking a sumary
        tf.summary.image("pred_annotation", tf.cast(tf.expand_dims(anotation_prediction, dim=3), tf.uint8), max_outputs=2)

        # Get all model "parameters" that are trainable
        train_vars = tf.trainable_variables()

        # Add loss
        # Segmentation problems often uses this "spatial" softmax (Basically we want to classify each pixel)
        with tf.name_scope("SPATIAL_SOFTMAX"):
            loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out,labels=tf.squeeze(labels_in, squeeze_dims=[3]),name="spatial_softmax"))) + tf.add_n(
            [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst

        # Add model accuracy
        with tf.name_scope("Loss_Validation"):
            loss_val = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=model_out, labels=tf.squeeze(labels_in, squeeze_dims=[3]), name="spatial_softmax")))

        # Solver configuration
        # Get ops to update moving_mean and moving_variance from batch_norm
        # Reference: https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope("Solver"):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = learning_rate_init
            # decay every 10000 steps with a base of 0.96
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       30000, 0.1, staircase=True)

            # Basically update the batch_norm moving averages before the training step
            # http://ruishu.io/2016/12/27/batchnorm/
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Initialize all random variables (Weights/Bias)
        sess.run(tf.global_variables_initializer())

        # Load checkpoint if needed
        if checkpoint != '':
            # Load tensorflow model
            print("Loading pre-trained model: %s" % checkpoint)
            # Create saver object to save/load training checkpoint
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, checkpoint)
        else:
            # Just create saver for saving checkpoints
            saver = tf.train.Saver(max_to_keep=None)

        # Monitor loss, learning_rate, global_step, etc...
        tf.summary.scalar("loss_train", loss)
        tf.summary.scalar("loss_val", loss_val)
        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("global_step", global_step)
        # merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Configure where to save the logs for tensorboard
        summary_writer = tf.summary.FileWriter(self.__logdir, graph=tf.get_default_graph())

        data = HandleData(path=self.__input, path_val=self.__input_val)
        num_images_epoch = int(data.get_num_images() / batch_size)
        print('Num samples', data.get_num_images(), 'Iterations per epoch:', num_images_epoch, 'batch size:',batch_size, 'nb of class',nclass)

        # For each epoch
        for epoch in range(epochs):
            for i in range(int(data.get_num_images() / batch_size)):
                # Get training batch
                xs_train, ys_train = data.LoadTrainBatch(batch_size, should_augment=False, do_resize=do_resize)

                # Send training batch to tensorflow graph (Dropout enabled)
                train_step.run(feed_dict={model_in: xs_train, labels_in: ys_train})

                # Display some information each x iterations
                if i % 100 == 0:
                    # Get validation batch
                    xs, ys = data.LoadValBatch(batch_size, do_resize=False,)
                    # Send validation batch to tensorflow graph (Dropout disabled)
                    loss_value = loss_val.eval(feed_dict={model_in: xs, labels_in: ys})
                    print("Epoch: %d, Step: %d, Loss(Val): %g" % (epoch, epoch * batch_size + i, loss_value))


                # write logs at every iteration
                summary = merged_summary_op.eval(feed_dict={model_in: xs_train, labels_in: ys_train})
                summary_writer.add_summary(summary, epoch * batch_size + i)

            # Save checkpoint after each epoch
            if not os.path.exists(self.__savedir):
                os.makedirs(self.__savedir)
            checkpoint_path = os.path.join(self.__savedir, "model")
            filename = saver.save(sess, checkpoint_path, global_step=epoch)
            print("Model saved in file: %s" % filename)

            # Shuffle data at each epoch end
            print("Shuffle data")
            data.shuffleData()
            

if __name__ == '__main__':
  fire.Fire(Run)
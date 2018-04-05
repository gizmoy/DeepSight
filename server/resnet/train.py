#!/usr/bin/env python

import os
import multiprocessing
import time
import sys
import select

import tensorflow as tf
import numpy as np

from datetime import datetime
from IPython import embed
from tensorflow.python.client import timeline

import imagenet_input as data_input
import resnet



# Dataset Configuration
tf.app.flags.DEFINE_string('train_dataset', './chunks_1000s/train_____________88.txt', """Path to the train dataset list file""")
tf.app.flags.DEFINE_string('train_image_root', 'C:/Users/Mike/Documents/mnt/ramdisk/max/90kDICT32px', """Path to the root of Synth train images""")
tf.app.flags.DEFINE_string('val_dataset', './chunks_1000/val_88.txt', """Path to the val dataset list file""")
tf.app.flags.DEFINE_string('val_image_root', 'C:/Users/Mike/Documents/mnt/ramdisk/max/90kDICT32px', """Path to the root of Synth val images""")
tf.app.flags.DEFINE_string('test_dataset', './chunks_1000/test_88.txt', """Path to the test dataset list file""")
tf.app.flags.DEFINE_string('test_image_root', 'C:/Users/Mike/Documents/mnt/ramdisk/max/90kDICT32px', """Path to the root of Synth test images""")
tf.app.flags.DEFINE_string('mean_path', './ResNet_mean_rgb.pkl', """Path to the imagenet mean""")
#tf.app.flags.DEFINE_integer('num_classes', 88172, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_classes', 88712, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('num_train_instance', 7224613, """Number of train images.""")
tf.app.flags.DEFINE_integer('num_val_instance', 802735, """Number of val images.""")
tf.app.flags.DEFINE_integer('num_test_instance', 891928, """Number of test images.""")

# Network Configuration
tf.app.flags.DEFINE_integer('batch_size', 80, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_gpus', 1, """Number of GPUs.""")

# Optimization Configuration
tf.app.flags.DEFINE_float('l2_weight', 0.0001, """L2 loss weight applied all the weights""")
tf.app.flags.DEFINE_float('momentum', 0.9, """The momentum of MomentumOptimizer""")
tf.app.flags.DEFINE_float('initial_lr', 0.002, """Initial learning rate""")
tf.app.flags.DEFINE_float('lr_decay', 0.5, """Learning rate decay factor""")
tf.app.flags.DEFINE_boolean('finetune', False, """Whether to finetune.""")

# Training Configuration
tf.app.flags.DEFINE_string('train_dir', './train', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('display', 100, """Number of iterations to display training info.""")
tf.app.flags.DEFINE_integer('val_interval', 2000, """Number of iterations to run a val""")
tf.app.flags.DEFINE_integer('val_iter', 10035, """Number of iterations during a val""")
tf.app.flags.DEFINE_integer('checkpoint_interval', 2000, """Number of iterations to save parameters as a checkpoint""")
tf.app.flags.DEFINE_float('gpu_fraction', 0.99, """The fraction of GPU memory to be allocated""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', './train/model.ckpt-462000', """Model checkpoint to load""") # ./train/0_1000/model.ckpt-25000
tf.app.flags.DEFINE_integer('plateau_interval', 2000, """Number of iteration to calculate moving avarage""")
tf.app.flags.DEFINE_float('plateau_min_improvement', 1.08, """Minimum improvement that accuracy should has between two plateau intervals""")
tf.app.flags.DEFINE_boolean('incremental_training', False, """Whether to perform incremental training partial initialization phase""")
tf.app.flags.DEFINE_integer('chunk_size', 22712, """Number of classes in one chunk""")

FLAGS = tf.app.flags.FLAGS


def train():
    print('[Dataset Configuration]')
    print('\tSynthText training root: %s' % FLAGS.train_image_root)
    print('\tSynthText training list: %s' % FLAGS.train_dataset)
    print('\tSynthText val root: %s' % FLAGS.val_image_root)
    print('\tSynthText val list: %s' % FLAGS.val_dataset)
    print('\tNumber of classes: %d' % FLAGS.num_classes)
    print('\tNumber of training images: %d' % FLAGS.num_train_instance)
    print('\tNumber of val images: %d' % FLAGS.num_val_instance)

    print('[Network Configuration]')
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tNumber of GPUs: %d' % FLAGS.num_gpus)
    print('\tBasemodel file: %s' % FLAGS.basemodel)

    print('[Optimization Configuration]')
    print('\tL2 loss weight: %f' % FLAGS.l2_weight)
    print('\tThe momentum optimizer: %f' % FLAGS.momentum)
    print('\tInitial learning rate: %f' % FLAGS.initial_lr)
    print('\tLearning rate decay: %f' % FLAGS.lr_decay)

    print('[Training Configuration]')
    print('\tTrain dir: %s' % FLAGS.train_dir)
    print('\tTraining max steps: %d' % FLAGS.max_steps)
    print('\tSteps per displaying info: %d' % FLAGS.display)
    print('\tSteps per validation: %d' % FLAGS.val_interval)
    print('\tSteps during validation: %d' % FLAGS.val_iter)
    print('\tSteps per saving checkpoints: %d' % FLAGS.checkpoint_interval)
    print('\tGPU memory fraction: %f' % FLAGS.gpu_fraction)
    print('\tLog device placement: %d' % FLAGS.log_device_placement)
    print('\tPlateau step interval: %d' % FLAGS.plateau_interval)
    print('\tPlateau minimimum improvement between intervals: %f' % FLAGS.plateau_min_improvement)
    print('\tIncremental training: %d' % FLAGS.incremental_training)
    print('\tChunk size: %d' % FLAGS.chunk_size)


    with tf.Graph().as_default():
        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Get images and labels of SynthText
        num_threads = int(multiprocessing.cpu_count() / FLAGS.num_gpus)
        print('Load SynthText dataset(%d threads)' % num_threads)
        with tf.device('/cpu:0'):
            print('\tLoading training data from %s' % FLAGS.train_dataset)
            with tf.variable_scope('train_batch'):
                train_images, train_labels = data_input.distorted_inputs(FLAGS.train_image_root, FLAGS.train_dataset
                                               , FLAGS.batch_size, shuffle=True, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            print('\tLoading validation data from %s' % FLAGS.val_dataset)
            with tf.variable_scope('val_batch'):
                val_images, val_labels = data_input.inputs(FLAGS.val_image_root, FLAGS.val_dataset
                                               , FLAGS.batch_size, shuffle=False, num_threads=num_threads, num_sets=FLAGS.num_gpus)
            print('\tLoading test data from %s' % FLAGS.test_dataset)
            with tf.variable_scope('test_batch'):
                test_images, test_labels = data_input.inputs(FLAGS.test_image_root, FLAGS.test_dataset
                                               , FLAGS.batch_size, shuffle=False, num_threads=num_threads, num_sets=FLAGS.num_gpus)
        tf.summary.image('images', train_images[0])

        # Build model
        hp = resnet.HParams(batch_size=FLAGS.batch_size,
                            num_gpus=FLAGS.num_gpus,
                            num_classes=FLAGS.num_classes,
                            weight_decay=FLAGS.l2_weight,
                            momentum=FLAGS.momentum,
                            finetune=FLAGS.finetune)
        network_train = resnet.ResNet(hp, train_images, train_labels, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        network_val = resnet.ResNet(hp, val_images, val_labels, global_step, name="val", reuse_weights=True)
        network_val.build_model()
        network_test = resnet.ResNet(hp, test_images, test_labels, global_step, name="test", reuse_weights=True)
        network_test.build_model()
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=FLAGS.gpu_fraction,
                allow_growth=True),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if FLAGS.checkpoint is not None:
            print('Load checkpoint %s' % FLAGS.checkpoint)
            saver.restore(sess, FLAGS.checkpoint)
            init_step = global_step.eval(session=sess)

            # Expand fc weights and biases by chunk size and initialize new subtensor to 0
            if FLAGS.incremental_training:
                # Get weights and biases as well as theirs momentums
                print('Incremental training - expanding fc layer by %d' % FLAGS.chunk_size)
                vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                w  = next(v for v in vars if v.name =='logits/fc/weights:0')
                wm = next(v for v in vars if v.name =='logits/fc/weights/Momentum:0')
                b  = next(v for v in vars if v.name =='logits/fc/biases:0') 
                bm = next(v for v in vars if v.name =='logits/fc/biases/Momentum:0')

                # Expand with zeros       
                stddev = np.sqrt(1.0/(FLAGS.num_classes + FLAGS.chunk_size))
                w_new = tf.concat([w, tf.random_normal([512, FLAGS.chunk_size], stddev=stddev)], 1)
                wm_new = tf.concat([wm, tf.random_normal([512, FLAGS.chunk_size], stddev=stddev)], 1)
                b_new = tf.concat([b, tf.random_normal([FLAGS.chunk_size], stddev=stddev)], 0)
                bm_new = tf.concat([bm, tf.random_normal([FLAGS.chunk_size], stddev=stddev)], 0)

                # Run assign ops
                sess.run([tf.assign(w, w_new, validate_shape=False), 
                        tf.assign(wm, wm_new, validate_shape=False),
                        tf.assign(b, b_new, validate_shape=False),
                        tf.assign(bm, bm_new, validate_shape=False)])

        elif FLAGS.basemodel:
            # Define a different saver to save model checkpoints
            print('Load parameters from basemodel %s' % FLAGS.basemodel)
            variables = tf.global_variables()
            vars_restore = [var for var in variables
                            if not "Momentum" in var.name and
                               not "global_step" in var.name]
            saver_restore = tf.train.Saver(vars_restore, max_to_keep=10000)
            saver_restore.restore(sess, FLAGS.basemodel)
        else:
            print('No checkpoint file of basemodel found. Start from the scratch.')

        # Start queue runners & summary_writer
        tf.train.start_queue_runners(sess=sess)

        if not os.path.exists(FLAGS.train_dir):
            os.mkdir(FLAGS.train_dir)
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.train_dir, str(global_step.eval(session=sess))),
                                                sess.graph)

        # Training
        val_best_acc = 0.0
        acc_sum = 0.0
        loss_sum = 0.0
        prev_loss_mean = float("inf")
        prev_val_acc = 0.9606
        lr = FLAGS.initial_lr
        
        # Training loop
        for step in range(init_step, FLAGS.max_steps):
            # Train
            start_time = time.time()
            _, loss_value, acc_value, train_summary_str = \
                    sess.run([network_train.train_op, network_train.loss, network_train.acc, train_summary_op],
                            feed_dict={network_train.is_train:True, network_train.lr:lr})
            duration = time.time() - start_time
            loss_sum += loss_value
            acc_sum += acc_value
            assert not np.isnan(loss_value)

            # Save the model checkpoint periodically.
            if (step > init_step and step % FLAGS.checkpoint_interval == 0) or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            # Display & Summary(training)
            if step % FLAGS.display == 0 or step < init_step + 30:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ('%s: (train) step %d, loss=%.4f, acc=%.4f, lr=%f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (format_str % (datetime.now(), step, loss_value, acc_value, lr, 
                                     examples_per_sec, sec_per_batch))
                summary_writer.add_summary(train_summary_str, step)

            # Val
            if step % FLAGS.val_interval == 0:
                val_loss, val_acc = 0.0, 0.0
                for i in range(FLAGS.val_iter):
                    loss_value, acc_value = sess.run([network_val.loss, network_val.acc],
                                feed_dict={network_val.is_train:False})
                    val_loss += loss_value
                    val_acc += acc_value
                val_loss /= FLAGS.val_iter
                val_acc /= FLAGS.val_iter
                val_best_acc = max(val_best_acc, val_acc)
                format_str = ('%s: (val)   step %d, loss=%.4f, acc=%.4f')
                print (format_str % (datetime.now(), step, val_loss, val_acc))

                val_summary = tf.Summary()
                val_summary.value.add(tag='val/loss', simple_value=val_loss)
                val_summary.value.add(tag='val/acc', simple_value=val_acc)
                val_summary.value.add(tag='val/best_acc', simple_value=val_best_acc)
                summary_writer.add_summary(val_summary, step)
                summary_writer.flush()

            # Check whether training loss is on plateau
            if step % FLAGS.plateau_interval == 0 and step > init_step:
                loss_mean, acc_mean = loss_sum/FLAGS.plateau_interval, acc_sum/FLAGS.plateau_interval
                loss_sum, acc_sum = 0.0, 0.0      
                improvement = prev_loss_mean / loss_mean
                if improvement < FLAGS.plateau_min_improvement and prev_val_acc > val_acc:
                    lr *= FLAGS.lr_decay      
                format_str = ('%s: (train) [!] step %d, loss_mean=%.4f, prev_loss_mean=%.4f, acc_mean=%.4f, improvement=%.4f')
                print (format_str % (datetime.now(), step, loss_mean, prev_loss_mean, acc_mean, improvement))
                prev_loss_mean = loss_mean
                prev_val_acc = val_acc;

        # Test
        test_iter = int(FLAGS.num_test_instance / FLAGS.batch_size) + 1
        test_loss, test_acc = 0.0, 0.0
        for i in range(test_iter):
            loss_value, acc_value = sess.run([network_test.loss, network_test.acc],
                        feed_dict={network_test.is_train:False})
            test_loss += loss_value
            test_acc += acc_value
        test_loss /= test_iter
        test_acc /= test_iter
        format_str = ('%s: (test)     loss=%.4f, acc=%.4f')
        print (format_str % (datetime.now(), test_loss, test_acc))

        test_summary = tf.Summary()
        test_summary.value.add(tag='test/loss', simple_value=test_loss)
        test_summary.value.add(tag='test/acc', simple_value=test_acc)
        summary_writer.add_summary(test_summary, 0)
        summary_writer.flush()


def main(argv=None):
  train()


if __name__ == '__main__':
  tf.app.run()

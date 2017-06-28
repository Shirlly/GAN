# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 18:00:29 2017

@author: Zheng Xin
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


LOG_DIR = 'logging'
if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

#mnist = input_data.read_data_sets('MNIST_data')
embed = np.load('./genData/auto_embed.npy')
label = np.load('./genData/auto_label.npy')
images = tf.Variable(embed, name='images')
# images = tf.Variable(mnist.test.images, name='images')

with open(metadata, 'w') as metadata_file:
    #import pdb; pdb.set_trace()
    for row in label:
    #for row in mnist.test.labels:
        # import pdb; pdb.set_trace()
        metadata_file.write('%d\n' % row)
    # metadata_file.close()

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
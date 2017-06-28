# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:47:37 2017

@author: Zheng Xin
"""

from __future__ import division, print_function, absolute_import

from tensorgraph.dataset import Mnist
import tensorflow as tf
import tensorgraph as tg
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10
batchsize = 64
max_epoch = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)


# Load data
X_train, y_train, X_valid, y_valid = Mnist()
n, h, w, c = X_train.shape  
dim = int(h * w * c)
X_train = np.reshape(X_train, [n, dim])   

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, dim])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):     
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)

# Initializing the variables
init = tf.global_variables_initializer()
# Launch the graph

with tf.Session() as sess:
    sess.run(init)
    # total_batch = int(mnist.train.num_examples/batch_size)
    # Training 
    output = np.empty([0, n_hidden_2], 'float')
    labels = np.empty([0, 10], 'int')
    for epoch in range(max_epoch):
        # import pdb; pdb.set_trace()
        for X_batch, y_batch in data_train:            
            # Run optimization op (backprop) and cost op (to get loss value)
            _, co = sess.run([optimizer, cost], feed_dict={X: X_batch})
            # Display logs per epoch step
            if epoch == max_epoch-1:
                embed = sess.run(encoder_op, feed_dict={X: X_batch})
                output = np.concatenate((output, embed), axis = 0)
                labels = np.concatenate((labels, y_batch), axis = 0)
            
        print("Optimization Finished!")
        
        
    embeddir = './genData/'
    lab = np.nonzero(labels)[1]
    np.save(embeddir + 'auto_embed.npy', output)
    np.save(embeddir + 'auto_label.npy', lab)        

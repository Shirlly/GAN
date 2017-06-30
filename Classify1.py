# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:28:21 2017

@author: Zheng Xin
"""
import os
from sklearn import svm
from sklearn import metrics
import tensorgraph as tg
import tensorflow as tf
import numpy as np
from tensorgraph.layers.activation import Softmax
from tensorgraph.layers.conv import Conv2D, MaxPooling, AvgPooling
from utils import total_mse, total_accuracy
from tensorgraph.graph import Graph
from tensorgraph.layers.linear import Linear
from layer import Reshape
from math import ceil
from tensorgraph.dataset import Mnist, Cifar10
from tensorgraph.layers import RELU, Sigmoid, Flatten, Tanh, TFBatchNormalization, Dropout


def SVM_Classifier(X_train, y_train, X_valid, y_valid, restore):
    clf = svm.SVC(gamma = 0.001, C = 100)
    
    # fprint '===== Training without data augmentation ====='
    X_num, _, _, _ = X_train.shape
    X_train = np.reshape(X_train, (X_num, -1))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print 'Precision: \t', metrics.precision_score(y_valid, y_pred, average='micro')
    
    # clf_au = svm.SVM(gamma = 0.001, C = 100)
    # print '===== Training without data augmentation ====='
    # AuX_num, _, _, _ = AuX_train.shape
    # AuX_train = np.reshape(AuX_train, (AuX_train, -1))
    # clf_au.fit(AuX_train, Auy_train)
    # y_pred_au = clf_au.predict(X_valid)
    # print 'Precision: \t', metrics.precision_score(y_valid, y_pred_au, average='micro')
    
 

def same(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height) / float(strides[0]))
    out_width  = ceil(float(in_width) / float(strides[1]))
    return out_height, out_width

def valid(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height - filters[0] + 1) / float(strides[0]))
    out_width  = ceil(float(in_width - filters[1] + 1) / float(strides[1]))
    return out_height, out_width
    
def Vanilla_Classifier(X_train, y_train, X_valid, y_valid, restore):
    batchsize = 100
    learning_rate = 0.001
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape
    
    g = tf.Graph()
    with g.as_default():
    
        data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
        data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)

        X_ph = tf.placeholder('float32', [None, h, w, c])
        # y_ph = tf.placeholder('float32', [None, nclass])
        y_phs = []
        for comp in [nclass]:
            y_phs.append(tf.placeholder('float32', [None, comp]))
    
        dim = int(h*w*c)
        scope = 'encoder'
        start = tg.StartNode(input_vars=[X_ph])
        h1_Node = tg.HiddenNode(prev=[start], 
                                layers=[Sigmoid(),
                                        TFBatchNormalization(name= scope + '/vanilla1'),
                                        RELU(),
                                        Flatten(),
                                        Sigmoid(),
                                        TFBatchNormalization(name=scope + '/vanilla2')])
                                    
        h2_Node = tg.HiddenNode(prev=[h1_Node],
                                layers=[Linear(prev_dim=dim, this_dim=nclass),
                                        Softmax()])                                
        end_nodes = [tg.EndNode(prev=[h2_Node])]
    
        graph = Graph(start=[start], end=end_nodes)

        train_outs_sb = graph.train_fprop()
        test_outs = graph.test_fprop()
    
        ttl_mse = []
        # import pdb; pdb.set_trace()
        for y_ph, out in zip(y_phs, train_outs_sb):
            #ttl_mse.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_ph, out)))
            ttl_mse.append(tf.reduce_mean((y_ph-out)**2))


        mse = sum(ttl_mse)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)
    
        saver = tf.train.Saver()
        vardir = './var/3'
        if not os.path.exists(vardir):
            os.makedirs(vardir)

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        tf.set_random_seed(1)
        init = tf.global_variables_initializer()
    
    
        with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
            # print '=======session start'
            sess.run(init)
            if restore == 1:
                re_saver = tf.train.Saver()
                re_saver.restore(sess, vardir + "/model.ckpt")
                print("Model restored.")
            max_epoch = 100
            temp_acc = []
            
            for epoch in range(max_epoch):

                train_error = 0
                train_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_train:
                    feed_dict = {X_ph:X_batch}
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch
                
                    sess.run(optimizer, feed_dict=feed_dict)
                    train_outs = sess.run(train_outs_sb, feed_dict=feed_dict)
                    train_error += total_mse(train_outs, [ys])[0]
                    train_accuracy += total_accuracy(train_outs, [ys])[0]
                    ttl_examples += len(X_batch)               

                valid_error = 0
                valid_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_valid:
                    feed_dict = {X_ph:X_batch}  
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch

                    valid_outs = sess.run(test_outs, feed_dict=feed_dict)
                    valid_error += total_mse(valid_outs, [ys])[0]
                    valid_accuracy += total_accuracy(valid_outs, [ys])[0]
                    ttl_examples += len(X_batch)

                save_path = saver.save(sess, vardir + "/model.ckpt")
                # print("Model saved in file: %s" % save_path)
                temp_acc.append(valid_accuracy/float(ttl_examples))
            print 'max accuracy is:\t', max(temp_acc)
    
    

def CNN_Classifier(X_train, y_train, X_valid, y_valid, restore):
    batchsize = 64
    learning_rate = 0.001
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape
    
    g = tf.Graph()
    with g.as_default():
        data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
        data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)

        X_ph = tf.placeholder('float32', [None, h, w, c])
    
        y_phs = []
        for comp in [nclass]:
            y_phs.append(tf.placeholder('float32', [None, comp]))
    
    
        start = tg.StartNode(input_vars=[X_ph])
    
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(1,1), filters=(2,2))
        h, w = valid(in_height=h, in_width=w, strides=(2,2), filters=(2,2))
        # import pdb; pdb.set_trace()
        #h1, w1 = valid(ch_embed_dim, word_len, strides=(1,1), filters=(ch_embed_dim,4))
        num = 32
        dim = int(h * w * num )
        
        h1_Node = tg.HiddenNode(prev=[start], 
                                layers=[Conv2D(input_channels=c, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer1'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer2'),
                                        RELU(),  
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer3'),
                                        RELU(), 
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer4'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer5'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer6'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer7'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer8'),
                                        RELU(),
                                        Dropout(dropout_below=0.5),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer9'),
                                        RELU(),
                                        Conv2D(input_channels=num, num_filters=num, padding='VALID', kernel_size=(2,2), stride=(1,1)),
                                        TFBatchNormalization(name='layer10'),
                                        RELU(),
                                        MaxPooling(poolsize=(2,2), stride=(2,2), padding='VALID'),
                                        Reshape(shape=(-1, dim))] )
                                       
        h2_Node = tg.HiddenNode(prev=[h1_Node],
                               layers=[Linear(prev_dim=dim, this_dim=nclass),
                                       Softmax()])
                                    
        end_nodes = [tg.EndNode(prev=[h2_Node])]
    
        graph = Graph(start=[start], end=end_nodes)

        train_outs_sb = graph.train_fprop()
        test_outs = graph.test_fprop()
    
        ttl_mse = []
        # import pdb; pdb.set_trace()
        for y_ph, out in zip(y_phs, train_outs_sb):
            #ttl_mse.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_ph, out)))
            ttl_mse.append(tf.reduce_mean((y_ph-out)**2))


        mse = sum(ttl_mse)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        saver = tf.train.Saver()
        vardir = './var/5'
        if not os.path.exists(vardir):
            os.makedirs(vardir)
        
        tf.set_random_seed(1)
        init = tf.global_variables_initializer()
        with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        
            sess.run(init)
            if restore == 1:
                re_saver = tf.train.Saver()
                re_saver.restore(sess, vardir + "/model.ckpt")
                print("Model restored.")
                
                    
            max_epoch = 100
            temp_acc = []
            for epoch in range(max_epoch):
                # print 'epoch:', epoch
                train_error = 0
                train_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_train:
                    feed_dict = {X_ph:X_batch}
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch
                        # import pdb; pdb.set_trace() 
                    sess.run(optimizer, feed_dict=feed_dict)
                    train_outs = sess.run(train_outs_sb, feed_dict=feed_dict)
                    train_error += total_mse(train_outs, [ys])[0]
                    train_accuracy += total_accuracy(train_outs, [ys])[0]
                    ttl_examples += len(X_batch)
               

                valid_error = 0
                valid_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_valid:
                    feed_dict = {X_ph:X_batch}  
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch

                    valid_outs = sess.run(test_outs, feed_dict=feed_dict)
                    valid_error += total_mse(valid_outs, [ys])[0]
                    valid_accuracy += total_accuracy(valid_outs, [ys])[0]
                    ttl_examples += len(X_batch)

                temp_acc.append(valid_accuracy/float(ttl_examples))
            save_path = saver.save(sess, vardir + "/model.ckpt") 
            print("Model saved in file: %s" % save_path)
            print 'max accuracy is:\t', max(temp_acc)
        
        
        
def Encoder_Classifier(X_train, y_train, X_valid, y_valid, restore):
        
    batchsize = 64
    learning_rate = 0.001
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape
    
    g = tf.Graph()
    with g.as_default():
    
        data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
        data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)

        X_ph = tf.placeholder('float32', [None, h, w, c])
    
        y_phs = []
        for comp in [nclass]:
            y_phs.append(tf.placeholder('float32', [None, comp]))
    
    
        start = tg.StartNode(input_vars=[X_ph])
    
        h1, w1 = valid(h, w, filters=(5,5), strides=(1,1))
        h2, w2 = valid(h1, w1, filters=(5,5), strides=(2,2))
        h3, w3 = valid(h2, w2, filters=(5,5), strides=(2,2))
        flat_dim = int(h3*w3*32)
        scope = 'encoder'
        bottleneck_dim = 300
        enc_hn = tg.HiddenNode(prev=[start],
                               layers=[Conv2D(input_channels=c, num_filters=32, kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                       TFBatchNormalization(name=scope + '/genc1'),
                                       RELU(),
                                       Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                       TFBatchNormalization(name=scope + '/genc2'),
                                       RELU(),
                                       Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                       TFBatchNormalization(name=scope + '/genc3'),
                                       RELU(),
                                       Flatten(),
                                       Linear(flat_dim, 300),
                                       TFBatchNormalization(name=scope + '/genc4'),
                                       RELU(),
                                       Linear(300, bottleneck_dim),
                                       Tanh()
                                       ])
                                       
        h2_Node = tg.HiddenNode(prev=[enc_hn],
                                layers=[Linear(prev_dim=bottleneck_dim, this_dim=nclass),
                                        Softmax()])
                                    
        end_nodes = [tg.EndNode(prev=[h2_Node])]
    
        graph = Graph(start=[start], end=end_nodes)

        train_outs_sb = graph.train_fprop()
        test_outs = graph.test_fprop()
    
        ttl_mse = []
        # import pdb; pdb.set_trace()
        for y_ph, out in zip(y_phs, train_outs_sb):
            #ttl_mse.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_ph, out)))
            ttl_mse.append(tf.reduce_mean((y_ph-out)**2))


        mse = sum(ttl_mse)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    
        # saver_init = tf.train.Saver()
        saver = tf.train.Saver()
        vardir = './var/2'
        if not os.path.exists(vardir):
            os.makedirs(vardir)
       
        tf.set_random_seed(1)
        init = tf.global_variables_initializer()
            
        with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
            sess.run(init)
            if restore == 1:
                re_saver = tf.train.Saver()
                re_saver.restore(sess, vardir + "/model.ckpt")
                print("Model restored.")
            
            # save_path = saver_init.save(sess, vardir + "/init.ckpt")
            # print("Model saved in file: %s" % save_path)
            max_epoch = 100
            temp_acc = []
            for epoch in range(max_epoch):
                # print 'epoch:', epoch
                train_error = 0
                train_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_train:
                    feed_dict = {X_ph:X_batch}
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch
                        # import pdb; pdb.set_trace() 
                    sess.run(optimizer, feed_dict=feed_dict)
                    train_outs = sess.run(train_outs_sb, feed_dict=feed_dict)
                    train_error += total_mse(train_outs, [ys])[0]
                    train_accuracy += total_accuracy(train_outs, [ys])[0]
                    ttl_examples += len(X_batch)

                valid_error = 0
                valid_accuracy = 0
                ttl_examples = 0
                for X_batch, ys in data_valid:
                    feed_dict = {X_ph:X_batch}  
                    for y_ph, y_batch in zip(y_phs, [ys]):
                        feed_dict[y_ph] = y_batch

                    valid_outs = sess.run(test_outs, feed_dict=feed_dict)
                    valid_error += total_mse(valid_outs, [ys])[0]
                    valid_accuracy += total_accuracy(valid_outs, [ys])[0]
                    ttl_examples += len(X_batch)


                temp_acc.append(valid_accuracy/float(ttl_examples))
            save_path = saver.save(sess, vardir + "/model.ckpt")
            print("Model saved in file: %s" % save_path)
            print 'max accuracy is:\t', max(temp_acc)        
        
def classify(X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy):
    print '\n===== Training with original data =====\n'
    CNN_Classifier(X_train, y_train, X_valid, y_valid, restore = 0)  
    
    print '\n===== Training with augmentation data =====\n'
    CNN_Classifier(aux, auy, X_valid, y_valid, restore = 0)
    
    print '\n===== Training with data augmentation =====\n'
    CNN_Classifier(X_train, y_train, X_valid, y_valid, restore = 1) 
    
    # print '\n===== Training with all the data =====\n'
    # Encoder_Classifier(AuX_train, Auy_train, X_valid, y_valid, restore = 0)

def data():
    # X_train, y_train, X_valid, y_valid = Mnist()
    X_train, y_train, X_valid, y_valid = Cifar10(contrast_normalize=False, whiten=False)
    # import pdb; pdb.set_trace()
    print '====== shape of Cifar10 validation data ', X_valid.shape
    X_train, y_train = X_train[:30000], y_train[:30000]
    # import pdb;pdb.set_trace()
    xname = 'cifar_genx5_50.npy'
    yname = 'cifar_geny5_50.npy'
    datadir = './genData'
    aux = np.load('{}/{}'.format(datadir, xname))
    auy = np.load('{}/{}'.format(datadir, yname))
    num, _, _, _ = aux.shape
    snum = 10000
    aux = aux[:snum]
    auy = auy[:snum]
    print 'Augmented data size ', snum
    AuX_train = np.concatenate((aux, X_train), axis = 0)
    Auy_train = np.concatenate((auy, y_train), axis = 0)    
    return X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy
    
    
if __name__ == '__main__':
    X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy = data()
    classify(X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy)
    
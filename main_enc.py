# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:12:52 2017

@author: Zheng Xin
"""

import tensorgraph as tg
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative, Sigmoid, Flatten, Tanh, Linear
from tensorgraph.utils import valid, same, put_kernels_on_grid
from tensorgraph.cost import entropy
import tensorflow as tf
import numpy as np
from data import *
import os
from datetime import datetime
from tensorgraph.dataset import Mnist, Cifar100, Cifar10
from tensorflow.contrib.tensorboard.plugins import projector
import model_init as model
from utils import total_mse, total_accuracy

def clip(X):
    epsilon = 1e-15
    return tf.clip_by_value(X, epsilon, 1.0-epsilon)


def ph2onehot(ph, char_embed_dim):
    seq = tg.Sequential()
    seq.add(OneHot(char_embed_dim))
    seq.add(Transpose((0,3,2,1)))
    oh = seq.train_fprop(ph)
    return oh

def sigmoid_cross_entropy_with_logits(x, y):
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
  except:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

def discriminator_cost(y_ph, real, fake):
    real_clss, real_judge = real
    fake_clss, fake_judge = fake
    real_mse = tf.reduce_mean((y_ph - real_clss)**2)

    real_entropy = tf.losses.softmax_cross_entropy(logits=real_clss, onehot_labels=y_ph) * 0.2

    cost = tf.reduce_mean(fake_judge) - tf.reduce_mean(real_judge) + real_entropy
    return cost

def generator_cost(y_ph, real, fake):
    real_clss, real_judge = real
    fake_clss, fake_judge = fake
    # real_entropy = entropy(y_ph, real_clss)
    fake_mse = tf.reduce_mean((y_ph - fake_clss)**2)
    fake_entropy = tf.losses.softmax_cross_entropy(logits=fake_clss, onehot_labels=y_ph) * 0.2
    cost = -tf.reduce_mean(fake_judge) + fake_entropy
    return cost


def encoder_cost(y_ph, G_train_enc):
    ttl_mse = []
    for y, out in zip([y_ph], [G_train_enc]):
        ttl_mse.append(tf.reduce_mean((y-out)**2))
    mse = sum(ttl_mse)
    return mse


def train(modelclass, dt=None):

    batchsize = 64
    gen_learning_rate = 0.001
    dis_learning_rate = 0.001
    enc_learning_rate = 0.001
    bottleneck_dim = 300

    max_epoch = 100
    epoch_look_back = 3
    percent_decrease = 0
    noise_factor = 0.1  #  20170616_1459: 0.05   20170616_1951: 0.01    
    max_outputs = 10

    noise_type = 'normal'

    print('gen_learning_rate:', gen_learning_rate)
    print('dis_learning_rate:', dis_learning_rate)
    print('noise_factor:', noise_factor)
    print('noise_type:', noise_type)


    if dt is None:
        timestamp = tg.utils.ts()
    else:
        timestamp = dt
    save_path = './save/{}/model'.format(timestamp)
    logdir = './log/{}'.format(timestamp)

    X_train, y_train, X_valid, y_valid = Mnist()  
    # X_train, y_train, X_valid, y_valid = X_train[0:10000], y_train[0:10000], X_valid[0:10000], y_valid[0:10000]
    # 0617_1346: 0.05   #0619_1033: 0.01   0619_1528:0.1  0619_1944: 0.3
    # X_train, y_train, X_valid, y_valid = Cifar100()
    # X_train, y_train, X_valid, y_valid = Cifar10(contrast_normalize=False, whiten=False)

    _, h, w, c = X_train.shape
    _, nclass = y_train.shape

    data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
    data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)
    
    gan = getattr(model, modelclass)(h, w, c, nclass, bottleneck_dim)

    y_ph, noise_ph, G_train_sb, G_test_sb, gen_var_list, G_train_enc, G_test_enc, G_train_embed, G_test_embed = gan.generator()
    real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list = gan.discriminator()
    # real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list = gan.discriminator_allconv()

    print('..using model:', gan.__class__.__name__)

    print('Generator Variables')
    for var in gen_var_list:
        print(var.name)

    print('\nDiscriminator Variables')
    for var in dis_var_list:
        print(var.name)
    with gan.tf_graph.as_default():

        gen_train_cost_sb = generator_cost(y_ph, real_train, fake_train)
        fake_clss, fake_judge = fake_train

        dis_train_cost_sb = discriminator_cost(y_ph, real_train, fake_train)
        
        enc_train_cost_sb = encoder_cost(y_ph, G_train_enc)

        gen_train_sm = tf.summary.image('gen_train_img', G_train_sb, max_outputs=max_outputs)
        gen_train_mg = tf.summary.merge([gen_train_sm])

        gen_train_cost_sm = tf.summary.scalar('gen_cost', gen_train_cost_sb)
        dis_train_cost_sm = tf.summary.scalar('dis_cost', dis_train_cost_sb)
        enc_train_cost_sm = tf.summary.scalar('enc_cost', enc_train_cost_sb)
        cost_train_mg = tf.summary.merge([gen_train_cost_sm, dis_train_cost_sm, enc_train_cost_sm])

        gen_optimizer = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_train_cost_sb, var_list=gen_var_list)
        dis_optimizer = tf.train.AdamOptimizer(dis_learning_rate).minimize(dis_train_cost_sb, var_list=dis_var_list)
        enc_optimizer = tf.train.AdamOptimizer(enc_learning_rate).minimize(enc_train_cost_sb)

        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_var_list]
        
        # embedding_var = tf.Variable(tf.zeros([60000, 300]), trainable=False, name="embedding")
        # prepare projector config
        
        # summary_writer = tf.summary.FileWriter(logdir)
        # saver = tf.train.Saver([embedding_var])
            
        
        
        init = tf.global_variables_initializer()
        gan.sess.run(init)
        # es = tg.EarlyStopper(max_epoch=max_epoch,
        #                      epoch_look_back=epoch_look_back,
        #                      percent_decrease=percent_decrease)

        ttl_iter = 0
        error_writer = tf.summary.FileWriter(logdir + '/experiment', gan.sess.graph)
        

        img_writer = tf.summary.FileWriter('{}/orig_img'.format(logdir))
        orig_sm = tf.summary.image('orig_img', real_ph, max_outputs=max_outputs)
        img_writer.add_summary(orig_sm.eval(session=gan.sess, feed_dict={real_ph:data_train[:100].data[0]}))
        img_writer.flush()
        img_writer.close()
        
        #embed = gan.sess.graph.get_tensor_by_name('Generator/genc4')
        # Create metadata
        # embeddir = logdir 
        # if not os.path.exists(embeddir):
        #     os.makedirs(embeddir)
        # metadata_path = os.path.join(embeddir, 'metadata.tsv')
        
        temp_acc = []
        
        for epoch in range(1, max_epoch):
            print('epoch:', epoch)
            print('..training')
            print('..logdir', logdir)
            pbar = tg.ProgressBar(len(data_train))
            n_exp = 0
            ttl_mse = 0
            ttl_gen_cost = 0
            ttl_dis_cost = 0
            ttl_enc_cost = 0
            error_writer.reopen()
            
            if epoch == max_epoch-1:
                output = np.empty([0,300], 'float32')
                labels = np.empty([0,10], 'int32')
            
            # metadata = open(metadata_path, 'w')
            # metadata.write("Name\tLabels\n")

            for X_batch, y_batch in data_train:

                for i in range(3):
                    if noise_type == 'normal':
                        noise = np.random.normal(loc=0, scale=noise_factor, size=(len(X_batch), bottleneck_dim))
                    else:
                        noise = np.random.uniform(-1,1, size=(len(X_batch), bottleneck_dim)) * noise_factor

                    feed_dict = {noise_ph:noise, real_ph:X_batch, y_ph:y_batch}
                    gan.sess.run([dis_optimizer, clip_D], feed_dict=feed_dict)

                for i in range(1):
                    if noise_type == 'normal':
                        noise = np.random.normal(loc=0, scale=noise_factor, size=(len(X_batch), bottleneck_dim))
                    else:
                        noise = np.random.uniform(-1,1, size=(len(X_batch), bottleneck_dim)) * noise_factor

                    feed_dict = {noise_ph:noise, real_ph:X_batch, y_ph:y_batch}
                    gan.sess.run([enc_optimizer, gen_optimizer], feed_dict={noise_ph:noise, real_ph:X_batch, y_ph:y_batch})
                                
                fake_judge_v, cost_train,enc_cost, gen_cost, dis_cost = gan.sess.run([fake_judge, cost_train_mg, enc_train_cost_sb,gen_train_cost_sb,dis_train_cost_sb],
                                                               feed_dict=feed_dict)

                ttl_gen_cost += gen_cost * len(X_batch)
                ttl_dis_cost += dis_cost * len(X_batch)
                ttl_enc_cost += enc_cost * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
                error_writer.add_summary(cost_train, n_exp + ttl_iter)
                error_writer.flush()
                
                if epoch == max_epoch-1:
                    results = gan.sess.run(G_train_embed, feed_dict = {real_ph:X_batch, y_ph:y_batch})
                    output = np.concatenate((output, results), axis = 0)
                    labels = np.concatenate((labels, y_batch), axis = 0)
                # import pdb; pdb.set_trace()
                # for x_row, y_row in zip(X_batch, y_batch):
                #    metadata.write('{}\t{}\n'.format(x_row, y_row))
            # metadata.close()
            error_writer.close()
            
            # import pdb; pdb.set_trace()
            # for ot in output:
            #     temp = tf.stack(ot, axis = 0)
            
            #embedding_var = tf.Variable(temp)
            
            # sess.run(tf.variables_initializer([embedding_var]))
            
            # saver.save(gan.sess, os.path.join(embeddir, 'model.ckpt'))
            
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = embedding_var.name
            # embedding.metadata_path = metadata_path  
            # save embedding_var
            # projector.visualize_embeddings(summary_writer, config)
            
            ttl_iter += n_exp

            mean_gan_cost = ttl_gen_cost / n_exp
            mean_dis_cost = ttl_dis_cost / n_exp
            mean_enc_cost = ttl_enc_cost / n_exp
            print('\nmean train gen cost:', mean_gan_cost)
            print('mean train dis cost:', mean_dis_cost)
            print('enc train dis cost:', mean_enc_cost)
            lab = []
            
            if epoch == max_epoch-1:
                embeddir = './genData/3'
                if not os.path.exists(embeddir):
                    os.makedirs(embeddir)
                lab = np.nonzero(labels)[1]
                np.save(embeddir + 'embed.npy', output)
                np.save(embeddir + 'label.npy', lab)                    
            
                       
            valid_error = 0
            valid_accuracy = 0
            ttl_examples = 0
            for X_batch, ys in data_valid:
                feed_dict = {real_ph:X_batch, y_ph:y_batch}

                valid_outs = gan.sess.run(G_test_enc, feed_dict=feed_dict)
                valid_error += total_mse([valid_outs], [ys])[0]
                valid_accuracy += total_accuracy([valid_outs], [ys])[0]
                ttl_examples += len(X_batch)

            temp_acc.append(valid_accuracy/float(ttl_examples))
            print 'max accuracy is:\t', max(temp_acc)        
        print 'max accuracy is:\t', max(temp_acc)  

    return save_path



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', help='datetime for the initialization of the experiment')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', help='test model')
    parser.add_argument('--modelclass', required=True, help='model class')
    # parser.add_argument('--th', help='threshold')


    args = parser.parse_args()
    print(args)
    if args.train:
        if args.dt:
            modelpath = train(args.modelclass, args.dt)
        else:
            dt = datetime.now()
            dt = dt.strftime('%Y%m%d_%H%M_%S%f')
            modelpath = train(args.modelclass, dt)


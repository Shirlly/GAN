# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:07:52 2017

@author: Zheng Xin
"""



import tensorgraph as tg
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative, Sigmoid, Flatten, Tanh, Linear
from tensorgraph.utils import valid, same, put_kernels_on_grid
from tensorgraph.cost import entropy
import tensorflow as tf
import numpy as np
from data import *
# from model import Gan
import os
from datetime import datetime
from tensorgraph.dataset import Mnist, Cifar100, Cifar10
from scipy.misc import imsave
# from wgan import WGAN
# from augan import AuGan
import __init__ as model
from Classify import classify


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
    fake_mse = tf.reduce_mean((y_ph - fake_clss)**2)
    fake_entropy = tf.losses.softmax_cross_entropy(logits=fake_clss, onehot_labels=y_ph) * 0.2
    cost = -tf.reduce_mean(fake_judge) + fake_entropy
    return cost



def train(modelclass, dt=None):

    batchsize = 64
    gen_learning_rate = 0.001
    dis_learning_rate = 0.001
    bottleneck_dim = 300

    max_epoch = 2
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
    AuX_train = X_train
    Auy_train = y_train
    aux = np.empty((0, 28, 28, 1), 'float32')
    auy = np.empty((0, 10), 'int32')
    # 0617_1346: 0.05   #0619_1033: 0.01   0619_1528:0.1  0619_1944: 0.3
    # X_train, y_train, X_valid, y_valid = Cifar100()
    # X_train, y_train, X_valid, y_valid = Cifar10(contrast_normalize=False, whiten=False)

    _, h, w, c = X_train.shape
    _, nclass = y_train.shape

    data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
    data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)
    
    print '\n====== Before augment data size ', X_train.shape , ' ======\n'
    
    gan = getattr(model, modelclass)(h, w, c, nclass, bottleneck_dim)

    y_ph, noise_ph, G_train_sb, G_test_sb, gen_var_list = gan.generator()
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

        gen_train_sm = tf.summary.image('gen_train_img', G_train_sb, max_outputs=max_outputs)
        gen_train_mg = tf.summary.merge([gen_train_sm])

        gen_train_cost_sm = tf.summary.scalar('gen_cost', gen_train_cost_sb)
        dis_train_cost_sm = tf.summary.scalar('dis_cost', dis_train_cost_sb)
        cost_train_mg = tf.summary.merge([gen_train_cost_sm, dis_train_cost_sm])

        gen_optimizer = tf.train.AdamOptimizer(gen_learning_rate).minimize(gen_train_cost_sb, var_list=gen_var_list)
        dis_optimizer = tf.train.AdamOptimizer(dis_learning_rate).minimize(dis_train_cost_sb, var_list=dis_var_list)

        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in dis_var_list]

        init = tf.global_variables_initializer()
        gan.sess.run(init)
        es = tg.EarlyStopper(max_epoch=max_epoch,
                             epoch_look_back=epoch_look_back,
                             percent_decrease=percent_decrease)

        ttl_iter = 0
        error_writer = tf.summary.FileWriter(logdir + '/experiment', gan.sess.graph)
        
        img_writer = tf.summary.FileWriter('{}/orig_img'.format(logdir))
        orig_sm = tf.summary.image('orig_img', real_ph, max_outputs=max_outputs)
        img_writer.add_summary(orig_sm.eval(session=gan.sess, feed_dict={real_ph:data_train[:100].data[0]}))
        img_writer.flush()
        img_writer.close()

        for epoch in range(1, max_epoch):
            print('epoch:', epoch)
            print('..training')
            print('..logdir', logdir)
            pbar = tg.ProgressBar(len(data_train))
            n_exp = 0
            ttl_mse = 0
            ttl_gen_cost = 0
            ttl_dis_cost = 0
            error_writer.reopen()
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
                    gan.sess.run(gen_optimizer, feed_dict={noise_ph:noise, real_ph:X_batch, y_ph:y_batch})
                                
                fake_judge_v, cost_train, gen_cost, dis_cost = gan.sess.run([fake_judge, cost_train_mg, gen_train_cost_sb, dis_train_cost_sb],
                                                               feed_dict=feed_dict)

                ttl_gen_cost += gen_cost * len(X_batch)
                ttl_dis_cost += dis_cost * len(X_batch)
                n_exp += len(X_batch)
                pbar.update(n_exp)
                error_writer.add_summary(cost_train, n_exp + ttl_iter)
                error_writer.flush()
                
            error_writer.close()

            ttl_iter += n_exp

            mean_gan_cost = ttl_gen_cost / n_exp
            mean_dis_cost = ttl_dis_cost / n_exp
            print('\nmean train gen cost:', mean_gan_cost)
            print('mean train dis cost:', mean_dis_cost)


            if save_path and epoch == max_epoch-1:
                # print('\n..saving best model to: {}'.format(save_path))
                dname = os.path.dirname(save_path)
                if not os.path.exists(dname):
                    os.makedirs(dname)
                print('saved to {}'.format(dname))
                train_writer = tf.summary.FileWriter('{}/experiment/{}'.format(logdir, epoch))
                
                for X_batch, y_batch in data_train:
                    #import pdb; pdb.set_trace()

                    if noise_type == 'normal':
                        noise = np.random.normal(loc=0, scale=noise_factor, size=(len(X_batch), bottleneck_dim))
                    else:
                        noise = np.random.uniform(-1,1, size=(len(X_batch), bottleneck_dim)) * noise_factor

                    feed_dict = {noise_ph:noise, real_ph:X_batch, y_ph:y_batch}
                    G_train, G_img, fake_dis = gan.sess.run([G_train_sb, gen_train_mg, fake_train], feed_dict=feed_dict)
                    fake_class_dis, fake_judge_dis = fake_dis
                    idx = [i for i,v in enumerate(fake_judge_dis) if v>0.5]
                    aux = np.concatenate((aux, G_train[idx]), axis = 0)
                    auy = np.concatenate((auy, fake_class_dis[idx]), axis = 0)
                    AuX_train = np.concatenate((G_train, AuX_train), axis = 0)
                    Auy_train = np.concatenate((y_batch, Auy_train), axis = 0)
                    # temp_data = zip(G_img, y_batch)
                    # aug_data.append(temp_data)
                    train_writer.add_summary(G_img)                    
                    train_writer.flush()
                train_writer.close()
                xname = 'genx.npy'
                yname = 'geny.npy'
                np.save('{}/{}'.format(logdir, xname), aux)
                np.save('{}/{}'.format(logdir, yname), auy)
        
        print '\n====== Augment data size ', AuX_train.shape , ' ======\n'
        print '\n====== Augment data size ', Auy_train.shape , ' ======\n'
        

    return save_path, X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy



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
            modelpath, X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy = train(args.modelclass, dt)
            classify(X_train, y_train, X_valid, y_valid, AuX_train, Auy_train, aux, auy)


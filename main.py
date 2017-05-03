

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

def clip(X):
    epsilon = 1e-15
    return tf.clip_by_value(X, epsilon, 1.0-epsilon)


def ph2onehot(ph, char_embed_dim):
    seq = tg.Sequential()
    seq.add(OneHot(char_embed_dim))
    seq.add(Transpose((0,3,2,1)))
    oh = seq.train_fprop(ph)
    return oh


# def generator_cost(y_ph, fake):
#     # real_clss, real_judge = real
#     fake_clss, fake_judge = fake
#     fake_judge = clip(fake_judge)
#     cost = -0.5*tf.reduce_mean(tf.log(fake_judge))
#     # cost = 0.5*tf.reduce_mean(tf.log(1-fake_judge))
#     # real_entropy = entropy(y_ph, real_clss)
#     # fake_entropy = entropy(y_ph, fake_clss)
#     # ttl_cost = cost + fake_entropy
#     # return ttl_cost
#     return cost
#
#
# def discriminator_cost(y_ph, real, fake):
#     real_clss, real_judge = real
#     fake_clss, fake_judge = fake
#     real_judge = clip(real_judge)
#     fake_judge = clip(fake_judge)
#     cost = -0.5*tf.reduce_mean(tf.log(real_judge)) - 0.5*tf.reduce_mean(tf.log(1-fake_judge))
#     # fake_entropy = entropy(y_ph, fake_clss)
#     # real_entropy = entropy(y_ph, real_clss)
#     # ttl_cost = cost + real_entropy
#     # return ttl_cost
#     return cost


def sigmoid_cross_entropy_with_logits(x, y):
  try:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
  except:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
#
# self.d_loss_real = tf.reduce_mean(
#   sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
# self.d_loss_fake = tf.reduce_mean(
#   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
# self.g_loss = tf.reduce_mean(
#   sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
#
# self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
# self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
#
# self.d_loss = self.d_loss_real + self.d_loss_fake
#
# self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
# self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
#
#


def discriminator_cost(y_ph, real, fake):
    real_clss, real_judge = real
    fake_clss, fake_judge = fake
    # real_judge = clip(real_judge)
    # fake_judge = clip(fake_judge)
    # cost = -0.5*tf.reduce_mean(tf.log(real_judge)) - 0.5*tf.reduce_mean(tf.log(1-fake_judge))
    # fake_entropy = entropy(y_ph, fake_clss)
    real_mse = tf.reduce_mean((y_ph - real_clss)**2)
    real_entropy = entropy(y_ph, real_clss)
    # ttl_cost = cost + real_entropy
    # return ttl_cost

    # d_loss_real = tf.reduce_mean(
    # sigmoid_cross_entropy_with_logits(real_judge, tf.ones_like(real_judge)))
    #
    # d_loss_fake = tf.reduce_mean(
    # sigmoid_cross_entropy_with_logits(fake_judge, tf.zeros_like(fake_judge)))
    #
    # cost = d_loss_real + d_loss_fake
    # return cost
    cost = tf.reduce_mean(fake_judge) - tf.reduce_mean(real_judge) + real_entropy
    return cost



def generator_cost(y_ph, real, fake):
    real_clss, real_judge = real
    fake_clss, fake_judge = fake
    # real_entropy = entropy(y_ph, real_clss)
    fake_mse = tf.reduce_mean((y_ph - fake_clss)**2)
    fake_entropy = entropy(y_ph, fake_clss)
    # fake_judge = clip(fake_judge)
    # cost = -0.5*tf.reduce_mean(tf.log(fake_judge))
    # cost = 0.5*tf.reduce_mean(tf.log(1-fake_judge))
    # real_entropy = entropy(y_ph, real_clss)
    # fake_entropy = entropy(y_ph, fake_clss)
    # ttl_cost = cost + fake_entropy
    # return ttl_cost
    cost = -tf.reduce_mean(fake_judge) + fake_entropy
    return cost
    # cost = tf.reduce_mean(sigmoid_cross_entropy_with_logits(fake_judge, tf.ones_like(fake_judge)))
    # return cost


def train(modelclass, dt=None):

    batchsize = 64
    gen_learning_rate = 0.001
    dis_learning_rate = 0.001
    bottleneck_dim = 300

    max_epoch = 1000
    epoch_look_back = 3
    percent_decrease = 0
    noise_factor = 0.05
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

    # X_train, y_train, X_valid, y_valid = Mnist()
    # X_train, y_train, X_valid, y_valid = Cifar100()
    X_train, y_train, X_valid, y_valid = Cifar10()
    _, h, w, c = X_train.shape
    _, nclass = y_train.shape

    data_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
    data_valid = tg.SequentialIterator(X_valid, y_valid, batchsize=batchsize)
    # gan = AuGan(h, w, nclass, bottleneck_dim)
    gan = getattr(model, modelclass)(h, w, c, nclass, bottleneck_dim)

    y_ph, noise_ph, G_train_sb, G_test_sb, gen_var_list = gan.generator()
    real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list = gan.discriminator()

    print('..using model:', gan.__class__.__name__)

    print('Generator Variables')
    for var in gen_var_list:
        print(var.name)

    print('\nDiscriminator Variables')
    for var in dis_var_list:
        print(var.name)

    with gan.tf_graph.as_default():
        # X_oh = ph2onehot(X_ph)


        # train_mse = tf.reduce_mean((X_ph - G_train_s)**2)
        # valid_mse = tf.reduce_mean((X_ph - G_valid_s)**2)
        # gen_train_cost_sb = generator_cost(class_train_sb, judge_train_sb)
        # gen_valid_cost_sb = generator_cost(class_test_sb, judge_test_sb)
        gen_train_cost_sb = generator_cost(y_ph, real_train, fake_train)
        fake_clss, fake_judge = fake_train

        dis_train_cost_sb = discriminator_cost(y_ph, real_train, fake_train)
        # dis_train_cost_sb = discriminator_cost(class_train_sb, judge_train_sb)
        # dis_valid_cost_sb = disciminator_cost(class_test_sb, judge_test_sb)

        # gen_train_img = put_kernels_on_grid(G_train_sb, batchsize)
        #
        gen_train_sm = tf.summary.image('gen_train_img', G_train_sb, max_outputs=max_outputs)
        gen_train_mg = tf.summary.merge([gen_train_sm])

        gen_train_cost_sm = tf.summary.scalar('gen_cost', gen_train_cost_sb)
        dis_train_cost_sm = tf.summary.scalar('dis_cost', dis_train_cost_sb)
        cost_train_mg = tf.summary.merge([gen_train_cost_sm, dis_train_cost_sm])


        # gen_optimizer = tf.train.RMSPropOptimizer(gen_learning_rate).minimize(gen_train_cost_sb, var_list=gen_var_list)
        # dis_optimizer = tf.train.RMSPropOptimizer(dis_learning_rate).minimize(dis_train_cost_sb, var_list=dis_var_list)

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
        # import pdb; pdb.set_trace()
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


            if save_path:
                # print('\n..saving best model to: {}'.format(save_path))
                dname = os.path.dirname(save_path)
                if not os.path.exists(dname):
                    os.makedirs(dname)
                print('saved to {}'.format(dname))
                # gan.save(save_path)

                for X_batch, y_batch in data_train:

                    if noise_type == 'normal':
                        noise = np.random.normal(loc=0, scale=noise_factor, size=(len(X_batch), bottleneck_dim))
                    else:
                        noise = np.random.uniform(-1,1, size=(len(X_batch), bottleneck_dim)) * noise_factor

                    feed_dict = {noise_ph:noise, real_ph:X_batch, y_ph:y_batch}
                    G_train, G_img = gan.sess.run([G_train_sb, gen_train_mg], feed_dict=feed_dict)
                    train_writer = tf.summary.FileWriter('{}/experiment/{}'.format(logdir, epoch))

                    train_writer.add_summary(G_img)

                    train_writer.flush()
                    train_writer.close()

                    break



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
    # if args.test:
    #     test(args.test, indexes=range(20,100))
    # else:
    #     test(modelpath, indexes=range(20,100))
        # load_model_test('./save/{}/model.tf'.format(args.test))

    # modelpath = train()
    # modelpath = './save/20170306_0943_38310463/model'
    # modelpath = './save/20170306_0911_01543045/model'
    # modelpath = './save/20170306_0906_24782458/model'
    # modelpath = './save/20170306_0559_16967210/model'
    # modelpath = './save/20170306_0855_59252744/model'
    # test(modelpath, indexes=range(20))
    # test(model)

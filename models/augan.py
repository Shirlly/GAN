
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative, \
                               Sigmoid, Flatten, Tanh, Linear, Reshape, Dropout, \
                               OneHot, Transpose, Template, Softmax, \
                               BatchNormalization, Select, SetShape, LeakyRELU, \
                               TFBatchNormalization, Concat, Sum, AvgPooling
from tensorgraph.utils import valid, same



class AuGan(object):

    def __init__(self, h, w, c, nclass, bottleneck_dim):
        self.h = h
        self.w = w
        self.c = c
        self.nclass = nclass
        self.bottleneck_dim = bottleneck_dim
        with tf.Graph().as_default() as self.tf_graph:
            self.sess = tf.Session()
            self.real_ph = tf.placeholder('float32', [None, self.h, self.w, self.c], name='real')
            self.noise_ph = tf.placeholder('float32', [None, self.bottleneck_dim], name='noise')
            self.y_ph = tf.placeholder('float32', [None, self.nclass], name='y')
        self.generator_called = False




    def discriminator_allconv(self):
        if not self.generator_called:
            raise Exception('self.generator() has to be called first before self.discriminator()')
        scope = 'Discriminator'
        with self.tf_graph.as_default():
            with tf.name_scope(scope):
                # h1, w1 = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                # h2, w2 = valid(h1, w1, kernel_size=(5,5), stride=(2,2))
                # h3, w3 = valid(h2, w2, kernel_size=(5,5), stride=(2,2))
                # flat_dim = int(h3*w3*32)

                dis_real_sn = tg.StartNode(input_vars=[self.real_ph])

                # fake_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='fake')
                # fake_sn = tg.StartNode(input_vars=[fake_ph])

                h, w = same(in_height=self.h, in_width=self.w, stride=(1,1), kernel_size=(3,3))
                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
                h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

                h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))

                h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
                print('h, w', h, w)
                print('===============')
                # h, w = valid(in_height=h, in_width=w, stride=(1,1), kernel_size=(h,w))

                disc_hn = tg.HiddenNode(prev=[dis_real_sn, self.gen_hn],
                        layers=[
                                Dropout(0.2),
                                # TFBatchNormalization(name='b0'),
                                Conv2D(input_channels=self.c, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                TFBatchNormalization(name='b1'),
                                # Dropout(0.5),

                                Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                # TFBatchNormalization(name='b2'),
                                Dropout(0.5),

                                Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(2, 2), padding='SAME'),
                                LeakyRELU(),
                                TFBatchNormalization(name='b3'),
                                # Dropout(0.5),

                                Conv2D(input_channels=96, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                # TFBatchNormalization(name='b4'),
                                Dropout(0.5),

                                Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                TFBatchNormalization(name='b5'),
                                # Dropout(0.5),

                                Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(2, 2), padding='SAME'),
                                LeakyRELU(),
                                # TFBatchNormalization(name='b6'),
                                Dropout(0.5),

                                Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                TFBatchNormalization(name='b7'),
                                # Dropout(0.5),


                                Conv2D(input_channels=192, num_filters=192, kernel_size=(1, 1), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                # TFBatchNormalization(name='b8'),
                                Dropout(0.5),

                                Conv2D(input_channels=192, num_filters=self.nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'),
                                LeakyRELU(),
                                TFBatchNormalization(name='b9'),
                                # Dropout(0.5),

                                AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'),
                                Flatten(),
                                ])

                print('h,w', h, w)
                print('==============')
                class_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.nclass, self.nclass),
                                                 #　Softmax()
                                                 ])

                judge_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.nclass, 1),
                                                #  Sigmoid()
                                                 ])

                real_class_en = tg.EndNode(prev=[class_hn])
                real_judge_en = tg.EndNode(prev=[judge_hn])

                fake_class_en = tg.EndNode(prev=[class_hn])
                fake_judge_en = tg.EndNode(prev=[judge_hn])

                graph = tg.Graph(start=[dis_real_sn], end=[real_class_en, real_judge_en])
                real_train = graph.train_fprop()
                real_valid = graph.test_fprop()

                graph = tg.Graph(start=[self.noise_sn, self.gen_real_sn], end=[fake_class_en, fake_judge_en])
                fake_train = graph.train_fprop()
                fake_valid = graph.test_fprop()

                dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return self.real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list



    def discriminator(self):
        if not self.generator_called:
            raise Exception('self.generator() has to be called first before self.discriminator()')
        scope = 'Discriminator'
        with self.tf_graph.as_default():
            with tf.name_scope(scope):
                h1, w1 = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                h2, w2 = valid(h1, w1, kernel_size=(5,5), stride=(2,2))
                h3, w3 = valid(h2, w2, kernel_size=(5,5), stride=(2,2))
                flat_dim = int(h3*w3*32)

                dis_real_sn = tg.StartNode(input_vars=[self.real_ph])

                # fake_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='fake')
                # fake_sn = tg.StartNode(input_vars=[fake_ph])

                disc_hn = tg.HiddenNode(prev=[dis_real_sn, self.gen_hn],
                                        layers=[Conv2D(input_channels=self.c, num_filters=32, kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d1'),
                                                # BatchNormalization(layer_type='conv', dim=32, short_memory=0.01),
                                                LeakyRELU(),
                                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d2'),
                                                # BatchNormalization(layer_type='conv', dim=32, short_memory=0.01),
                                                LeakyRELU(),
                                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d3'),
                                                # BatchNormalization(layer_type='conv', dim=32, short_memory=0.01),
                                                LeakyRELU(),
                                                #    Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                #    RELU(),
                                                Flatten(),
                                                Linear(flat_dim, self.bottleneck_dim),
                                                # BatchNormalization(layer_type='fc', dim=self.bottleneck_dim, short_memory=0.01),
                                                TFBatchNormalization(name=scope + '/d4'),
                                                LeakyRELU(),
                                                # Dropout(0.5),
                                                ])


                class_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.bottleneck_dim, self.nclass),
                                                 Softmax()
                                                 ])

                judge_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.bottleneck_dim, 1),
                                                 Sigmoid()
                                                 ])

                real_class_en = tg.EndNode(prev=[class_hn])
                real_judge_en = tg.EndNode(prev=[judge_hn])

                fake_class_en = tg.EndNode(prev=[class_hn])
                fake_judge_en = tg.EndNode(prev=[judge_hn])

                graph = tg.Graph(start=[dis_real_sn], end=[real_class_en, real_judge_en])
                real_train = graph.train_fprop()
                real_valid = graph.test_fprop()

                graph = tg.Graph(start=[self.noise_sn, self.gen_real_sn], end=[fake_class_en, fake_judge_en])
                fake_train = graph.train_fprop()
                fake_valid = graph.test_fprop()

                dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return self.real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list




    def generator(self):
        self.generator_called = True
        with self.tf_graph.as_default():
            scope = 'Generator'
            with tf.name_scope(scope):
                h1, w1 = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                h2, w2 = valid(h1, w1, kernel_size=(5,5), stride=(2,2))
                h3, w3 = valid(h2, w2, kernel_size=(5,5), stride=(2,2))
                flat_dim = int(h3*w3*32)
                print('h1:{}, w1:{}'.format(h1, w1))
                print('h2:{}, w2:{}'.format(h2, w2))
                print('h3:{}, w3:{}'.format(h3, w3))
                print('flat dim:{}'.format(flat_dim))

                self.gen_real_sn = tg.StartNode(input_vars=[self.real_ph])

                enc_hn = tg.HiddenNode(prev=[self.gen_real_sn],
                                       layers=[Conv2D(input_channels=self.c, num_filters=32, kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                               TFBatchNormalization(name=scope + '/genc1'),
                                               RELU(),
                                               Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                               TFBatchNormalization(name=scope + '/genc2'),
                                               RELU(),
                                               Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                               TFBatchNormalization(name=scope + '/genc3'),
                                               RELU(),
                                            #    Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                            #    RELU(),
                                               Flatten(),
                                               Linear(flat_dim, 300),
                                               TFBatchNormalization(name=scope + '/genc4'),
                                               RELU(),
                                               Linear(300, self.bottleneck_dim),
                                               Tanh(),
                                               ])


                self.noise_sn = tg.StartNode(input_vars=[self.noise_ph])

                self.gen_hn = tg.HiddenNode(prev=[self.noise_sn, enc_hn], input_merge_mode=Sum(),
                                            layers=[Linear(self.bottleneck_dim, flat_dim),
                                               RELU(),


                                               ######[ Method 0 ]######
                                            #    Reshape((-1, h3, w3, 32)),
                                            #    Conv2D_Transpose(input_channels=32, num_filters=100, output_shape=(h2,w2),
                                            #                     kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                               ######[ End Method 0 ]######




                                               ######[ Method 1 ]######
                                               Reshape((-1, 1, 1, flat_dim)),
                                            #    Reshape((-1, h))
                                               Conv2D_Transpose(input_channels=flat_dim, num_filters=200, output_shape=(h3,w3),
                                                                kernel_size=(h3,w3), stride=(1,1), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=200, short_memory=0.01),
                                               TFBatchNormalization(name=scope + '/g1'),
                                               RELU(),
                                               Conv2D_Transpose(input_channels=200, num_filters=100, output_shape=(h2,w2),
                                                                kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=100, short_memory=0.01),
                                               ######[ End Method 1 ]######


                                               TFBatchNormalization(name=scope + '/g2'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=100, num_filters=50, output_shape=(h1,w1),
                                                                kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=50, short_memory=0.01),
                                               TFBatchNormalization(name=scope + '/g3'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=50, num_filters=self.c, output_shape=(self.h, self.w),
                                                                kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                               SetShape((-1, self.h, self.w, self.c)),
                                               Sigmoid()])

                h,w = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                h,w = valid(h, w, kernel_size=(5,5), stride=(2,2))
                h,w = valid(h, w, kernel_size=(5,5), stride=(2,2))
                h,w = valid(h, w, kernel_size=(h3,w3), stride=(1,1))

                y_en = tg.EndNode(prev=[self.gen_hn])

                graph = tg.Graph(start=[self.noise_sn, self.gen_real_sn], end=[y_en])

                G_train_sb = graph.train_fprop()[0]
                G_test_sb = graph.test_fprop()[0]
                gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return self.y_ph, self.noise_ph, G_train_sb, G_test_sb, gen_var_list


    # def load(self, modelpath):
    #     X_ph, G_train_s, G_valid_s = self.make_model()
    #     with self.tf_graph.as_default():
    #         saver = tf.train.Saver()
    #         saver.restore(self.sess, modelpath)
    #     return X_ph, noise_ph, G_train_s, G_valid_s


    def save(self, path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path)


    def close(self):
        self.sess.close()


import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative, \
                               Sigmoid, Flatten, Tanh, Linear, Reshape, Dropout, \
                               OneHot, Transpose, Template, Softmax, \
                               BatchNormalization, Select, SetShape, LeakyRELU, \
                               TFBatchNormalization, Concat, Sum
from tensorgraph.utils import valid, same


class WGan(object):


    def __init__(self, char_embed_dim, sent_len, word_len, nclass, bottleneck_dim):
        self.char_embed_dim = char_embed_dim
        self.sent_len = sent_len
        self.word_len = word_len
        self.nclass = nclass
        self.bottleneck_dim = bottleneck_dim

        with tf.Graph().as_default() as self.tf_graph:
            self.sess = tf.Session()
            self.real_ph = tf.placeholder('int32', [None, sent_len, word_len], name='real')
            self.noise_ph = tf.placeholder('float32', [None, self.bottleneck_dim], name='noise')
            self.y_ph = tf.placeholder('float32', [None, self.nclass], name='y')
        self.generator_called = False


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

                h1, w1 = valid(self.char_embed_dim, self.word_len, kernel_size=(self.char_embed_dim,3), stride=(1,1))
                print('h1:{}, w1:{}'.format(h1, w1))
                h2, w2 = valid(h1, w1, kernel_size=(1,3), stride=(1,1))
                print('h2:{}, w2:{}'.format(h2, w2))
                h3, w3 = valid(h2, w2, kernel_size=(1,3), stride=(1,1))
                print('h3:{}, w3:{}'.format(h3, w3))
                # h4, w4 = valid(h3, w3, kernel_size=(1,6), stride=(1,1))
                # print('h4:{}, w4:{}'.format(h4, w4))
                # hf, wf = h4, w4
                hf, wf = h3, w3
                n_filters = 100


                real_sn = tg.StartNode(input_vars=[self.real_ph])

                real_hn = tg.HiddenNode(prev=[real_sn], layers=[OneHot(self.char_embed_dim), Transpose(perm=[0,3,2,1])])

                disc_hn = tg.HiddenNode(prev=[real_hn, self.gen_hn],
                                        layers=[Conv2D(input_channels=self.sent_len, num_filters=100, kernel_size=(self.char_embed_dim,3), stride=(1,1), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d1'),
                                                LeakyRELU(),

                                                Conv2D(input_channels=100, num_filters=100, kernel_size=(1,3), stride=(1,1), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d2'),
                                                LeakyRELU(),

                                                Conv2D(input_channels=100, num_filters=100, kernel_size=(1,3), stride=(1,1), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/d3'),
                                                LeakyRELU(),
                                                # Conv2D(input_channels=32, num_filters=128, kernel_size=(1,6), stride=(1,1), padding='VALID'),
                                                # RELU(),
                                                Flatten(),
                                                Linear(int(hf*wf*n_filters), self.bottleneck_dim),
                                                TFBatchNormalization(name=scope + '/d4'),
                                                LeakyRELU(),
                                                ])


                class_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.bottleneck_dim, self.nclass),
                                                 Softmax()])

                judge_hn = tg.HiddenNode(prev=[disc_hn],
                                         layers=[Linear(self.bottleneck_dim, 1),
                                                #  Sigmoid()
                                                 ])

                real_class_en = tg.EndNode(prev=[class_hn])
                real_judge_en = tg.EndNode(prev=[judge_hn])

                fake_class_en = tg.EndNode(prev=[class_hn])
                fake_judge_en = tg.EndNode(prev=[judge_hn])


                graph = tg.Graph(start=[real_sn], end=[real_class_en, real_judge_en])

                real_train = graph.train_fprop()
                real_valid = graph.test_fprop()

                graph = tg.Graph(start=[self.noise_sn], end=[fake_class_en, fake_judge_en])
                fake_train = graph.train_fprop()
                fake_valid = graph.test_fprop()

                dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


        return self.real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list




    def generator(self):
        self.generator_called = True
        with self.tf_graph.as_default():
            scope = 'Generator'
            with tf.name_scope(scope):
                # X_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='X')
                # X_sn = tg.StartNode(input_vars=[X_ph])

                self.noise_sn = tg.StartNode(input_vars=[self.noise_ph])

                h1, w1 = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                h2, w2 = valid(h1, w1, kernel_size=(5,5), stride=(2,2))
                h3, w3 = valid(h2, w2, kernel_size=(5,5), stride=(2,2))
                flat_dim = int(h3*w3*32)
                print('h1:{}, w1:{}'.format(h1, w1))
                print('h2:{}, w2:{}'.format(h2, w2))
                print('h3:{}, w3:{}'.format(h3, w3))
                print('flat dim:{}'.format(flat_dim))


                self.y_sn = tg.StartNode(input_vars=[self.y_ph])


                self.gen_hn = tg.HiddenNode(prev=[self.noise_sn],
                                       layers=[Linear(self.bottleneck_dim, flat_dim),
                                               RELU(),


                                               ######[ Method 0 ]######
                                            #    Reshape((-1, h3, w3, 32)),
                                            #    Conv2D_Transpose(input_channels=32, num_filters=100, output_shape=(h2,w2),
                                            #                     kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                               ######[ End Method 0 ]######




                                               ######[ Method 1 ]######
                                               Reshape((-1, 1, 1, flat_dim)),
                                               Conv2D_Transpose(input_channels=flat_dim, num_filters=200, output_shape=(2,2),
                                                                kernel_size=(2,2), stride=(1,1), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=200, short_memory=0.01),
                                               TFBatchNormalization(name=scope + '/g1'),
                                               RELU(),
                                               Conv2D_Transpose(input_channels=200, num_filters=100, output_shape=(h2,w2),
                                                                kernel_size=(9,9), stride=(1,1), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=100, short_memory=0.01),
                                               ######[ End Method 1 ]######


                                               TFBatchNormalization(name=scope + '/g2'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=100, num_filters=50, output_shape=(h1,w1),
                                                                kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                            #    BatchNormalization(layer_type='conv', dim=50, short_memory=0.01),
                                               TFBatchNormalization(name=scope + '/g3'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=50, num_filters=1, output_shape=(self.h, self.w),
                                                                kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                               SetShape((-1, self.h, self.w, 1)),
                                               Sigmoid()])

                y_en = tg.EndNode(prev=[self.gen_hn])
                graph = tg.Graph(start=[self.noise_sn], end=[y_en])

                G_train_sb = graph.train_fprop()[0]
                G_test_sb = graph.test_fprop()[0]
                gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return self.y_ph, self.noise_ph, G_train_sb, G_test_sb, gen_var_list


    def save(self, path):
        with self.tf_graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, path)


    def close(self):
        self.sess.close()

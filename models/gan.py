
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.layers import Conv2D, Conv2D_Transpose, RELU, Iterative, \
                               Sigmoid, Flatten, Tanh, Linear, Reshape, Dropout, \
                               OneHot, Transpose, Template, Softmax, \
                               BatchNormalization, Select, SetShape, LeakyRELU, \
                               TFBatchNormalization, Concat
from tensorgraph.utils import valid, same


def ph2onehot(ph, charlen):
    seq = tg.Sequential()
    seq.add(OneHot(charlen)) #[?, charlen, char_embed_dim]
    seq.add(Expand_Dims(-1)) # [?, charlen, char_embed_dim, 1]
    seq.add(Transpose((0,2,1,3))) # [?, char_embed_dim, charlen, 1]
    oh = seq.train_fprop(ph)
    return oh



class Gan(object):

    def __init__(self, h, w, c, nclass, bottleneck_dim):
        self.h = w
        self.w = w
        self.c = c
        self.nclass = nclass
        self.bottleneck_dim = bottleneck_dim
        with tf.Graph().as_default() as self.tf_graph:
            self.sess = tf.Session()
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
                real_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='real')
                real_sn = tg.StartNode(input_vars=[real_ph])

                # fake_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='fake')
                # fake_sn = tg.StartNode(input_vars=[fake_ph])

                disc_hn = tg.HiddenNode(prev=[real_sn, self.gen_hn],
                                        layers=[Conv2D(input_channels=1, num_filters=32, kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/c1'),
                                                LeakyRELU(),
                                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/c2'),
                                                LeakyRELU(),
                                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                TFBatchNormalization(name=scope + '/c3'),
                                                LeakyRELU(),
                                                #    Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                                #    RELU(),
                                                Flatten(),
                                                Linear(flat_dim, self.bottleneck_dim),
                                                TFBatchNormalization(name=scope + '/l1'),
                                                LeakyRELU(),
                                                # Dropout(0.5),
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
                # dis_var_list = graph.variables
                # for var in dis_var_list:
                    # print var.name


                graph = tg.Graph(start=[self.noise_sn, self.y_sn], end=[fake_class_en, fake_judge_en])
                fake_train = graph.train_fprop()
                fake_valid = graph.test_fprop()

                # print('========')
                # for var in graph.variables:
                    # print var.name





                dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                # for var in dis_var_list:
                #     print(var.name)
                #
                # print('=========')
                # for var in tf.global_variables():
                #     print(var.name)
                # import pdb; pdb.set_trace()
                # print()

            # graph = tg.Graph(start=[G_sn], end=[class_en, judge_en])
            # class_train_sb, judge_train_sb = graph.train_fprop() # symbolic outputs
            # class_test_sb, judge_test_sb = graph.test_fprop() # symbolic outputs

        return real_ph, real_train, real_valid, fake_train, fake_valid, dis_var_list




    def generator(self):
        self.generator_called = True
        with self.tf_graph.as_default():
            scope = 'Generator'
            with tf.name_scope(scope):
                # X_ph = tf.placeholder('float32', [None, self.h, self.w, 1], name='X')
                # X_sn = tg.StartNode(input_vars=[X_ph])
                noise_ph = tf.placeholder('float32', [None, self.bottleneck_dim], name='noise')
                self.noise_sn = tg.StartNode(input_vars=[noise_ph])

                h1, w1 = valid(self.h, self.w, kernel_size=(5,5), stride=(1,1))
                h2, w2 = valid(h1, w1, kernel_size=(5,5), stride=(2,2))
                h3, w3 = valid(h2, w2, kernel_size=(5,5), stride=(2,2))
                flat_dim = int(h3*w3*32)
                print('h1:{}, w1:{}'.format(h1, w1))
                print('h2:{}, w2:{}'.format(h2, w2))
                print('h3:{}, w3:{}'.format(h3, w3))
                print('flat dim:{}'.format(flat_dim))

                # enc_hn = tg.HiddenNode(prev=[X_sn],
                #                        layers=[Conv2D(input_channels=1, num_filters=32, kernel_size=(5,5), stride=(1,1), padding='VALID'),
                #                                RELU(),
                #                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                #                                RELU(),
                #                                Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                #                                RELU(),
                #                             #    Conv2D(input_channels=32, num_filters=32, kernel_size=(5,5), stride=(2,2), padding='VALID'),
                #                             #    RELU(),
                #                                Flatten(),
                #                                Linear(flat_dim, 300),
                #                                RELU(),
                #                                # seq.add(Dropout(0.5))
                #                                Linear(300, self.bottleneck_dim),
                #                                Tanh(),
                #                                ])

                y_ph = tf.placeholder('float32', [None, self.nclass], name='y')
                self.y_sn = tg.StartNode(input_vars=[y_ph])

                noise_hn = tg.HiddenNode(prev=[self.noise_sn, self.y_sn], input_merge_mode=Concat(1))

                self.gen_hn = tg.HiddenNode(prev=[noise_hn],
                                       layers=[Linear(self.bottleneck_dim+10, flat_dim),
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
                                               TFBatchNormalization(name=scope + '/dc1'),
                                               RELU(),
                                               Conv2D_Transpose(input_channels=200, num_filters=100, output_shape=(h2,w2),
                                                                kernel_size=(9,9), stride=(1,1), padding='VALID'),
                                               ######[ End Method 1 ]######


                                               TFBatchNormalization(name=scope + '/dc2'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=100, num_filters=50, output_shape=(h1,w1),
                                                                kernel_size=(5,5), stride=(2,2), padding='VALID'),
                                               TFBatchNormalization(name=scope + '/dc3'),
                                               RELU(),

                                               Conv2D_Transpose(input_channels=50, num_filters=1, output_shape=(self.h, self.w),
                                                                kernel_size=(5,5), stride=(1,1), padding='VALID'),
                                               SetShape((-1, self.h, self.w, 1)),
                                               Sigmoid()])

                y_en = tg.EndNode(prev=[self.gen_hn])
                graph = tg.Graph(start=[self.noise_sn, self.y_sn], end=[y_en])

                G_train_sb = graph.train_fprop()[0]
                G_test_sb = graph.test_fprop()[0]
                # import pdb; pdb.set_trace()
                gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

        return y_ph, noise_ph, G_train_sb, G_test_sb, gen_var_list


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

# coding=utf-8
# =============================================
# @Time      : 2022-03-24 15:02
# @Author    : DongWei1998
# @FileName  : network.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, Dense, Concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model





# 获取变量默认的初始化器
def fetch_variable_initializer():
    """
    :return:
    """
    return tf.random_normal_initializer(0.0, 0.1)

# 获取变量默认的正则化器
def fetch_variable_regularizer():
    return None





class TextCNN(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 kernel_sizes=None,
                 kernel_filter = None,
                 kernel_filters = None,
                 kernel_regularizer=None,
                 last_activation=None
                 ):
        '''
        :param maxlen: 文本最大长度
        :param max_features: 词典大小
        :param embedding_dims: embedding维度大小
        :param kernel_sizes: 滑动卷积窗口大小的list, eg: [2,3,4]
        :param kernel_filters: 滑动卷积盒子深度, eg: [32,64,128]
        :param kernel_regularizer: eg: tf.keras.regularizers.l2(0.001)
        :param class_num:
        :param last_activation:
        '''
        super(TextCNN, self).__init__()
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.embedding = Embedding(input_dim=max_features, output_dim=embedding_dims, input_length=maxlen)
        self.conv1s = []
        self.maxpools = []
        if kernel_regularizer is None or kernel_regularizer == 'None':
            kernel_regularizer = tf.keras.regularizers.l2(0.001)
        if isinstance (kernel_filters,list) and len(kernel_sizes) == len(kernel_filters):
            for i in range(len(kernel_sizes)):
                self.conv1s.append(Conv1D(filters=kernel_filters[i], kernel_size=kernel_sizes[i], activation='relu',
                                          kernel_regularizer=kernel_regularizer))
                self.maxpools.append(GlobalMaxPooling1D())
        else:
            for kernel_size in kernel_sizes:
                self.conv1s.append(Conv1D(filters=kernel_filter[0], kernel_size=kernel_size, activation='relu',
                                          kernel_regularizer=kernel_regularizer))
                self.maxpools.append(GlobalMaxPooling1D())

        self.classifier = Dense(class_num, activation=last_activation, )

    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))

        emb = self.embedding(inputs)
        conv1s = []
        for i in range(len(self.kernel_sizes)):
            c = self.conv1s[i](emb) # (batch_size, maxlen-kernel_size+1, filters)
            c = self.maxpools[i](c) # # (batch_size, filters)
            conv1s.append(c)
        x = Concatenate()(conv1s) # (batch_size, len(self.kernel_sizes)*filters)
        output = self.classifier(x)
        return output


    def build_graph(self, input_shape):
        '''自定义函数，在调用model.summary()之前调用
        '''
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        if not hasattr(self, 'call'):
            raise AttributeError("User should define 'call' method in sub-class model!")
        _ = self.call(inputs)



if __name__ == '__main__':
    model  = TextCNN(
        maxlen=2000,
        max_features=1000,
        embedding_dims=128,
        class_num=2,
        kernel_sizes=[2, 3, 4],
        kernel_filter = [128],
        kernel_filters = [32,64,128],
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        last_activation='softmax',
    )

    # 自定义函数 用于查看网络结构
    model.build_graph(input_shape=(None, 2000))
    model.summary()










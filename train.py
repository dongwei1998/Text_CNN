# coding=utf-8
# =============================================
# @Time      : 2022-03-23 17:55
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.network import TextCNN
from utils.parameter import parser_opt
from utils.gpu_git import check_gpus
from utils.data_help import *
from utils.token_2_data import Tokenizer_tool

class ModelHepler:
    def __init__(self, args):
        self.args = args
        self.callback_list = []
        self.create_model()
        self.tensorboard_log_dir = args.tensorboard_dir + '/' +f'{args.model_name}-epoch-{args.num_epochs}-emb-{args.embedding_dims}'
    def create_model(self):
        self.args.logger.info('Bulid Model...')
        model = TextCNN(maxlen=self.args.maxlen,
                        max_features=self.args.max_features,
                        embedding_dims=self.args.embedding_dims,
                        class_num=self.args.class_num,
                        kernel_sizes=self.args.kernel_sizes,
                        kernel_filter=self.args.kernel_filter,
                        kernel_filters=self.args.kernel_filters,
                        kernel_regularizer=self.args.kernel_regularizer,
                        last_activation=self.args.last_activation, )
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy'],
        )
        model.build_graph(input_shape=(None, self.args.maxlen))
        model.summary()
        if os.path.exists(self.args.checkpoint_prefix):
            latest = tf.train.latest_checkpoint(self.args.output_dir)
            model.load_weights(latest)
        self.model = model

    def get_callback(self):
        callback_list = []
        if self.args.use_early_stop:
            # EarlyStopping
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, mode='max')
            callback_list.append(early_stopping)
        if self.args.checkpoint_prefix is not None:
            # 创建一个保存模型权重的回调
            cp_callback = ModelCheckpoint(filepath=self.args.checkpoint_prefix,
                                          monitor='val_accuracy',
                                          mode='max',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=2,
                                          )
            callback_list.append(cp_callback)


        if self.tensorboard_log_dir is not None:
            self.args.logger.info('Load the pre-training model')
            tensorboard_callback = TensorBoard(log_dir=self.tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        self.args.logger.info('model train ...')
        self.model.fit(x=x_train, y=y_train,
                       batch_size=self.args.batch_size,
                       epochs=self.args.num_epochs,
                       verbose=2,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self):
        checkpoint_dir = os.path.dirname((self.args.checkpoint_prefix))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.args.logger.info(f'restore model name is : {latest}')
        # 创建一个新的模型实例
        # model = self.create_model()
        # 加载以前保存的权重
        self.model.load_weights(latest)


def train():
    # 参数加载
    model = 'train'
    args = parser_opt(model)
    tf.device = check_gpus(mode=args.mode, logger=args.logger)
    # 加载训练数据
    x_train, y_train,x_val,y_val = read_data_alphamind_v2(args.train_data_dir,data_class=['train','val'])
    # 文本序列化
    tokenizer_tool = Tokenizer_tool(args,x_train, y_train,x_val,y_val)
    if not os.path.exists(args.class_2_id_dir) and not os.path.exists(args.word_2_id_dir):
        class_2_id = tokenizer_tool.create_class_2_id()
        word_2_id = tokenizer_tool.create_word_2_id()
        tokenizer_encoded_train = tokenizer_tool.keras_tokenizer(x_train)
        tokenizer_encoded_val = tokenizer_tool.keras_tokenizer(x_val)
    else:
        class_2_id, word_2_id = tokenizer_tool.load_dict_files()
        tokenizer_encoded_train = tokenizer_tool.host_tokenizer(x_train,word_2_id)
        tokenizer_encoded_val = tokenizer_tool.host_tokenizer(x_val, word_2_id)

    x_train = pad_sequences(tokenizer_encoded_train, maxlen=args.maxlen, padding='post')
    x_val = pad_sequences(tokenizer_encoded_val, maxlen=args.maxlen, padding='post')

    # 标签序列化
    y_train = tokenizer_tool.label_2_id(y_train)
    y_val = tokenizer_tool.label_2_id(y_val)
    args.logger.info(f'x_train shape:{x_train.shape} --- '
                     f'x_val shape:{x_val.shape} --- '
                     f'y_train shape:{y_train.shape} --- '
                     f'y_val shape: {y_val.shape}')
    args.update({'max_features':len(word_2_id)+1})
    args.logger.info("Vocabulary Size（词汇大小）: {:d}".format(args.max_features))
    args.logger.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))

    # 模型初始化
    model_hepler = ModelHepler(args)
    # 模型回调函数  保存模型  图
    model_hepler.get_callback()
    # 模型训练
    model_hepler.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    # 模型测试
    args.logger.info('Test...')
    x_test, y_test = read_data_alphamind_v2(args.train_data_dir, data_class=['test'])
    tokenizer_encoded_test = tokenizer_tool.host_tokenizer(x_test, word_2_id)
    x_test = pad_sequences(tokenizer_encoded_test, maxlen=args.maxlen, padding='post')
    y_test = tokenizer_tool.label_2_id(y_test)
    # result = model_hepler.model.predict(x_test)
    # args.logger.info(f'{x_test}, {type(x_test)},{result}, {type(result)}, {result.argmax()}')

    # 模型加载
    model_hepler.load_model()
    # 重新评估模型
    loss, acc = model_hepler.model.evaluate(x_test, y_test, verbose=2)
    args.logger.info("Restored model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    ''' https://blog.csdn.net/sinat_18127633/article/details/105860790 '''




    train()

# coding=utf-8
# =============================================
# @Time      : 2022-03-25 11:24
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil








# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    args.mode = os.environ.get("mode")
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.model_name = os.environ.get("network_name")
        args.train_data_dir = os.environ.get('train_data_dir')
        args.output_dir = os.environ.get("output_dir")
        args.checkpoint_prefix = os.path.join(args.output_dir, os.environ.get("model_ckpt_name"))
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        if bool(os.environ.get('clear_the_cache')) and os.path.exists(args.output_dir) and os.path.exists(args.tensorboard_dir):
            shutil.rmtree(args.output_dir)
            shutil.rmtree(args.tensorboard_dir)
        args.class_2_id_dir = os.path.join(args.output_dir, 'class_2_id.json')
        args.word_2_id_dir = os.path.join(args.output_dir, 'word_2_id_dir.json')
        args.vocab_processor_path = os.path.join(args.output_dir, "vocab")
        args.maxlen = int(os.environ.get('maxlen'))
        args.max_features = int(os.environ.get('max_features'))
        args.embedding_dims = int(os.environ.get('embedding_dims'))
        args.class_num = int(os.environ.get('class_num'))
        args.kernel_sizes = [int(i) for i in os.environ.get('kernel_sizes').split(',')]
        if os.environ.get('kernel_filter') != 'None':
            args.kernel_filter = [int(os.environ.get('kernel_filter'))]
        else:
            args.kernel_filter = []
        if os.environ.get('kernel_filters') != 'None':
            args.kernel_filters = [int(i) for i in os.environ.get('kernel_filters').split(',')]
        else:
            args.kernel_filters = []
        args.kernel_regularizer = os.environ.get('kernel_regularizer')
        args.last_activation = os.environ.get('last_activation')
        args.use_early_stop = bool(os.environ.get('use_early_stop'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.num_epochs = int(os.environ.get('num_epochs'))

        # opt.class_2_id_dir = os.path.join(opt.output_dir,'class_2_id.json')
        # opt.vocab_processor_path = os.path.join(opt.output_dir, "vocab")
        # opt.checkpoint_prefix = os.path.join(opt.output_dir, os.environ.get("model_ckpt_name"))

        # opt.max_document_length = int(os.environ.get('max_document_length'))
        # opt.dev_sample_percentage = float(os.environ.get('dev_sample_percentage'))

        # opt.allow_soft_placement = bool(os.environ.get('allow_soft_placement'))
        # opt.log_device_placement = bool(os.environ.get('log_device_placement'))
        # opt.embedding_dim = int(os.environ.get('embedding_dim'))
        # opt.num_classes = int(os.environ.get('num_classes'))
        # opt.batch_size = int(os.environ.get('batch_size'))
        # opt.filter_heights = os.environ.get('filter_heights')
        # opt.num_filters = os.environ.get('num_filters')
        # if opt.num_filters == None:
        #     opt.num_filters = False
        # opt.learning_rate = float(os.environ.get('learning_rate'))
        # opt.num_checkpoints = int(os.environ.get('num_checkpoints'))
        # opt.dropout_keep_prob = float(os.environ.get('dropout_keep_prob'))
        # opt.evaluate_every = int(os.environ.get('evaluate_every'))
        # opt.checkpoint_every = int(os.environ.get('checkpoint_every'))
        for path in [args.output_dir,args.tensorboard_dir,args.train_data_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
    elif model =='env':
        pass
    elif model == 'server':
        args.model_name = os.environ.get("network_name")
        args.train_data_dir = os.environ.get('train_data_dir')
        args.output_dir = os.environ.get("output_dir")
        args.checkpoint_prefix = os.path.join(args.output_dir, os.environ.get("model_ckpt_name"))
        args.tensorboard_dir = os.environ.get('tensorboard_dir')
        args.logger = logging.getLogger('model_log')
        args.class_2_id_dir = os.path.join(args.output_dir, 'class_2_id.json')
        args.word_2_id_dir = os.path.join(args.output_dir, 'word_2_id_dir.json')
        args.vocab_processor_path = os.path.join(args.output_dir, "vocab")
        args.maxlen = int(os.environ.get('maxlen'))
        args.max_features = int(os.environ.get('max_features'))
        args.embedding_dims = int(os.environ.get('embedding_dims'))
        args.class_num = int(os.environ.get('class_num'))
        args.kernel_sizes = [int(i) for i in os.environ.get('kernel_sizes').split(',')]
        if os.environ.get('kernel_filter') != 'None':
            args.kernel_filter = [int(os.environ.get('kernel_filter'))]
        else:
            args.kernel_filter = []
        if os.environ.get('kernel_filters') != 'None':
            args.kernel_filters = [int(i) for i in os.environ.get('kernel_filters').split(',')]
        else:
            args.kernel_filters = []
        args.kernel_regularizer = os.environ.get('kernel_regularizer')
        args.last_activation = os.environ.get('last_activation')
        args.use_early_stop = bool(os.environ.get('use_early_stop'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.num_epochs = int(os.environ.get('num_epochs'))
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    parser_opt('train')
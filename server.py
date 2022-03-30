# coding=utf-8
# =============================================
# @Time      : 2022-03-23 17:55
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import jieba
import numpy as np
from flask import Flask, jsonify, request
import os
import json
import config
from utils.parameter import *
from train import ModelHepler
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Predictor(object):
    def __init__(self, args):
        self.args = args
        # 加载word_2_id
        with open(self.args.word_2_id_dir, 'r', encoding='utf-8') as r:
            self.word_2_id = json.loads(r.read())

        # 加载class_2_id
        with open(self.args.class_2_id_dir, 'r', encoding='utf-8') as r:
            class_2_id = json.loads(r.read())
            self.id_2_class = {v:k for k,v in class_2_id.items()}

        self.args.update({'max_features': len(self.word_2_id) + 1})

        # 加载模型类
        self.model_hepler = ModelHepler(args)
        self.model_hepler.load_model()




    def predict(self,text):

        # 文本分词\序列化
        # for word in jieba.cut(text):
        #     {}.keys()
        word_list = [self.word_2_id[word] if word in self.word_2_id.keys() else self.word_2_id['UNK'] for word in jieba.cut(text)]

        x = pad_sequences([word_list], maxlen=args.maxlen, padding='post')
        result = self.model_hepler.model.predict(x)


        class_id = result.argmax()
        class_prob = result[0][class_id]
        class_name = self.id_2_class[class_id]

        return class_id, class_prob, class_name








# text = "好您稍等帮您看一下啊，嗯帮您查一下您这个套餐吗现在其实还是比较合算一百二十八啊？是您这个套餐是一个五七对吗？这个号码多少用的还是比较多的一百六十八就多了十个地方。等到六十八小时的五百分钟十个g的话一旦超出他妈的逼十五块然后加上的十五兆的一个的三块打电话说有一个一毛九然后您这个号码有个两块钱八百一十八包如果转。详单号码卡还是验证身份证号的。那我帮您转分钟的，那么多，好那转接成功呢你这个号码的话上十一号我现在这个申请的话以后它这个四月一号之前办理完成会有短信通知到您四月一号以后六十八套餐了如果真的有什么问题再给您回电如果一旦改成功？流量无法转到可以使用吗？好那祝您生活愉快再见。|你好我想改一下套餐改了六十八的。不要以后设置不了，你让我根本用不了你给我看看我用不了了。不用了不需要五g手机都不收。不用，可以。嗯，噢可以，嗯，没有。有。好谢谢不想。恩好谢谢啊再见啊。"
#
# class_id, class_prob, class_name = detector.predict(text)
# result, key_word = downshifts_user.predict_key(text)



if __name__ == '__main__':
    model = 'server'
    args = parser_opt(model)
    detector = Predictor(args)
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    @app.route('/predict', methods=['POST'])
    def predict():
        args.logger.info("基于模型：降档用户数据检测.....")
        try:
            # 参数获取
            data = request.files
            if 'input' not in data:
                return 'input not exsit', 500
            file = data['input']
            queries = file.read().decode('utf-8')
            queries = queries.replace('\n', '').replace('\r', '')
            # 参数检查
            if queries is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })

            # 直接调用预测的API

            class_id, class_prob, class_name = detector.predict(queries)
            data = [
                {
                    'text': queries,
                    'class_name': class_name,
                    'class_id': int(class_id),
                    'class_prob': float(class_prob),
                    'result': True if class_name == '降档' else False
                }
            ]
            return jsonify({
                'code': 200,
                'msg': '成功',
                'data': data
            })
        except Exception as e:
            args.logger.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!'
            })
    # 启动
    app.run(host='0.0.0.0')
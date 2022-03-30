# coding=utf-8
# =============================================
# @Time      : 2022-03-28 18:02
# @Author    : DongWei1998
# @FileName  : token_2_data.py
# @Software  : PyCharm
# =============================================
import os
import numpy as np
import jieba

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import tensorflow as tf

class Tokenizer_tool(object):
    def __init__(self,args,x_train, y_train,x_val,y_val):
        self.args = args
        self.text = []
        self.label = y_train + y_val
        self.class_2_id = {}
        self.unk_token = "[UNK]"
        for sen in x_train + x_val:
            t = []
            for word in jieba.cut(sen):
                t.append(word.strip())
            self.text.append(t)


    def create_class_2_id(self):
        self.args.logger.info('保存 label_2_id 映射！！！')
        for i,label in enumerate(set(self.label)):
            self.class_2_id[label] = i
            with open(self.args.class_2_id_dir, 'w') as cid:
                json.dump(self.class_2_id, cid)
        return self.class_2_id

    def create_word_2_id(self):
        self.args.logger.info('构建词汇表映射关系！！！')
        self.tf_keras_tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tf_keras_tokenizer.fit_on_texts(self.text)
        self.args.logger.info('保存 word_2_id 映射！！')
        self.tf_keras_tokenizer.word_index.update({'UNK':list(self.tf_keras_tokenizer.word_index.values())[-1]+1})
        with open(self.args.word_2_id_dir, 'w') as wid:
            json.dump(self.tf_keras_tokenizer.word_index, wid)
        return self.tf_keras_tokenizer.word_index

    def load_dict_files(self):
        self.args.logger.info('加载词汇表映射关系！！！')
        with open(self.args.class_2_id_dir,'r',encoding='utf-8') as cid:
            self.class_2_id = json.loads(cid.read())
        with open(self.args.word_2_id_dir, 'r', encoding='utf-8') as wid:
            self.word_2_id = json.loads(wid.read())
        return self.class_2_id,self.word_2_id

    def keras_tokenizer(self,datas):
        tokenizer_encoded = self.tf_keras_tokenizer.texts_to_sequences(datas)
        return tokenizer_encoded

    def host_tokenizer(self,datas,word_2_id):
        tokenizer_encoded = []
        for words_list in datas:
            word_id = []
            for word in words_list:
                if word in word_2_id.keys():
                    word_id.append(word_2_id[word])
                else:
                    word_id.append(word_2_id['UNK'])
            tokenizer_encoded.append(word_id)
        return tokenizer_encoded


    def label_2_id(self,label):
        label = [self.class_2_id[label] for label in label]
        return np.array(label,dtype=np.int32)






# coding=utf-8
# =============================================
# @Time      : 2022-03-28 17:05
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
import os
import numpy as np
import jieba

def read_data_alphamind(data_files):
    X = []
    Y = []
    dict_number_to_class = {}
    for f_1 in os.listdir(data_files):
        if f_1 in ['train','val']:
            f_2 = os.path.join(data_files,f_1)
            for i,label in enumerate(os.listdir(f_2)):
                if label not in dict_number_to_class.keys():
                    dict_number_to_class[label] = i
                f_3 = os.path.join(f_2,label)
                for file in os.listdir(f_3):
                    with open(os.path.join(f_3,file),'r',encoding='utf-8') as r:

                        X.append(r.read())
                        Y.append(dict_number_to_class[label])
        else:
            pass
    return np.array(X), np.array(Y), dict_number_to_class


def read_data_alphamind_v1(data_files):
    X = []
    Y = []
    dict_number_to_class = {}
    for f_1 in os.listdir(data_files):
        f_2 = os.path.join(data_files,f_1)
        for i,label in enumerate(os.listdir(f_2)):
            if label not in dict_number_to_class.keys():
                dict_number_to_class[label] = i
            f_3 = os.path.join(f_2,label)
            for file in os.listdir(f_3):
                with open(os.path.join(f_3,file),'r',encoding='utf-8') as r:
                    x = [word.strip() for word in jieba.cut(r.read())]
                    X.append(x)
                    Y.append(dict_number_to_class[label])
        else:
            pass
    return np.array(X,dtype=type(X)), np.array(Y,dtype=type(Y)), dict_number_to_class


def read_data_alphamind_v2(data_files,data_class=[]):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    for f_1 in os.listdir(data_files):
        if f_1 in data_class and f_1 == 'train':
            f_2 = os.path.join(data_files,f_1)
            for i,label in enumerate(os.listdir(f_2)):
                f_3 = os.path.join(f_2,label)
                for file in os.listdir(f_3):
                    with open(os.path.join(f_3,file),'r',encoding='utf-8') as r:
                        X_train.append(r.read())
                        Y_train.append(label)
        elif f_1 in data_class and f_1 == 'val':
            f_2 = os.path.join(data_files, f_1)
            for i, label in enumerate(os.listdir(f_2)):
                f_3 = os.path.join(f_2, label)
                for file in os.listdir(f_3):
                    with open(os.path.join(f_3, file), 'r', encoding='utf-8') as r:
                        X_val.append(r.read())
                        Y_val.append(label)
        elif f_1 in data_class and f_1 == 'test':
            X_test = []
            Y_test = []
            f_2 = os.path.join(data_files, f_1)
            for i, label in enumerate(os.listdir(f_2)):
                f_3 = os.path.join(f_2, label)
                for file in os.listdir(f_3):
                    with open(os.path.join(f_3, file), 'r', encoding='utf-8') as r:
                        X_test.append(r.read())
                        Y_test.append(label)
            return X_test, Y_test
        else:
            pass
    return X_train,Y_train,X_val,Y_val
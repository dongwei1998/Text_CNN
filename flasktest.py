# coding=utf-8
# =============================================
# @Time      : 2022-03-23 17:55
# @Author    : DongWei1998
# @FileName  : flasktest.py
# @Software  : PyCharm
# =============================================
import time
import requests
import sys
import os
from tqdm import tqdm
from glob import glob

def pic_post(text):
    postdata = {
        "input":(r'flasktest.txt', text)
                }

    r = requests.post('http://10.19.234.179:5000/predict', files=postdata)
    print(r.json())


if __name__ == '__main__':

    # file_path = sys.argv[1]
    file_path = './flasktext.txt'
    file_path = [file_path] if os.path.isfile(file_path) else glob(f"{file_path}/*")

    for i,file in enumerate(tqdm(file_path)):
        text = open(file, "rb")
        results = pic_post(text)
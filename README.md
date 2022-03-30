# Text_CNN算法

目前为初版 一些细节需要自己添加 如  停用出  数据预处理等

导包：pipreqs . --encoding=utf8 --force  

##conda 第三方包下载

 conda 命令： conda create --name tf26_gpu --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

 pip   命令：pip install tensorflow-gpu -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

## 数据格式

### 数据集导入格式
```
文本分类
├── 类别1 文件夹
│   ├── 1.txt
│   └── 2.txt
├── 类别2 文件夹
│   ├── 3.txt
│   └── 4.txt
├── 5.txt # 类别文件夹外的，当作未标注
├── 6.txt
```

### 数据集导出格式
```
文本分类
├── test 文件夹
│   ├── 类别1
│   │   ├── 1.txt
│   │   └── 2.txt
│   ├── 类别2
│   │   ├── 3.txt
│   │   └── 4.txt
├── train 文件夹
│   ├── 类别1
│   │   ├── 5.txt
│   │   └── 6.txt
│   ├── 类别2
│   │   ├── 7.txt
│   │   └── 8.txt
└── val 文件夹
    ├── 类别1
    │   ├── 9.txt
    │   └── 10.txt
    └── 类别2
        ├── 11.txt
        └── 12.txt

```

### Dockerfile 构建 （后续更新）
  

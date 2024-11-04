环境
tensorflow 1.14
sklearn

建议用anaconda 搭建虚拟环境

将原始的fit文件放至1_cae_noise/cae_noise/文件夹中，命名为fit_raw，按照README进行数据转换和降噪处理

在2_cae_encode/中按照README对降噪图像进行无监督聚类，得到编码数据

在3_cae_class/中按照README对编码数据进行投票分类，得到有标签结果和丢弃数据，对结果进行目视分类检验，若效果可以则运行make_dataset.py，得到打好标签的数据集输出dataset/


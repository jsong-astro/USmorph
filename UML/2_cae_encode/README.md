0    配置config.py   对模型参数进行配置，设置图像尺寸、卷积核大小参数(若效果不好可以调大，如[3, 3, 1, 16]改为[5, 5, 1, 16])，若其中的the_dim设置过大，请适当降低维度（如缩小一倍）。若输入是极坐标图像，建议将尺寸设置为pix_x=128,pix_y=56。

1    python make_pkl.py  将目标图像转化为pickle格式的txt文件,cae_img/文件夹目录下为降噪后图像，pkl_img/pkl_img/目录下输出所有txt文件（两个压缩包）。

2    python MAIN.py --train True   训练模型进行自编码，支持断点续接。

3    python MAIN.py --encode True  得到输出的encode_result文件，其中含有编码数据。

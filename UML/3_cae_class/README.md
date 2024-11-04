1 配置config.py   设置聚类的类别总数

2 python main.py    在result文件夹中得到多模型聚类的结果

3 python get_final_result.py  生成result_final/，其中为混合聚类的最终结果(编码数据)

4 python readimg_fits.py  将result_final/中的编码数据转换为png图像，并输出至result_img/

5 对最终聚类结果（result_img/）进行目视分类，得到result.txt

6 执行:python make_dataset.py，得到打好标签的数据集并输出至dataset/。其中，图片名的首位为类别(0:UNC,1:IRR,2:LTD,3:ETD,4:SPH)。


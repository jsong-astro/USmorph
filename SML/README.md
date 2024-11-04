各部分文件说明如下：
		    checkpoint/：储存有监督模型的参数数据。
			no_labels_images/: 储存无标签数据的文件夹
			no_labels_images_175/: 储存极坐标展开处理后的无标签数据的文件夹
			log/train_log1.txt:每步的训练准确率
			log/train_log2.txt:每轮的测试集准确率
			log/train_log3.txt:每100步的训练集准确率
			log/train_log4.txt:每100步的测试集准确率

0  在conf/global_settings.py中进行输入数据的尺寸设置、权重储存的间隔轮数、测试的间隔步数（默认值是100，即每100步测试一次准确率）数据集地址、测试集地址

1  模型训练，执行命令：python train.py -net 模型名称(如googlenet) -gpu(若设备无gpu不写即可，默认为cpu) -b 32(每批的遍历数据量，可根据服务器性能调整，调小可减小使用内存,值为2的倍数)
	使用例子如下：
	输入：
		python train.py -net googlenet -gpu  -b 32



2  模型测试，执行命令：python test.py -net 模型名称(googlenet/densenet121/attention56) -weight checkpoint/中的参数数据文件地址 -b 1000(每批的遍历数据量，可根据服务器性能调整，调小可减小使用内存)
		使用例子如下：
		输入：
			python test.py -net googlenet -weight checkpoint/googlenet-426-best.pth  -b 1000
		输出(模型在验证集中的分类准确率)：
			(2900, 1, 28, 175) data_test处理后的形状
			[tensor(954), tensor(949), tensor(856)] ================== 2759 每批(1000)遍历中分类正确的文件数以及分类正确的文件总数
			{'3_2': 42, '1_2': 29, '2_3': 28, '3_4': 9, '2_1': 17, '4_3': 12, '1_3': 1, '0_1': 1, '4_1': 1, '1_0': 1} 说明：总体的错误分类情况，格式为预测标签+'_'+真实标签+':'+该预测标签与真实标签不符的数据量，例如:'3_2'：42表示验证集中属于类别2但被判断为类别3的图像有42张
			Evaluating Network.....
			Test set:Accuracy: 0.9514 

3  训练完成后，使用模型储存权重对无标签图像进行分类：
	执行命令：
	python getlabel.py -net 模型名称(googlenet/densenet121/attention56) -weight checkpoint/中的参数数据文件地址 -b 1000(每批标签标注的遍历数据量，可根据服务器性能调整，调小可减小使用内存)


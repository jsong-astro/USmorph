import numpy as np
import the_net as m
import utils
import random
import logging
import config as CF
import tensorflow as tf
import time
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.description='训练还是生成bp'
parser.add_argument("--train", default=False,help="是否训练",type=bool)
parser.add_argument("--savePB", default=False,help="我是B",type=bool)
parser.add_argument("--test", default=False,help="我是B",type=bool)
parser.add_argument("--encode", default=False,help="我是B",type=bool)
args = parser.parse_args()
is_train=args.train
is_savePB=args.savePB
is_test=args.test
is_encode=args.encode
if not is_train and not is_savePB and not is_test and not is_encode :
   print("请使用 python main.py --train True,或者 python main.py --savePB True")
   exit()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m_%d %H:%M:%S",
                    filename=CF.logging_name,
                    filemode="w")
sess=tf.Session()
model=m.CAE(sess=sess,logging=logging)
#model_1.print_var()
Data=utils.data_utils()


if is_train:
    st=time.time()
    for k in range(CF.maxstep):
        batch=Data.give_batch(CF.batch_size)
        #print(np.array(batch).shape)
        #batch=Data.fake_batch()
        is_show=True if (k+1)%CF.step_show==0 else False
        is_save=True if (k+1)%CF.step_save==0 else False
        if is_show:
            print("用时%s秒"%(time.time()-st))
            st=time.time()
        model.train(batch, is_show,is_save)
 
if is_encode:
    lenth = Data.give_len()
    with open("encode_result","w",encoding="utf-8")as f:
        for _ in range(1,10000):
            print("正在编码第%s批数据"%_)
            if _==int(lenth/CF.batch_size):
                batch,label=Data.give_batch_label(int(lenth%CF.batch_size))
            else:
                batch,label=Data.give_batch_label(CF.batch_size)
            encode = model.encode(batch)
            #print(batch[:10])
            #print(encode[:10])
            #exit()
            for la,en in zip(label,encode):
                en="<=>".join([str(x)for x in en])
                f.write("%s\t%s\n"%(la,en))
            if _==int(lenth/CF.batch_size):
                break

if is_test:
    batch=Data.give_batch(20)
    model.test(batch)
               
if is_savePB:
    model.savePB()

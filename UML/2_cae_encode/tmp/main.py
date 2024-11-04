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
import pickle
parser = argparse.ArgumentParser()
parser.description='训练还是生成bp'
parser.add_argument("--train", default=False,help="是否训练",type=bool)
parser.add_argument("--savePB", default=False,help="我是B",type=bool)
parser.add_argument("--reconstruct",default=False,help="re",type=bool)
#parser.add_argument("--savePB", default=False,help="我是B",type=bool)
args = parser.parse_args()
is_train=args.train
is_savePB=args.savePB
reconstruct=args.reconstruct
if not is_train and not is_savePB and not reconstruct:
   print("请使用 python main.py --train True,或者 python main.py --savePB True")
   exit()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m_%d %H:%M:%S",
                    filename=CF.logging_name,
                    filemode="w")
sess=tf.Session()
model=m.CAE(sess=sess,logging=logging)
#model_1.print_var()
Data=utils.data_utils()

def to_lst(ar):
    ret=[]
    for _ in ar:
        tmp=[]
        for __ in _:
            tmp.append(list(__))
        ret.append(tmp)
    return ret
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
        
if is_savePB:
    model.savePB()
if reconstruct:
    path_all=[]
    batch_all=[]
    rec_all=[]
    for _ in range(1000):
        print(_)
        path,batch=Data.give_batch_reconstruct(128)
        try:
            rec=model.sess.run(model.reconstruction,feed_dict={model.input:batch})
            rec=np.reshape(np.array(rec),[-1,28,28])
            rec=to_lst(rec)
            path_all +=path
            batch_all +=batch
            rec_all +=rec
        except Exception as e:
            print(e)
            break
    print(len(path_all),len(batch_all),len(rec_all))
    the_data=[path_all,batch_all,rec_all]
    with open("reconstruct","wb")as f:
        pickle.dump(the_data,f)
        #for ll,ba_re in enumerate(zip(batch,rec)):
        #    ba,re=ba_re
       #     ba=np.array(ba)
         #   re=np.array(re)
          #  re=np.reshape(re,[28,28])
           # re=list(re)    

            #f.write("%s\t%s\n"%(json.dumps(ba),json.dumps(re)))
        

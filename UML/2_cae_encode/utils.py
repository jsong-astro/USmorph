#!-*-coding:utf-8-*-
# 获得图片的label2path

import glob
import pickle
import time 
import random
from PIL import Image
import numpy as np
import config as C


class data_utils():
    def __init__(self):
        self.check()
        self.x=C.pix_x
        self.y=C.pix_y
        self.get_label2path()
        self.jth_batch=0
    def check(self):
        if glob.glob("pkl_img")==[]:
            print("原始文件缺失，请阅读README.md")
            exit()
        label_file_lst=glob.glob("pkl_img/*")
        for label_file in label_file_lst:
            if len(glob.glob(label_file+"/*"))==0:
                print("%s 这个类别里面没有图片，请正确设置原始图片"%label_file)
                exit()
        self.label_file_lst=label_file_lst

    def get_label2path(self):    
        st=time.time()
        label_file_lst=self.label_file_lst
    
        t={}
        pool=[]
        total_file =0
        for label_file in label_file_lst:
            label=label_file.split("pkl_img/")[-1]
            t[label]=[]
            for file_ in glob.glob(label_file+"/*"):
                t[label].append(file_)
                pool.append(file_)
                total_file +=1
        print("一共有%s个类别，共计%s个样本，其中"%(len(t),total_file))
        for label,contents in t.items():
            print("类别%s 有%s 个样本"%(label,len(contents)))
    
        with open("label2path.pkl","wb")as f:
            pickle.dump(t,f)
        print("label2path.pkl 文件成功保存，共计耗时%s秒"%(time.time()-st))
        self.label2path=t
        self.pool=pool
        random.shuffle(self.pool)
        self.total_file=total_file

    def give_batch_label(self,batch_size):
        try:
            path_lst=self.pool[self.jth_batch*batch_size:(self.jth_batch+1)*batch_size]
            self.jth_batch +=1
            ret=self.give_feature(path_lst)
        except Exception as e:
            print("wrong!")
            exit()
        return ret,path_lst
        
    def give_len(self):
        return self.total_file

    def give_batch(self,batch_size,shuffle=False):
        if shuffle or self.jth_batch*batch_size>self.total_file:
            random.shuffle(self.pool)
            print("---------------------用完了，开始重新洗数据-----------------")
            self.jth_batch=0
        path_lst=self.pool[self.jth_batch*batch_size:(self.jth_batch+1)*batch_size]
        #print(path_lst)
        #exit()
        self.jth_batch +=1
        ret=self.give_feature(path_lst)
        return ret
    def give_batch_reconstruct(self,batch_size,shuffle=False):
        if shuffle or self.jth_batch*batch_size>self.total_file:
            random.shuffle(self.pool)
            print("---------------------用完了，开始重新洗数据-----------------")
            self.jth_batch=0
            return [],[]
        path_lst=self.pool[self.jth_batch*batch_size:(self.jth_batch+1)*batch_size]
        #print(path_lst)
        #exit()
        self.jth_batch +=1
        ret=self.give_feature(path_lst)
        return path_lst,ret
        
    def array2list(self,array):
        ret=[]
        for ar in array:
            #print(ar)
            #exit()
            tmp=[x for x in ar]
            ret.append(tmp)
        return ret



    def norm(self,data,max_,min_):
        ma = np.max(data)
        mi = np.min(data)
        ret = (data-mi)/(ma-mi)
        ret=ret*(max_)
        ret=ret.astype(int)
        #ret[ret > max_] = max_
        #ret[ret < min_] = min_
        #ret=ret -min_

        return ret
 


    def norm_(self,data,max_,min_):
        mean = np.mean(data)
        std = np.std(data)
        ret = (data-mean)/std
        ret=ret*(max_/3)
        ret=ret.astype(int)
        #ret[ret > max_] = max_
        ret[ret < min_] = min_
        ret=ret -min_

        return ret
    
    def give_feature(self,path_lst):
        features=[]
        for path_ in path_lst:
            try:
                with open(path_,"rb")as f:#r时以只读方式打开文件，文件的指针回放在文件的开头
                    img  = pickle.load(f)
                    img=self.norm(img,500,-500)#img*100
                img_=self.array2list(np.array(img))
                #print(img_)
                #exit()
                features.append(img_)
            except Exception as e:
                print(e)
                print("这张图片%s 有问题，跳过处理"%path_)
                continue
        #print(np.array(features).shape)
        #print(features[:1])
        return features
    def fake_batch(self,shape=[16,self.x,self.y]):
        return np.ones(shape)
if __name__=="__main__":
    Data=data_utils()
    path,batch=Data.give_batch_reconstruct(100)
    print(path)
    print(len(batch[1]),len(batch[1][0]))
    print(np.array(batch).shape)#N*x*y
    #print(np.array(batch).shape)

















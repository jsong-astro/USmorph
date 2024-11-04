import pickle
import os,glob
import numpy as np
import PIL.Image as I
import config as C


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def make_pkl(path,savepath):
    k = 0
    filelist = glob.glob(path+'/*')
    for i in filelist:
        img = I.open(i).convert('L')
        #print(img.size,'==1==')
        img = img.resize((C.pix_y,C.pix_x)) #图片的长和宽
        img = np.array(img)
        #print(img.shape,'==2==')
        #exit()
        savepath = make_path(savepath)
        fn = savepath+i.split('/')[-1].split('.png')[0]+'.txt'
        with open(fn,'wb') as f:
            pickle.dump(img,f)
            f.close()
        k+=1
        print('======%s======'%str(k))


path = './1_cae_class/0_cae_noise/cae_noise/cae_img'
savepath = 'pkl_img/pkl_img/'

make_pkl(path,savepath)


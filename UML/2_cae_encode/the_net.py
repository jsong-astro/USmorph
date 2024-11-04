import tensorflow as tf
import numpy as np
from models import *
import config as C
import time

class CAE(object):
    def __init__(self,sess,logging):
        
        self.sess=sess
        self.logging=logging
        self.encode_structure=C.encode_structure
        self.global_step=tf.Variable(0,trainable=False)
        self._checkpoint_path=C.checkpoint_path
        self.buid_input()
        self.build_encode_decode()
        
        self.print_var()
        self.loggingAll()
        self._saver=tf.train.Saver(tf.global_variables(),max_to_keep=1)
        self.initialize()
        
    
    def buid_input(self):
        self.input = tf.placeholder(tf.float32, shape=[None, C.pix_x, C.pix_y])  # [#batch, img_height, img_width, #channels]
        self.input_expand=tf.expand_dims(self.input,-1)
    
    def print_var(self):
        for item in dir(self):
            type_string=str(type(getattr(self,item)))
            print(item,type_string)
    def encode(self,batch):
        feed_dict={self.input:batch}
        encode=self.sess.run(self.encoded,feed_dict=feed_dict)
        return encode
    def build_encode_decode(self):
        # encode
        self.shapes_conv=[]
        self.shapes_pool=[]
        for ii,stru in enumerate(C.encode_structure):
            if ii==0:
                self.conv_pool = Convolution2D(stru["cov"], strides=stru["cov_stride"],padding='SAME',activation=tf.nn.relu, scope='conv_%s'%ii)(self.input_expand)
                #self.shapes_conv.append(self.conv_pool.get_shape().as_list())
                print("conv0",self.conv_pool.get_shape().as_list())
                self.shapes_conv.append(tf.shape(self.conv_pool))
                self.conv_pool = MaxPooling(kernel_shape=stru["pool"], strides=stru["pool_stride"], padding='SAME', scope='pool_%s'%ii)(self.conv_pool)
                #self.shapes_pool.append(self.conv_pool.get_shape().as_list())
                self.shapes_pool.append(tf.shape(self.conv_pool))
                print(self.conv_pool.get_shape().as_list())
            else: 
                self.conv_pool = Convolution2D(stru["cov"], strides=stru["cov_stride"],padding='SAME',activation=tf.nn.relu, scope='conv_%s'%ii)(self.conv_pool)
                #self.shapes_conv.append(self.conv_pool.get_shape().as_list())
                print(self.conv_pool.get_shape().as_list())
                self.shapes_conv.append(tf.shape(self.conv_pool))
                self.conv_pool = MaxPooling(kernel_shape=stru["pool"], strides=stru["pool_stride"], padding='SAME', scope='pool_%s'%ii)(self.conv_pool)
                #self.shapes_pool.append(self.conv_pool.get_shape().as_list())
                self.shapes_pool.append(tf.shape(self.conv_pool))
                print(self.conv_pool.get_shape().as_list())
        self.last_pool_shape=self.conv_pool.get_shape().as_list()
        self.unfold = Unfold(scope='unfold')(self.conv_pool)
        print(self.unfold.get_shape().as_list())
        self.shape_unfold=self.unfold.get_shape().as_list()
        self.encoded = FullyConnected(C.the_dim, activation=tf.nn.relu, scope='encode')(self.unfold)
        print(self.encoded.get_shape().as_list())
        self.shapes_conv.reverse()
        self.shapes_pool.reverse()
        self.encode_structure.reverse()
        
        #print()
        # decode
        self.decoded = FullyConnected(self.shape_unfold[1], activation=tf.nn.relu, scope='decode')(self.encoded)
        print("decode",self.decoded.get_shape().as_list())
        #print()
        
        self.fold = Fold([-1]+self.last_pool_shape[1:], scope='fold')(self.decoded)
        print("fold",self.fold.get_shape().as_list())
        for ll, pool_shape_conv_shape_structure in enumerate(zip(self.shapes_pool,self.shapes_conv,self.encode_structure)):
            pool_shape,conv_shape,structure=pool_shape_conv_shape_structure
            if ll==0:
                up_shape=(structure["pool_stride"][1],structure["pool_stride"][2])
                
                self.unpool_deconv = UnPooling(up_shape, output_shape=conv_shape, scope='unpool_%s'%ll)(self.fold)
                print(self.unpool_deconv.get_shape().as_list())
                self.unpool_deconv = DeConvolution2D(structure["cov"], output_shape=self.shapes_pool[ll+1], activation=tf.nn.relu, scope='deconv_%s'%ll)(self.unpool_deconv)
                print(self.unpool_deconv.get_shape().as_list())
            elif ll==len(self.encode_structure)-2:
                up_shape=(structure["pool_stride"][1],structure["pool_stride"][2])
                self.unpool_deconv = UnPooling(up_shape, output_shape=conv_shape, scope='unpool_%s'%ll)(self.unpool_deconv)
                print(self.unpool_deconv.get_shape().as_list())
                self.unpool_deconv = DeConvolution2D(structure["cov"], output_shape=self.shapes_pool[ll+1], activation=tf.nn.relu, scope='deconv_%s'%ll)(self.unpool_deconv)
                print(self.unpool_deconv.get_shape().as_list())
            elif ll==len(self.encode_structure)-1:
                up_shape=(structure["pool_stride"][1],structure["pool_stride"][2])
                self.unpool_deconv = UnPooling(up_shape, output_shape=conv_shape, scope='unpool_%s'%ll)(self.unpool_deconv)
                print(self.unpool_deconv.get_shape().as_list())
                self.reconstruction = DeConvolution2D(structure["cov"], output_shape=tf.shape(self.input_expand), activation=tf.nn.relu, scope='deconv_%s'%ll)(self.unpool_deconv)
                print(self.reconstruction.get_shape().as_list())
 
        # loss function
        
        print(self.input_expand.get_shape().as_list())
        print(self.reconstruction.get_shape().as_list())
        #exit()
        self.loss = tf.nn.l2_loss(self.input_expand - self.reconstruction)  # L2 loss
        
        #inb=np.ones(shape)
        #self.sess.run(tf.global_variables_initializer())
        #loss=self.sess.run(self.loss,feed_dict={self.input:np.ones([16,28,28])})
        #print(loss)
        #exit()
        
        
        print("loss",self.loss.get_shape().as_list())
        # training
        self.training = tf.train.AdamOptimizer(C.learn_rate).minimize(self.loss,global_step=self.global_step)
        
    def loggingAll(self):
        for name in dir(self):
            if name.find("_")==0 and name.find("_")==-1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))
    
    def savePB(self):
        print ("保存BP",self.encoded)
        name_pb=self.encoded.name.split(":")[0]
        if self.readCKPT():
            output_node_names = [name_pb]
            output_graph_def = tf.graph_util.convert_variables_to_constants(self.sess,self.sess.graph_def,output_node_names=output_node_names)
            with tf.gfile.FastGFile("model.pb",mode="wb") as f:
                f.write(output_graph_def.SerializeToString())
            self.logging.info("pb 文件被保存了")
        else:
            self.logging.warn("保存出错，没有CKPT文件夹")
    
    def readCKPT(self):
        self.ckpt = tf.train.get_checkpoint_state(self._checkpoint_path)
        if self.ckpt:
            self.logging.info("reading training record from '%s'"%self.ckpt.model_checkpoint_path)
            self._saver.restore(self.sess,self.ckpt.model_checkpoint_path)
            return True
        return False  
        
    def initialize(self):
        print ("now initialize")
        if not self.readCKPT():
            print ("no ckpt")
            self.sess.run(tf.global_variables_initializer())
    def train(self, batch_in, is_show=False,is_save=False):
        
        feed_dict={self.input:batch_in}
        #loss=self.sess.run(self.loss,feed_dict=feed_dict)
        #print(loss)
        _,loss,global_step=self.sess.run([self.training,self.loss,self.global_step],feed_dict=feed_dict)
        if is_show:
            print("loss is %s,global_step is %s, %s"%(loss,global_step,loss/(C.batch_size*C.pix_x*C.pix_y)))
            self.logging.info("loss is %s,global_step is %s"%(loss,global_step))
            
            
        if is_save:
            print("保存模型")
            self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)

if __name__ == '__main__':
    main()

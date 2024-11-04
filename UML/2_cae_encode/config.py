
pix_x=128 #图像宽
pix_y=56  #图像长    
the_dim=pix_x*pix_y
learn_rate=0.00001
checkpoint_path="ckpt/"
logging_name="log_cae"
maxstep=5000000
batch_size=128
step_show=10
step_save=50

encode_structure=[{"cov":[3, 3, 1, 16],#first_layer，卷积核尺寸
                  "cov_stride":[1,1,1,1],#最好不改
                  "cov_padding":"SAME",#最好不改
                  "pool":[1, 2, 2, 1],#最好不改
                  "pool_stride":[1, 2, 2, 1],#最好不改
                  "pool_padding":"SAME",
                  },
                  {"cov":[3, 3, 16, 16],#second_layer，卷积核尺寸
                  "cov_stride":[1,1,1,1],
                  "cov_padding":"SAME",
                  "pool":[1, 2, 2, 1],#最好不改
                  "pool_stride":[1, 2, 2, 1],#最好不改
                  "pool_padding":"SAME",
                  },
                  ]    
structure=[{"encode":[]
}]


















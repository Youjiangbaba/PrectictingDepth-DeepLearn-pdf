#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
这个文件，输入图像，进行深度图估计、深度信息分层、dibr
'''
import argparse
import os
import numpy as np

import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import NYU_FCRNmodels as models
import cv2

import time
from numba import jit             # numba jit技术实时编译

import define_size as size_


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('in_photo', help='输入图像')
parser.add_argument('Depth_lever', help='Depth lever to input(1 to 99)')
args = parser.parse_args()

size_img = cv2.imread(args.in_photo)
size_.size_int(size_img,1)

my_width = size_.w
my_height =  size_.h

print(my_width,my_height)

import my_dibr as dibr

def start_run():

	#Depth_lever 
	steps = 5
	if len(args.Depth_lever) == 1:
		steps = ord(args.Depth_lever[0]) - ord('0')
	elif len(args.Depth_lever) == 2:		
		steps = (ord(args.Depth_lever[0]) - ord('0'))*10 + ord(args.Depth_lever[1]) - ord('0')
	


	# Default input size
	width = my_width
	height = my_height
	channels = 3
	batch_size = 1

	# Create a placeholder for the input image
    #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存
	input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
	net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
	#gpu memory set
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7


	with tf.Session(config=config,) as sess:
		starttime = time.time()
        # Load the converted parameters
		print('Loading the model')

        # Use to load from ckpt file
		saver = tf.train.Saver()     
		saver.restore(sess, '/home/jiang/桌面/tensorflow_old/old2.10/NYU_FCRN.ckpt')

		print ("load time:",time.time()-starttime)

		source_img = cv2.imread(args.in_photo)
		#deal image	
		img = 	Image.fromarray(cv2.cvtColor(source_img,cv2.COLOR_BGR2RGB))             #opencv 转 PIL.Iamge
		source_img = cv2.resize(source_img,(my_width,my_height))

		img = img.resize([width,height], Image.ANTIALIAS)       #缩放
		img = np.array(img).astype('float32')                   #数据类型转换
		img = np.expand_dims(np.asarray(img), axis = 0)         #表示在0位置添加数据					
			

       	# Evalute the network for the given image
		predicttime = time.time()
		pred = sess.run(net.get_output(), feed_dict={input_node: img})

		plt.imshow(pred[0,:,:,0])
		plt.show()

		img_depth = cv2.cvtColor(np.asarray(pred[0,:,:,0]),cv2.COLOR_RGB2BGR)
		img_depth = cv2.cvtColor(img_depth,cv2.COLOR_BGR2GRAY)
		max_depth = np.max(img_depth)
		min_depth = np.min(img_depth)
		img_depth = (np.array(img_depth) - min_depth)/(max_depth - min_depth)   #归一化，归为０－１之间
		# img_depth = np.array(img_depth)*0.2
		#cv2.waitKey(0)

		#img_depth = cv2.equalizeHist(img_depth)		        #直方图均衡化  8u1		
		#cv2.imshow("depth img",img_depth)       
		img_depth = cv2.resize(img_depth,(my_width,my_height)) 
		cv2.imshow("deep img",img_depth)  

		save_img = cv2.resize(source_img,(source_img.shape[1]*2,source_img.shape[0]))
		save_d = cv2.cvtColor(img_depth*255,cv2.COLOR_GRAY2BGR)
		save_img[:,0:source_img.shape[1]] = source_img.copy()
		save_img[:,source_img.shape[1]:save_img.shape[1]] = save_d.copy()
		cv2.imwrite("test_rgb-d/"+args.in_photo+"fcrn.jpg",save_img)			##保存深度图
		cv2.imwrite("test_rgb-d/"+args.in_photo+"d.jpg",save_d)			##保存深度图
		
		#img_depth = np.array(img_depth)*255               #直接生成的0 - 1，显示为灰度图


		#dibr deal
		# #sbs_img,left,right = dibr.run_myDibr(source_img,img_depth,steps)	
		# sbs_img,imgdd,left,right = dibr.myDibr_dd(source_img,img_depth,steps)
		# cv2.imshow("sbs img",sbs_img)
		# img3d = dibr.Merge3D(left,right)
		# cv2.imshow("3d img",img3d)
		# imgdd = np.array(imgdd)/255
		# cv2.imshow("dd img",imgdd)
		


		#cv2.imwrite("depth/"+str(frame_count)+".jpg",img_depth)			##保存深度图 ---------------------------------------------save
		#cv2.imwrite("cap_sbs/"+str(frame_count)+".jpg",sbs_img)				#保存ｓｂｓ

		# #记录：如果imread读取的灰度图，显示为０－２５５的；如果是处理过后再转换的灰度图，显示为0-1,保存时 *255才是灰度图
		# kkk = cv2.imread("/home/jiang/图片/2.png")
		# kkk = cv2.cvtColor(kkk,cv2.COLOR_BGR2GRAY)
		# print (kkk)              #读取灰度图读出来是０－２５５
		# cv2.imshow("kkk",kkk)


		
		cv2.waitKey(0)

	cv2.destroyAllWindows()
	return 0
	

if __name__ == '__main__':
	start_run()

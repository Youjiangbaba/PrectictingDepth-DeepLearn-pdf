# -*- coding: utf-8 -*-
#!/usr/bin/env python

import argparse
import torch
import torch.nn.parallel

from torchvision import transforms, utils

from pytorch_models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb
import model_path as pm

import matplotlib.image
import matplotlib.pyplot as plt


import cv2
from PIL import Image


plt.set_cmap("jet")

import define_size as size_


parser = argparse.ArgumentParser()
parser.add_argument('Photo_string', help='图片')
parser.add_argument('Depth_lever', help='Depth lever to input(1 to 99)')
args = parser.parse_args()

size_img = cv2.imread(args.Photo_string)
size_.size_int(size_img,1)

my_width = 640
my_height =  480

import my_dibr as dibr


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():



    imggg = cv2.imread(args.Photo_string)
    imggg = cv2.resize(imggg,(my_width,my_height))
	#Depth_lever 
    steps = 10
    if len(args.Depth_lever) == 1:
        steps = ord(args.Depth_lever[0]) - ord('0')
    elif len(args.Depth_lever) == 2:		
        steps = (ord(args.Depth_lever[0]) - ord('0'))*10 + ord(args.Depth_lever[1]) - ord('0')
    print(steps)

    model_name = 'senet'
    if  model_name == 'resnet':
		model = define_model(is_resnet=True, is_densenet=False, is_senet=False)
		model = torch.nn.DataParallel(model).cuda()
		model.load_state_dict(torch.load(pm.mum_path+'/pretrained_model/model_resnet'))
		model.eval()
    elif  model_name == 'densenet':
		model = define_model(is_resnet=False, is_densenet=True, is_senet=False)
		model = torch.nn.DataParallel(model).cuda()
		model.load_state_dict(torch.load(pm.mum_path+'/pretrained_model/model_densenet'))
		model.eval()
    elif  model_name == 'senet':
		model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
		model = torch.nn.DataParallel(model).cuda()
		model.load_state_dict(torch.load(pm.mum_path+'/pretrained_model/model_senet'))
		model.eval()
	
    print ("model loaded!")

    nyu2_loader = loaddata.readNyu2(imggg)
    for  i,image in enumerate(nyu2_loader):

		with torch.no_grad():                  #释放内存
		    image = torch.autograd.Variable(image, volatile=True).cuda()
		    out = model(image)
		    mid_img = out.view(out.size(2),out.size(3)).data.cpu().numpy()

		    # plt.imshow(mid_img)
		    # plt.show()

		    #outt = np.array(mid_img)/np.max(mid_img) - np.min(mid_img)/np.max(mid_img)
		    outt = (np.array(mid_img) - np.min(mid_img))/(np.max(mid_img) - np.min(mid_img))
		    #print outt
		    outt = cv2.resize(outt,(imggg.shape[1],imggg.shape[0]))
			#print outt.shape,outt
		    #cv2.imshow("depth",outt)
		    
		    # save_img = cv2.resize(imggg,(imggg.shape[1]*2,imggg.shape[0]))
		    # save_d = cv2.cvtColor(outt*255,cv2.COLOR_GRAY2BGR)
		    # save_img[:,0:imggg.shape[1]] = imggg.copy()
		    # save_img[:,imggg.shape[1]:save_img.shape[1]] = save_d.copy()
		    # cv2.imwrite("test_rgb-d/senet"+args.Photo_string+".jpg",save_img)			##保存深度图
		    cv2.imwrite("test_rgb-d/d-senet"+args.Photo_string+".jpg",outt*255)			##保存深度图


    # # #dibr处理
    # #outtt = cv2.imread("test_rgb-d/d-senet"+args.Photo_string+".jpg",0) 
    # frame = cv2.imread(args.Photo_string) 
    # frame = cv2.resize(frame,(my_width,my_height))
    # imgdd, steps = dibr.depth_gradation(outt*255)
    # cv2.imshow("d",imgdd/255)
    # left,right,sbs_img = dibr.segementation(frame,imgdd,steps)
    # rb_3d = dibr.Merge3D(left,right)
    # #cv2.imshow("0",frame)
    # cv2.imshow("left",left)
    # cv2.imshow("right",right)
    # cv2.imshow("d",outt)

    # cv2.imwrite(args.Photo_string+"dpth.jpg",outt*255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''
使用torch.autograd.Variable（）将张量转换为计算图中的节点。

    使用x.data访问其值。
    使用x.grad访问其渐变。

'''
def test(nyu2_loader, model):
	for i, image in enumerate(nyu2_loader):   
  
		image = torch.autograd.Variable(image, volatile=True).cuda()
		#print "the node's data is the tensor:", image.data.size()
		#print "the node's gradient is empty at creation:", image.grad # the grad is empty right now


		#cv2.namedWindow("OpenCV", cv2.WINDOW_NORMAL)
		#img = cv2.cvtColor(np.asarray(image.view(image.size(2),image.size(3))),cv2.COLOR_RGB2BGR)
		#cv2.imshow("OpenCV",img)
		
		out = model(image)

		
		mid_img = out.view(out.size(2),out.size(3)).data.cpu().numpy()
		#print mid_img
		plt.imshow(mid_img)
		plt.show()
		matplotlib.image.imsave('data/out5.png', mid_img)
		outt = np.array(mid_img)/np.max(mid_img) - np.min(mid_img)/np.max(mid_img)
		print outt
		outt = cv2.resize(outt,(960,1080))
		#print outt.shape,outt
		cv2.imshow("f",outt)
		#cv2.imwrite("depth_imgs/"+str(i)+".jpg",outt*255)
		kkk = cv2.waitKey(1)
		if kkk == 27:
			break
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
    main()	

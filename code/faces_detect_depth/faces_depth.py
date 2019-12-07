#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
这个文件，输入图像，进行深度图估计、深度信息分层、dibr
'''
import argparse
import os
import numpy as np
from PIL import Image

import cv2

import time
from numba import jit             # numba jit技术实时编译


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('in_photo', help='输入图像')
parser.add_argument('model', help='input model :fcrn or  senet')
args = parser.parse_args()


#利用dlib进行前脸检测，输出cv_rects
def faces_detect(img):
    import dlib
    detector = dlib.get_frontal_face_detector()

    face_rects = detector(img, 0)#face_rects[0] -> (x1,y1),(x2,y2)  格式为dlib.rectangle
    list_rects = []
    for k, d in enumerate(face_rects):
        list_rects.append((d.left(),d.top(),d.right(),d.bottom()))
    return list_rects




def start_run():

    model_name = args.model

    input_img = cv2.imread(args.in_photo)

    my_width = input_img.shape[1]
    my_height = input_img.shape[0]

    radio = 1.0
    if my_width > my_height and my_width > 640:
        radio = 640.0/my_width
        my_width = 640
        my_height = int(radio*my_height)
    elif my_width < my_height and my_height > 640:
        radio = 640.0/my_height
        my_height = 640
        my_width = int(radio*my_width)

    #检测 原图输入
    radio_rects = []
    list_rects = faces_detect(input_img)
    print ('faces:'+str(len(list_rects)),list_rects)
    for (x1,y1,x2,y2) in list_rects:
        x1 = int(radio*x1)
        y1 = int(radio*y1)        
        x2 = int(radio*x2)
        y2 = int(radio*y2)       
        radio_rects.append((x1,y1,x2,y2))
    print ('radio_faces:'+str(len(radio_rects)),radio_rects)
    
    print (my_width,my_height),model_name

    input_img = cv2.resize(input_img,(my_width,my_height))
    img_depth = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)

    if model_name == 'fcrn':
        import tensorflow as tf
        import NYU_FCRNmodels as models

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
            saver.restore(sess, '/home/jiang/Repositories/2.23dchange/depth_image-py/FCRN-DepthPrediction-master/tensorflow/NYU_FCRN.ckpt')

            print ("load time:",time.time()-starttime)

            source_img = input_img.copy()#cv2.imread(args.in_photo)
            #deal image	
            img = 	Image.fromarray(cv2.cvtColor(source_img,cv2.COLOR_BGR2RGB))             #opencv 转 PIL.Iamge
            source_img = cv2.resize(source_img,(my_width,my_height))

            img = img.resize([width,height], Image.ANTIALIAS)       #缩放
            img = np.array(img).astype('float32')                   #数据类型转换
            img = np.expand_dims(np.asarray(img), axis = 0)         #表示在0位置添加数据					
                

            # Evalute the network for the given image
            predicttime = time.time()
            pred = sess.run(net.get_output(), feed_dict={input_node: img})

            img_depth = cv2.cvtColor(np.asarray(pred[0,:,:,0]),cv2.COLOR_RGB2BGR)
            img_depth = cv2.cvtColor(img_depth,cv2.COLOR_BGR2GRAY)
            max_depth = np.max(img_depth)
            min_depth = np.min(img_depth)
            img_depth = (np.array(img_depth) - min_depth)/(max_depth - min_depth)   #归一化，归为０－１之间
            img_depth = cv2.resize(img_depth,(my_width,my_height)) 

    else:
        import torch.nn.parallel
        from torchvision import transforms, utils
        from pytorch_models import modules, net, resnet, densenet, senet
        import loaddata_demo as loaddata
        import pdb
        import model_path as pm
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

        if 1:
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

            nyu2_loader = loaddata.readNyu2(input_img)
            for  i,image in enumerate(nyu2_loader):

                with torch.no_grad():                  #释放内存
                    image = torch.autograd.Variable(image, volatile=True).cuda()
                    out = model(image)
                    mid_img = out.view(out.size(2),out.size(3)).data.cpu().numpy()

                    img_depth = (np.array(mid_img) - np.min(mid_img))/(np.max(mid_img) - np.min(mid_img))
                    img_depth = cv2.resize(img_depth,(input_img.shape[1],input_img.shape[0]))
                    #cv2.imwrite("faces_result/senet_"+args.in_photo,img_depth*255)			##保存深度图


		
    save_img = input_img.copy()
    save_img = cv2.resize(input_img,(input_img.shape[1]*2,input_img.shape[0]))
    save_d = cv2.cvtColor(img_depth*255,cv2.COLOR_GRAY2BGR)

    #缩放后，乘了比例系数后的rects
    mask = cv2.imread("mask1.png")
    for (x1,y1,x2,y2) in radio_rects:
        cv2.rectangle(save_d,(x1,y1),(x2,y2),(255,0,0),1,cv2.LINE_AA)
        pos = (x1,y1)
        rect_mean_value = int(np.mean(save_d[y1:y2,x1:x2,0]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(save_d,str(rect_mean_value),pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)

        cv2.putText(input_img,str(rect_mean_value),pos, font, 0.4, (255, 255, 0), 1,cv2.LINE_AA)
        mask = cv2.resize(mask,(x2-x1-4,y2-y1-4))
        input_img[y1+2:y2-2,x1 +2:x2 -2] = mask.copy()
        cv2.rectangle(input_img,(x1,y1),(x2,y2),(0,0,255),1,cv2.LINE_AA)


    save_img[:,0:input_img.shape[1]] = input_img.copy()
    save_img[:,input_img.shape[1]:save_img.shape[1]] = save_d.copy()
    cv2.imwrite("faces_depth/"+model_name+'-'+args.in_photo,save_img)			##保存深度图
    cv2.imshow(model_name+"show",save_img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0



   


if __name__ == '__main__':
	start_run()
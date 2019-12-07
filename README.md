# PrectictingDepth-DeepLearn-pdf

###################################################################

参考：https://blog.csdn.net/qq_39732684/article/details/80936492
其原文说明：
	作者：知乎用户
	链接：https://www.zhihu.com/question/53354718/answer/209398048
	来源：知乎
##################################################################

说明：
	根据总结，下载了所述论文，并分类
#################################################################


原文：

我觉得近几年采用深度学习来解决深度估计的思路可以分为好几类：
第一类 仅仅依靠深度学习和网络架构得到结果

最近这部分文章我较为详细的总结在了专栏里：深度学习之单目深度估计 (Chapter.1)：基础篇

1. 引用最多、最早的是Eigen组的两篇文章，相对于简单粗暴的使用卷积神经网络回归来得到结果，主要卖点是采用Multi-scale的卷积网络结构（2015年）：

    Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
    Depth Map Prediction from a Single Image using a Multi-Scale Deep Network

2. 之后在2016年，Laina依靠更深层次的网络和一个“novel”的逆卷积结构再加上一个"novel"的loss来得到结果。其实我认为这篇文章的贡献点不是很大，主要是pretrain的ResNet-50帮了很大的忙。这个方法被他们组改进然后用到了之后CVPR2017用来重建SLAM的文章中。

    Deeper Depth Prediction with Fully Convolutional Residual Networks （3DV 2016）
    CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction（2017 CVPR）

第二类 依靠于深度信息本身的性质

1. 深度信息和语义分割信息具有很强的相关性：场景中语义分割信息相似的物体所拥有的深度信息是相似的。

    Towards Unified Depth and Semantic Prediction From a Single Image (CVPR 2015)

2. 之后接下来又有文章试图去做了语义分割和深度信息的升级版：除了语义分割信息有没有其他信息和深度信息也相似的？

SURGE: Surface Regularized Geometry Estimation from a Single Image（NIPS 2016）

3. 深度信息本就是一个从远到近一层一层的分类，是不是把预测深度当做一个分类问题更好解一点，搜文章的时候搜到了这两篇用到了这个思路：

Estimating Depth from Monocular Images as Classification Using Deep Fully Convolutional Residual Networks

Single image depth estimation by dilated deep residual convolutional neural network and soft-weight-sum inference


第三类 基于CRF的方法

CRF之前一直在语义分割问题上表现的很好，包括CRFasRNN，DeepLab等等，几乎成为了这种回归问题的标配。这一类的方法采用CRF是因为通常CNN用来做回归时产生的图都比较糊(blur), CRF可以通过条件概率建模的方法将糊的图片变得不糊。这是一种纯数学解决问题的方法，与深度信息本身的物理性质关系不大。

Deep Convolutional Neural Fields for Depth Estimation from a Single Image（2015 CVPR）

Depth and surface normal estimation from monocular images using regression on deep features and hierarchical CRFs(2015 CVPR)

Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation (2017 CVPR)


第四类 基于相对深度

接下来介绍的这一类是我觉得最有意思的一个方法。 总的来说就是利用了深度信息的基本特征：图片中的点与点之间的是有相对远近关系的。NIPS2016这篇文章自己构建了一个相对深度的数据库，每张图片之中仅仅标注两个随机点之间的相对远近关系，通过一个神经网络的训练就能得到原图之中的相对深度信息。而且，一般的方法通常是针对某个数据库的数据范围的（NYUv2用来做室内深度预测，深度的ground truth 是 0~10m，KITTI用来处理行车道路信息，深度的ground truth 是 0~70m）,这篇文章的深度是没有这种限制的。这篇文章得到的深度信息感觉是很amazing的一件事，为这篇文章打call！ 但是缺点也很明显，由于是相对深度信息仅仅得到的数据表示了当前图片中物体的远近关系和真实深度信息有很大的区别。

Single-Image Depth Perception in the Wild (NIPS2016)

当然这种相对关系的想法第一个提出来应该是下面这篇文章。不过我觉得这篇文章生成的图看起来太“超像素”了，不太smooth（因为它训练的时候就是使用的是图片中超像素的中点）：

Learning Ordinal Relationships for Mid-Level Vision（2015ICCV）


第五类 非监督学习

最近这部分文章我较为详细的总结在了专栏里：深度学习之单目深度估计 (Chapter.2)：无监督学习篇

所谓使用非监督学习来训练就是利用不知道ground truth的输入图片训练来得到深度信息。既然没有深度的ground truth那肯定有来自于其他地方的约束，比如使用stereo image。stereo image是来自两个相机（或者双目相机）在同一水平线上左右相距一定位置得到的两幅图片。这种图片获取的代价要比深度的ground truth 低一些。 这些方法利用了深度信息和场景之间的一些物理规律来约束，感觉得到了很不错的结果，这三个方法可以说是一脉相承：

Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue（2016 ECCV)

Unsupervised Monocular Depth Estimation with Left-Right Consistency (2017 CVPR)

Semi-Supervised Deep Learning for Monocular Depth Map Prediction （2017 CVPR）
总结

其实感觉同样是image to image 的转换，深度信息相比于语义分割关注的人要少很多，很多语义分割方面的方法就会有可能直接用到深度预测方面。比如Estimating Depth from Monocular Images as Classification Using Deep Fully Convolutional Residual Networks 这篇文章其实就和CRFasRNN很像。



######################################################################################################

﻿12.7 人脸检测+单目深度估计代码整理(code/)


单目深度估计是我很感兴趣的方向，但没有太多时间去学，就跑了一下开源的两个比较好的程序。再结合人脸识别，看能不能制定一个指定识别第几个人的策略，抱着这样的想法，借用别人单目深度的实现以及基于dlib的人脸检测，实现效果如下。


以下为***Deeper Depth Prediction with Fully Convolutional Residual Networks***实现的深度效果，其开源代码[github-fcrn](https://github.com/iro-cp/FCRN-DepthPrediction)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/fcrn-faces1.jpg)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/fcrn-faces2.jpg)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/fcrn-faces3.jpg)

以下为
***Revisiting Single Image Depth Estimation:Toward Higher Resolution Maps with Accurate Object Boundaries***的深度估计效果，其开源代码[github-senet](https://github.com/junjH/Revisiting_Single_Depth_Estimation)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/senet-faces1.jpg)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/senet-faces2.jpg)

![image text](https://github.com/Youjiangbaba/PrectictingDepth-DeepLearn-pdf/tree/master/code/faces_detect_depth/faces_depth/senet-faces3.jpg)

可以看出，图二的对比，fcrn网络的实现是比较准确的，单但看深度图效果，文章2输出的深度图在边界轮廓的处理上有较好的效果。


考完试再好好研读一下论文，特别是第二种，如果可以的话，计划利用depth+rgb进行物件边缘的准确提取。下一步先看看hed的实现效果......学习基础网络、优化过程——>分析网络该网络结构进行优化——>迁移学习？？？

哎，还是复习考试吧。

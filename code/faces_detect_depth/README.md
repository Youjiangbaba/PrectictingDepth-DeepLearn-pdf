12.7 整理


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

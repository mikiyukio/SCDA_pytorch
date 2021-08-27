# SCDA_pytorch
This is a pytorch implement of SCDA (selective deep descriptors selection for fine-grained image retrieval), which is fully translated from its original matlab version



关于pytorch版本的SCDA的说明文档

首先需要明确的一点是，这是无监督的细粒度图像检索，在数据集的划分上与度量学习的习惯性划分是截然不同的，具体请参照SCDA 原论文
这份代码是对基于matlab版本的SCDA开源代码的翻译版，已经尽可能的以matlab版本的代码的设置为准了，当然可能有所疏漏，所以两份代码都要看，
使用复现的基于pytorch的代码时请仔细检查。

http://www.lamda.nju.edu.cn/code_SCDA.ashx
以上是SCDA matlab版的源码地址

关于运行复现的SCDA：
1.下载cub200-2011数据集
     下载vgg16预训练模型于 .\SCDA_for_LL\model文件夹下

2.更改 .\SCDA_for_LL\files.py 这个程序的第15行代码为自己下载的CUB 数据集的绝对路径，并运行该程序
       .\SCDA_for_LL\datafile   文件夹下会生成四个json文件，它们与加载数据集有关。

3.运行 .\SCDA_for_LL\WAOCD\original_SCDA.py  这是对于SCDA的尽可能精确的复现
      提取好的特征也存储在.\SCDA_for_LL\datafile 文件夹下（刚刚看了下，这份代码可以实现的更为简洁，请自行修改）

4. .\SCDA_for_LL\WAOCD\compute_recall_as_ms.py
      .\SCDA_for_LL\WAOCD\compute_map_test_batch_circle.py
      这两个程序用来衡量算法的效果，跑出来的结果一样，两版实现都是正确的

      .\SCDA_for_LL\WAOCD\compute_recall_as_ms.py 运行的速度相对要快很多 .但是我只实现了Recall@K 
       但是Recall@K是度量学习的评价指标，SCDA使用mAP@K作为评价指标（Recall@1等价于mAP@1）

      .\SCDA_for_LL\WAOCD\compute_map_test_batch_circle.py 运行较慢，但是Recall@K 与mAP@K都实现了
      并且计算的结果自动写入CSV文件，是很便利的；CSV文件在.\SCDA_for_LL\result  文件夹下

5. 如果读者想要进行真实的图像检索的话，也是可以的，以下的程序中是一个简易的版本
       .\SCDA_for_LL\WAOCD\img_retrival_original_scda.py  
      从测试集中选定一个query,然后修改程序第77行的query的路径
      第330行左右也是可以修改的，不多说，读者请看源码
       然后这份代码不光是图像检索相关，也对SCDA论文中的可视化的热力图进行了实现，具体请参照源码
       可视化的结果将会存储在.\SCDA_for_LL\retrivial_visualize文件夹下

以上便是全部的介绍
有任何问题请联系：hello_yuhan@163.com

2021.3.21      于涵

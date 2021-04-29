#define network
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np






#
# Sequential(
#   (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (1): ReLU(inplace=True)
#   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (3): ReLU(inplace=True)
#   (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (6): ReLU(inplace=True)
#   (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (8): ReLU(inplace=True)
#   (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (11): ReLU(inplace=True)
#   (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (13): ReLU(inplace=True)
#   (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (15): ReLU(inplace=True)
#   (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (18): ReLU(inplace=True)
#   (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (20): ReLU(inplace=True)
#   (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (22): ReLU(inplace=True)
#   (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (25): ReLU(inplace=True)
#   (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (27): ReLU(inplace=True)
#   (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#   (29): ReLU(inplace=True)
#   (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )








class FGIAnet(nn.Module):
    def __init__(self):
        super(FGIAnet, self).__init__()
        #下面这个地方的名字一定要是self.features,因为models.vgg16中有一个同名的属性，这样我们就可以顺利的加载预训练vgg16模型的部分参数了，根据字典的键值对
        self.features = nn.Sequential(*(list(models.vgg16(num_classes=1000,
                                                              pretrained=False).features.children())))  # 这个网络是3*448*448的原图像所使用的网络，最终输出特征图512*14*14
        #model.vgg19()我们调用了这样一个函数，这个函数返回的是VGG（）这个类的一个对象，而self.features便是VGG（）这个类的一个属性，代表了VGG系列网络中的卷积神经网络部分
        #这也正是我们所需要的部分，得到输出特征图；cnn部分的参数通过pretrained=True选用预训练模型中的参数值
        #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']，这是vgg16的CNN部分的示意图
        ## 这个网络是3*448*448的原图像所使用的网络，最终输出特征图是512*14*14左右附近
        #我看了一下pytorch中VGG（）的实现，输出特征图经过一个自适应平均池化为[512*7*7]的确定shape的tensor,然后flatten+full_connect,这是题外话了
        # self.patch_features = nn.Sequential(*(list(models.vgg16(num_classes=1000, pretrained=True).features.children())[
        #                                       :-1]))  ##这个网络是1*3*224*224的原图像所使用的网络，最终输出特征图512*14*14左右
        self.img_gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.img_classifier = nn.Linear(512, 200)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.img_classifier.weight, 0, 0.01)
        nn.init.constant_(self.img_classifier.bias, 0)

    def forward(self, imgs):
        # imgs:[batch_size,3,488,488]
        # GLobal-Stream
        out=[]
        for i in range(len(self.features)):
            imgs = self.features[i](imgs)
            if i in [27,30]:   #27对应relu5_2的特征图，30对应pool5的特征图
                # # print(self.net[i])
                # # 这个地方出现了一个问题，就是这里的数无论我是选26,28（卷积层的输出）；还是选27,29（relu层的输出）
                # # 最后的out里面存储的好像都是relu之后的特征图，我反复研究了原因；是这样的，imgs与存入列表out中的imgs.data
                # # 指向的是内存中的同一区域，也就是说是内存共享的，也就是说，对于imgs进行修改，imgs.data能够同步被修改，反之亦然
                # # 比方说imgs[0]=1这样一个在原内存上的修改操作，print（imgs.data）观察一下，也是被修改了，反之亦然
                # # 但是要警惕这样一种特殊的情况imgs = self.features[i](imgs)，这是一种类似于imgs=imgs+1这种形式的操作
                # # 但是要注意的是，这是一个定义新的变量且赋值的操作而非一个简单的在原内存上对于同一变量的修改的操作
                # # 定义了新变量的话，比方说out.append(imgs.data),然后imgs=imgs+1，此时的imgs与out列表里存储的变量实质上就是在两个内存上的
                # # 也就是说imgs=imgs+1虽然看似没变啥，但imgs已经是一个新内存上的新变量了，跟out里面存储的imgs.data自然没啥关系，也不会改变out里面存储的内容的值
                # # 但是如果说没有定义新变量的话，比方说out.append(imgs.data),然后imgs[0]=1这样一步修改,虽然imgs[0]=1是一步之后的操作，但是呢out里面存储的值也被同步的改变了
                # # 因为out里面存储的值与imgs是处在内存中的同一位置上的
                # # 最后，我们可以言归正传了，就是imgs=torch.nn.ReLU(imgs,im_place=True),关于这一步操作的特殊性，确实，这是一步定义了新的变量的操作
                # # 但是呢，im_place=True直接在原有输入内存上修改输入，不额外申请，不报错就能用，节省内存。但是却造成了我刚刚的困惑；问题发现且描述清楚了
                # # 以上。有两种解决的办法，一种是in_place=False,另一种是img_tmp=imgs.data+0  (imgs_tmp=imgs,data不是赋值操作);out.append(imgs_tmp)
                # imgs_tmp = imgs.data + 0
                # out.append(imgs_tmp)

                out.append(imgs)
        # img_features_pool5 = self.img_features(imgs)
        img_f = self.img_gap(imgs)
        img_f = img_f.view(img_f.shape[0], -1)  # [batchsize,512,1,1]====>(batch_size,512)
        img_f=self.img_classifier(img_f)
        # img_f=img_f.data
        return out,img_f  #[batchsize,200]
        # return out  #[batchsize,200]




class FGIAnet100(nn.Module):
    def __init__(self):
        super(FGIAnet100, self).__init__()
        #下面这个地方的名字一定要是self.features,因为models.vgg16中有一个同名的属性，这样我们就可以顺利的加载预训练vgg16模型的部分参数了，根据字典的键值对
        self.features = nn.Sequential(*(list(models.vgg16(num_classes=1000,
                                                              pretrained=False).features.children())))  # 这个网络是3*448*448的原图像所使用的网络，最终输出特征图512*14*14
        #model.vgg19()我们调用了这样一个函数，这个函数返回的是VGG（）这个类的一个对象，而self.features便是VGG（）这个类的一个属性，代表了VGG系列网络中的卷积神经网络部分
        #这也正是我们所需要的部分，得到输出特征图；cnn部分的参数通过pretrained=True选用预训练模型中的参数值
        #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']，这是vgg16的CNN部分的示意图
        ## 这个网络是3*448*448的原图像所使用的网络，最终输出特征图是512*14*14左右附近
        #我看了一下pytorch中VGG（）的实现，输出特征图经过一个自适应平均池化为[512*7*7]的确定shape的tensor,然后flatten+full_connect,这是题外话了
        # self.patch_features = nn.Sequential(*(list(models.vgg16(num_classes=1000, pretrained=True).features.children())[
        #                                       :-1]))  ##这个网络是1*3*224*224的原图像所使用的网络，最终输出特征图512*14*14左右
        self.img_gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.img_classifier = nn.Linear(512, 100)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.img_classifier.weight, 0, 0.01)
        nn.init.constant_(self.img_classifier.bias, 0)

    def forward(self, imgs):
        # imgs:[batch_size,3,488,488]
        # GLobal-Stream
        out=[]
        # print(self.features)
        for i in range(len(self.features)):
            imgs = self.features[i](imgs)
            if i in [27,29]:   #27对应relu5_2的特征图，30对应pool5的特征图
                # print(self.net[i])
                out.append(imgs.data)
        # img_features_pool5 = self.img_features(imgs)
        img_f = self.img_gap(imgs)
        img_f = img_f.view(img_f.shape[0], -1)  # [batchsize,512,1,1]====>(batch_size,512)
        img_f=self.img_classifier(img_f)
        # img_f=img_f.data
        return out,img_f  #[batchsize,200]
        # return out  #[batchsize,200]



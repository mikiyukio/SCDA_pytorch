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
        #???????????????????????????????????????self.features,??????models.vgg16???????????????????????????????????????????????????????????????????????????vgg16???????????????????????????????????????????????????
        self.features = nn.Sequential(*(list(models.vgg16(num_classes=1000,
                                                              pretrained=False).features.children())))  # ???????????????3*448*448??????????????????????????????????????????????????????512*14*14
        #model.vgg19()????????????????????????????????????????????????????????????VGG????????????????????????????????????self.features??????VGG??????????????????????????????????????????VGG??????????????????????????????????????????
        #???????????????????????????????????????????????????????????????cnn?????????????????????pretrained=True????????????????????????????????????
        #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']?????????vgg16???CNN??????????????????
        ## ???????????????3*448*448?????????????????????????????????????????????????????????512*14*14????????????
        #???????????????pytorch???VGG?????????????????????????????????????????????????????????????????????[512*7*7]?????????shape???tensor,??????flatten+full_connect,??????????????????
        # self.patch_features = nn.Sequential(*(list(models.vgg16(num_classes=1000, pretrained=True).features.children())[
        #                                       :-1]))  ##???????????????1*3*224*224??????????????????????????????????????????????????????512*14*14??????
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
            if i in [27,30]:   #27??????relu5_2???????????????30??????pool5????????????
                # # print(self.net[i])
                # # ?????????????????????????????????????????????????????????????????????26,28????????????????????????????????????27,29???relu???????????????
                # # ?????????out???????????????????????????relu???????????????????????????????????????????????????????????????imgs???????????????out??????imgs.data
                # # ?????????????????????????????????????????????????????????????????????????????????????????????imgs???????????????imgs.data????????????????????????????????????
                # # ?????????imgs[0]=1?????????????????????????????????????????????print???imgs.data???????????????????????????????????????????????????
                # # ??????????????????????????????????????????imgs = self.features[i](imgs)????????????????????????imgs=imgs+1?????????????????????
                # # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # # ????????????????????????????????????out.append(imgs.data),??????imgs=imgs+1????????????imgs???out????????????????????????????????????????????????????????????
                # # ????????????imgs=imgs+1???????????????????????????imgs????????????????????????????????????????????????out???????????????imgs.data????????????????????????????????????out???????????????????????????
                # # ??????????????????????????????????????????????????????out.append(imgs.data),??????imgs[0]=1??????????????????,??????imgs[0]=1????????????????????????????????????out??????????????????????????????????????????
                # # ??????out?????????????????????imgs???????????????????????????????????????
                # # ?????????????????????????????????????????????imgs=torch.nn.ReLU(imgs,im_place=True),???????????????????????????????????????????????????????????????????????????????????????
                # # ????????????im_place=True????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                # # ?????????????????????????????????????????????in_place=False,????????????img_tmp=imgs.data+0  (imgs_tmp=imgs,data??????????????????);out.append(imgs_tmp)
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
        #???????????????????????????????????????self.features,??????models.vgg16???????????????????????????????????????????????????????????????????????????vgg16???????????????????????????????????????????????????
        self.features = nn.Sequential(*(list(models.vgg16(num_classes=1000,
                                                              pretrained=False).features.children())))  # ???????????????3*448*448??????????????????????????????????????????????????????512*14*14
        #model.vgg19()????????????????????????????????????????????????????????????VGG????????????????????????????????????self.features??????VGG??????????????????????????????????????????VGG??????????????????????????????????????????
        #???????????????????????????????????????????????????????????????cnn?????????????????????pretrained=True????????????????????????????????????
        #[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']?????????vgg16???CNN??????????????????
        ## ???????????????3*448*448?????????????????????????????????????????????????????????512*14*14????????????
        #???????????????pytorch???VGG?????????????????????????????????????????????????????????????????????[512*7*7]?????????shape???tensor,??????flatten+full_connect,??????????????????
        # self.patch_features = nn.Sequential(*(list(models.vgg16(num_classes=1000, pretrained=True).features.children())[
        #                                       :-1]))  ##???????????????1*3*224*224??????????????????????????????????????????????????????512*14*14??????
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
            if i in [27,29]:   #27??????relu5_2???????????????30??????pool5????????????
                # print(self.net[i])
                out.append(imgs.data)
        # img_features_pool5 = self.img_features(imgs)
        img_f = self.img_gap(imgs)
        img_f = img_f.view(img_f.shape[0], -1)  # [batchsize,512,1,1]====>(batch_size,512)
        img_f=self.img_classifier(img_f)
        # img_f=img_f.data
        return out,img_f  #[batchsize,200]
        # return out  #[batchsize,200]



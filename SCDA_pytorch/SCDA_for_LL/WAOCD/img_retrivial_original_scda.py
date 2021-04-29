import numpy as np
from sklearn.neighbors import KDTree
import json
import matplotlib.pyplot as plt
from SCDA_for_LL.net import FGIAnet,FGIAnet100
from SCDA_for_LL.bwconncomp import largestConnectComponent
import os
import argparse
from os.path import join
import uuid
import json
from SCDA_for_LL import dataloader
from torch.nn.functional import interpolate
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Caltech-UCSD Birds-200 2011\Caltech-UCSD Birds-200-2011\CUB_200_2011',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='images',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=200)
parser.add_argument('--img_size', type=int, default = 700,help='进行特征提取的图片的尺寸的上界所对应的数量级')
# parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_11_0.6863.pth')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_36_0.9919.pth')

parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
args = parser.parse_args()



net = FGIAnet()


# checkpoint = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,args.model_name))
# net.load_state_dict(checkpoint['model'])



save_model = torch.load(join(os.path.abspath(os.path.dirname(os.getcwd())),args.savedir,'vgg16-397923af.pth'))
model_dict =  net.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
net.load_state_dict(model_dict)



net.eval()
net.cuda()



train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
with open(train_paths_name) as miki:
        train_paths=json.load(miki)
with open(test_paths_name) as miki:
        test_paths=json.load(miki)
with open(train_labels_name) as miki:
        train_labels=json.load(miki)
with open(test_labels_name) as miki:
        test_labels=json.load(miki)


#上面的这些行都没用，主要是指定test_paths,这里面指定这我们所要检索的query所在的路径


# img_name='006.Least_Auklet/Least_Auklet_0024_795120.jpg'
img_name = '200.Common_Yellowthroat/Common_Yellowthroat_0055_190967.jpg'
# img_name = '110.Geococcyx/Geococcyx_0116_104240.jpg'
# img_name = '200.Common_Yellowthroat/Common_Yellowthroat_0020_190720.jpg'
# img_name='006.Least_Auklet/Least_Auklet_0014_1901.jpg'









test_path=[]
test_path.append(join(args.datasetdir,args.imgdir,img_name))
# test_path.append(img_name)

loaders = dataloader.get_dataloaders(train_paths, test_path,train_labels,test_labels,args.img_size, 1,1,1,SCDA_flag=1)#返回值为由可迭代DataLoader对象所组成的字典



print("cnn model is ready.")

tr_L31_mean = []
tr_L31_flip_mean = []
te_L31_mean = []
te_L31_flip_mean = []
tr_L28_mean = []
tr_L28_flip_mean = []
te_L28_mean = []
te_L28_flip_mean = []


tr_L31_max = []
tr_L31_flip_max = []
te_L31_max = []
te_L31_flip_max = []
tr_L28_max = []
tr_L28_flip_max = []
te_L28_max = []
te_L28_flip_max = []
yuri=149

ii=0
for phase in ['test']:
        for images, labels in loaders[phase]:#一个迭代器，迭代数据集中的每一batch_size图片;迭代的返回值dataset的子类中的__getitem__()方法是如何进行重写的；
                print(ii)
                ii=ii+1
                for flip in range(2):
                        if flip==0:
                                pass
                        else:
                                images=images[:,:,:,torch.arange(images.size(3)-1,-1,-1).long()]  #整个batch_size的所有图像水平翻转
                        # print(images[0].size())
                        # image=images[0]#去除了batch_size那一维度，反正batch_size都是1，有没有无所谓
                        batch_size,c,h,w=images.size()
                        if min(h,w) > args.img_size:
                                images= interpolate(images,size=[int(h * (args.img_size / min(h, w))),int(w * (args.img_size / min(h, w)))],mode="bilinear",align_corners=True)
                                # %我就打个比方吧，min(h,w)=h的话  h*(700/min(h,w)=700   w*(700/min(h,w)=w*(700/h)=700*(w/h) 图像的size由[h,w]变为[700,700*(w/h)]
                                #%由此可见，在min(h,w) > 700的前提下，图像被适当的进行分辨率的缩小，到700这一级，但是长宽比是没有改变的，图像没有变形
                                # %这一步操作只是为了对于图像的分辨率的上限进行一个限制
                        batch_size, c, h, w = images.size()
                        #matlab版本的实现中这里是对可能出现的灰度图像进行通道数扩充，并减去图像在各个通道上的均值，以上过程我在dataloader.py中已经实现了，以上。
                        images,labels=images.cuda(),labels.cuda()
                        # labels=labels.cpu()

                        #传统的SCDA
                        cnnFeature_maps,_ = net(images)  #img_preds:[batch_size=1,200]

                        feature_maps_L28 = cnnFeature_maps[
                            0]  # 如果输入图像的shape为[batch_size,3,224,224]的tensor的话，关于输出的feature map,shape为  #[batch_size,512,14,14]左右
                        batch_size, c_L28, h_L28, w_L28 = feature_maps_L28.size()
                        feature_maps_L31 = cnnFeature_maps[
                            1]  ##[batch_size,512,7,7]左右     当然啦，我们在测试阶段或者度量学习阶段或者为图像生成特征描述阶段，输入图像的尺寸是任意的，这也就决定了batch_size只能为1；训练阶段强制性resize为3*224*224
                        batch_size, c_L31, h_L31, w_L31 = feature_maps_L31.size()



                        #Pool5   #feature_maps_L31  [512,7,7]==[512,]
                        feature_maps_L31_sum = torch.sum(feature_maps_L31[0],
                                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
                        L31_sum_mean = torch.mean(feature_maps_L31_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
                        highlight = torch.zeros(feature_maps_L31_sum.size())  # 生成了一个h*w的全零矩阵
                        # print(highlight)
                        highlight_index = torch.nonzero(feature_maps_L31_sum > L31_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
                        a, b = highlight_index.size()
                        # print(highlight_index.size())

                        for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
                            highlight[highlight_index[i][0], highlight_index[i][1]] = 1
                        # 以上，我们便初步获取得到了聚合特征图掩码矩阵
                        # 然后，我们需要对掩码矩阵进行进一步的处理，获得最大连通区域
                        # largestConnectComponent将最大连通区域所对应的像素点置为true
                        highlight_conn_L31 = torch.tensor(largestConnectComponent(highlight.numpy()) + 0)#[7*7],我们需要将其变成[512,7,7]
                        feature_maps_L31 = feature_maps_L31[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
                        highlight_conn_L31=highlight_conn_L31.cpu().data.numpy()
                        highlight_conn_L31_beifen = highlight_conn_L31
                        highlight_conn_L31 = highlight_conn_L31.reshape(1,h_L31,w_L31 ) * np.ones_like(feature_maps_L31)#[7,7]=>[1,7,7]=>[512,7,7]

                        feature_maps_L31=feature_maps_L31*highlight_conn_L31

                        feature_maps_L31_mean = np.sum(feature_maps_L31, axis=(1,2))/a
                        feature_maps_L31_mean_norm = feature_maps_L31_mean / np.linalg.norm(feature_maps_L31_mean)
                        feature_maps_L31_max = np.max(feature_maps_L31, axis=(1,2))
                        feature_maps_L31_max_norm = feature_maps_L31_max / np.linalg.norm(feature_maps_L31_max)

                        # % Relu5_2
                        feature_maps_L28_sum = torch.sum(feature_maps_L28[0],
                                                         0)  # 如果要将CAM或者gard_cam引入SCDA，需要更改的程序也就仅仅是这一步
                        L28_sum_mean = torch.mean(feature_maps_L28_sum)  # 返回的是scalar tensor ,这是与matlab的mean不同的地方
                        highlight = torch.zeros(feature_maps_L28_sum.size())  # 生成了一个h*w的全零矩阵
                        highlight_index = torch.nonzero(feature_maps_L28_sum > L28_sum_mean)  # 一个K*2的tensor,每一行一个符合要求的点
                        a, b = highlight_index.size()
                        for i in range(a):  # a就是被选择出来的点的个数，这些点我们需要将其标记为1
                            highlight[highlight_index[i][0], highlight_index[i][1]] = 1
                        highlight_conn_L31=torch.tensor(highlight_conn_L31_beifen)
                        highlight_conn_L31_to_L28 = interpolate(highlight_conn_L31.view(1, 1, h_L31, w_L31).float(),
                                                                size=[h_L28, w_L28], mode="nearest").view(h_L28, w_L28)
                        highlight_conn_L28 = highlight.mul(highlight_conn_L31_to_L28)  # 逐点按元素想乘，两个都是二值矩阵，可不就是按位与嘛

                        feature_maps_L28 = feature_maps_L28[0].cpu().data.numpy()  # [1,512,7,7]==>[512,7,7]
                        highlight_conn_L28=highlight_conn_L28.cpu().data.numpy()
                        highlight_conn_L28_beifen = highlight_conn_L28
                        highlight_conn_L28 = highlight_conn_L28.reshape(1, h_L28, w_L28) * np.ones_like(
                            feature_maps_L28)  # [7,7]=>[1,7,7]=>[512,7,7]

                        feature_maps_L28 = feature_maps_L28 * highlight_conn_L28

                        feature_maps_L28_mean = np.sum(feature_maps_L28, axis=(1,2))/a
                        feature_maps_L28_mean_norm = feature_maps_L28_mean / np.linalg.norm(feature_maps_L28_mean)
                        feature_maps_L28_max = np.max(feature_maps_L28, axis=(1, 2))
                        feature_maps_L28_max_norm = feature_maps_L28_max / np.linalg.norm(feature_maps_L28_max)

                        if flip==0:
                            feature_maps_L31_sum_noflip=feature_maps_L31_sum
                            feature_maps_L28_sum_noflip=feature_maps_L28_sum   #这是为最后的可视化准备的
                            # feature_maps_L31_sum_noflip = cnnFeature_maps[0][0][yuri]
                            # feature_maps_L28_sum_noflip = cnnFeature_maps[1][0][yuri]
                            highlight_conn_L28_noflip=highlight_conn_L28_beifen
                            highlight_conn_L31_noflip=highlight_conn_L31_beifen
                        if phase=='train':
                                if flip==0:
                                        tr_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                                        tr_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                                        tr_L31_max.append(feature_maps_L31_max_norm.tolist())
                                        tr_L28_max.append(feature_maps_L28_max_norm.tolist())
                                else:
                                        tr_L31_flip_mean.append(feature_maps_L31_mean_norm.tolist())
                                        tr_L28_flip_mean.append(feature_maps_L28_mean_norm.tolist())
                                        tr_L31_flip_max.append(feature_maps_L31_max_norm.tolist())
                                        tr_L28_flip_max.append(feature_maps_L28_max_norm.tolist())
                        else:
                                if flip==0:
                                        te_L31_mean.append(feature_maps_L31_mean_norm.tolist())
                                        te_L28_mean.append(feature_maps_L28_mean_norm.tolist())
                                        te_L31_max.append(feature_maps_L31_max_norm.tolist())
                                        te_L28_max.append(feature_maps_L28_max_norm.tolist())

                                else:
                                        te_L31_flip_mean.append(feature_maps_L31_mean_norm.tolist())
                                        te_L28_flip_mean.append(feature_maps_L28_mean_norm.tolist())
                                        te_L31_flip_max.append(feature_maps_L31_max_norm.tolist())
                                        te_L28_flip_max.append(feature_maps_L28_max_norm.tolist())


print('stacking starting...............................................')

# tr_L31_mean = np.array(tr_L31_mean)
# tr_L31_flip_mean = np.array(tr_L31_flip_mean)
te_L31_mean = np.array(te_L31_mean)
te_L31_flip_mean = np.array(te_L31_flip_mean)
# tr_L28_mean = np.array(tr_L28_mean)
# tr_L28_flip_mean = np.array(tr_L28_flip_mean)
te_L28_mean = np.array(te_L28_mean)
te_L28_flip_mean = np.array(te_L28_flip_mean)


# tr_L31_max = np.array(tr_L31_max)
# tr_L31_flip_max = np.array(tr_L31_flip_max)
te_L31_max = np.array(te_L31_max)
te_L31_flip_max = np.array(te_L31_flip_max)
# tr_L28_max = np.array(tr_L28_max)
# tr_L28_flip_max = np.array(tr_L28_flip_max)
te_L28_max = np.array(te_L28_max)
te_L28_flip_max = np.array(te_L28_flip_max)





test_data=te_L31_mean.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
final_features['test']=test_data
final_features_512_avg=final_features




test_data=te_L31_max.tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
final_features['test']=test_data
final_features_512_max=final_features





test_data=np.hstack([te_L31_mean,te_L31_max
                      ]).tolist()
print('test_data.shape:',np.array(test_data).shape)
final_features={}
final_features['test']=test_data
final_features_1024=final_features




ratio=0.5   #来自cnn不同层的特征图得到的feature的加权比
test_data=np.hstack([te_L31_mean,te_L31_max,
                      ratio*te_L28_mean,ratio*te_L28_max
                      ]).tolist()
print('test_data.shape:',np.array(test_data).shape)

final_features={}
final_features['test']=test_data
final_features_2048=final_features





ratio=0.5   #来自cnn不同层的特征图得到的feature的加权比
test_data=np.hstack([te_L31_mean,te_L31_max,
                      ratio*te_L28_mean,ratio*te_L28_max,
                     te_L31_flip_mean, te_L31_flip_max,
                     ratio * te_L28_flip_mean, ratio * te_L28_flip_max
                     ]).tolist()
print('test_data.shape:',np.array(test_data).shape)

final_features={}
final_features['test']=test_data
final_features_4096=final_features







#以下五组任选即可

test_data=final_features_512_max['test']
filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/final_representation_512_max.json')
#
#
# test_data=final_features_1024['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/final_representation_1024.json')
#
#
# test_data=final_features_2048['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/final_representation_2048.json')
#
#
# test_data=final_features_4096['test']
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/final_representation_4096.json')



with open(filename) as f_obj:
    final_features=json.load(f_obj)
train_data=final_features['train']



train_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_paths.json')
test_paths_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_paths.json')
train_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/train_labels.json')
test_labels_name=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/test_labels.json')
with open(train_paths_name) as miki:
        train_paths=json.load(miki)
with open(test_paths_name) as miki:
        test_paths=json.load(miki)
with open(train_labels_name) as miki:
        train_labels=json.load(miki)
with open(test_labels_name) as miki:
        test_labels=json.load(miki)


# train_paths=test_paths
# train_labels=test_labels


print(KDTree.valid_metrics)
X = np.array(train_data)
query=np.array(test_data)
kdt = KDTree(X, leaf_size=40, metric='euclidean')
# s = pickle.dumps(kdt)                     # doctest: +SKIP
# kdt = pickle.loads(s)
result=kdt.query(query, k=49, return_distance=False)
#大概就是这样子的，有K个query,最终返回的a矩阵就有K行，每一行都有九个元素，代表
#我们以第一行为例吧，第一行的九个元素。就代表距离第一个query距离最近的九张图片的索引（这九张图片来自训练集），我们通过这个索引
#就可以获知这九张图片的具体的类别（如果我们已知了query的类别，既可以据此来计算mAP），
# 以及对应的存储路径（在图像检索系统中进行最后的检索结果的输出）
print('检索结果的在训练集中的标号索引：')
print(result)
print(result.shape)
query_result_labels=np.zeros(result.shape)
h,w=result.shape
for i in range(h):
    for j in range(w):
        query_result_labels[i,j]=train_labels[result[i,j]]
print('检索结果的类别索引(注意是从0开始的)：')
print(query_result_labels)




query_result_paths=[]
h,w=result.shape
for i in range(h):
    query_result_paths.append([])
    for j in range(w):
        query_result_paths[i].append(train_paths[result[i,j]])



# miki_1=cv2.imread(str(test_paths[0]))
miki_1 = Image.open(test_path[0]).convert('RGB')
width,height = miki_1.size
print(miki_1.size)
# print(cam_L31)


highlight_L31=highlight_conn_L31_noflip * 255
highlight_L28=highlight_conn_L28_noflip * 255
feature_maps_L28_sum=feature_maps_L28_sum_noflip
feature_maps_L31_sum=feature_maps_L31_sum_noflip
feature_maps_L28_sum = feature_maps_L28_sum - torch.min(feature_maps_L28_sum)
feature_maps_L28_sum = feature_maps_L28_sum / torch.max(feature_maps_L28_sum)
feature_maps_L31_sum = feature_maps_L31_sum - torch.min(feature_maps_L31_sum)
feature_maps_L31_sum = feature_maps_L31_sum / torch.max(feature_maps_L31_sum)
feature_maps_L31_sum=feature_maps_L31_sum.cpu().data.numpy()*255
feature_maps_L28_sum=feature_maps_L28_sum.cpu().data.numpy()*255

cam_L31 = Image.fromarray(highlight_L31).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.6
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L31_original_scda.jpg'),result)
cv2.waitKey(0)

cam_L31 = Image.fromarray(highlight_L28).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.6
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/highlight_L28_original_scda.jpg'),result)
cv2.waitKey(0)


cam_L31 = Image.fromarray(feature_maps_L31_sum).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.6
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L31_sum.jpg'),result)
cv2.waitKey(0)

cam_L31 = Image.fromarray(feature_maps_L28_sum).convert('L')
cam_L31 = cam_L31.resize((width, height),Image.ANTIALIAS)
print(cam_L31.size)
# Image._show(cam_L31)
cam_L31 = cv2.cvtColor(np.asarray(cam_L31),cv2.COLOR_RGB2BGR)
# cam_L31 = cv2.cvtColor(np.asarray(cam_L31))
cam_L31 = cv2.applyColorMap(cam_L31, cv2.COLORMAP_JET)
miki_1=cv2.cvtColor(np.asarray(miki_1),cv2.COLOR_RGB2BGR)
# cam_L31 = Image.fromarray(cv2.cvtColor(cam_L31,cv2.COLOR_BGR2RGB))
# heatmap_L31,miki_1=np.array(heatmap_L31),np.array(miki_1)
result = cam_L31 + miki_1 * 0.6
print(result.shape)
# plt.imsave('1.jpg',heatmap_L31)
cv2.imwrite(join(os.path.abspath(os.path.dirname(os.getcwd())),'./retrivial_visualize/feature_maps_L28_sum.jpg'),result)
cv2.waitKey(0)



plt.figure()
miki=plt.imread(os.path.join(join(args.datasetdir,args.imgdir,img_name)))
plt.imshow(miki)





plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    # print(query_result_paths[0])
    miki=plt.imread(str(query_result_paths[0][i]))
    plt.imshow(miki)
    plt.xticks([])
    plt.yticks([])
plt.show()
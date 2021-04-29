#对于部分预训练的VGG16风格的网络进行微调


import argparse
from os.path import join

import json


# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)#这是归一化之后的数值
# mean = (123.6800, 116.7790, 103.9390)

parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'C:\Users\于涵\Desktop\Caltech-UCSD Birds-200 2011\Caltech-UCSD Birds-200-2011\CUB_200_2011',  help="path to cub200_2011 dir")
parser.add_argument('--imgdir', default='images',  help="path to train img dir")
parser.add_argument('--tr_te_split_txt', default=r'train_test_split.txt',  help="关于训练集与测试集的划分，0代表测试集，1代表训练集")
parser.add_argument('--tr_te_image_name_txt', default=r'images.txt',  help="关于训练集与测试集的图片的相对路径名字")
parser.add_argument('--image_labels_txt', default=r'image_class_labels.txt',  help="图像的类别标签标记")
parser.add_argument('--class_name_txt', default=r'classes.txt',  help="图像的200个类别名称")
args = parser.parse_args()


train_index=[]
test_index=[]
# print(join(args.datasetdir,args.tr_te_split_txt))
with open(join(args.datasetdir,args.tr_te_split_txt)) as miki:
    for line in miki:
        # print(line.rstrip())
        line=line.rstrip()
        index,tr_te_type=line.split(' ')
        index=int(index)
        tr_te_type=int(tr_te_type)
        if tr_te_type==0:
            test_index.append(index)
        else:
            train_index.append(index)


train_labels=[]
test_labels=[]
with open(join(args.datasetdir,args.image_labels_txt)) as lisalisa:
    for line in lisalisa:
        # print(line.rstrip())
        line=line.rstrip()
        index,label=line.split(' ')
        index=int(index)
        label=int(label)-1   #类别索引是0~199，用于在loss部分自动生成one_hot
        tr_te_type=int(tr_te_type)
        if index in train_index:
            train_labels.append(label)
        else:
            test_labels.append(label)
# print(len(train_labels))
# print(len(test_labels))
filename1='./datafile/train_labels.json'
filename2='./datafile/test_labels.json'
with open(filename1,'w') as f_obj:
    json.dump(train_labels,f_obj)
with open(filename2,'w') as f_obj:
    json.dump(test_labels,f_obj)


train_paths=[]
test_paths=[]
with open(join(args.datasetdir,args.tr_te_image_name_txt)) as iggy:
    for line in iggy:
        # print(line.rstrip())
        line = line.rstrip()
        index, img_name = line.split(' ')
        index = int(index)
        img_name=img_name.strip()
        # print(index)
        # print(img_name)
        if index in train_index:
            train_paths.append(join(args.datasetdir,args.imgdir,img_name))
        else:
            test_paths.append(join(args.datasetdir,args.imgdir,img_name))
# print(len(train_paths))
# print(len(test_paths))
filename3='./datafile/train_paths.json'
filename4='./datafile/test_paths.json'
with open(filename3,'w') as f_obj:
    json.dump(train_paths,f_obj)
with open(filename4,'w') as f_obj:
    json.dump(test_paths,f_obj)

index_class_dict={}
with open(join(args.datasetdir,args.class_name_txt)) as hata:
    for line in hata:
        # print(line.rstrip())
        line=line.rstrip()
        index,class_name=line.split(' ')
        index=int(index)
        index_class_dict[str(index)]=class_name
# print(index_class_dict)

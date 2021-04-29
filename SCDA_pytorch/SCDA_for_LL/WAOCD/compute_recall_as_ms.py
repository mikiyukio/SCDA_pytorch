
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv

PARA=[
      'vgg16-397923af.pth',
      ]

# PARA=[
#         'resnet50-19c8e357.pth'
#       ]


files_json=[
            'final_representation_512_avg.json',
            'final_representation_512_max.json',
            'final_representation_1024.json',
            'final_representation_2048.json',
            'final_representation_4096.json',

]



results_csv=[
            'final_representation_512_avg.csv',
            'final_representation_512_max.csv',
            'final_representation_1024.csv',
            'final_representation_2048.csv',
            'final_representation_4096.csv',

]





class RetMetric(object):
    def __init__(self, feats, labels):

        if len(feats) == 2 and type(feats) == list:
            print('无监督')
            """
            feats = [gallery_feats, query_feats]
            labels = [gallery_labels, query_labels]
            """
            self.is_equal_query = False

            self.gallery_feats, self.query_feats = feats
            self.gallery_labels, self.query_labels = labels

        else:
            print('有监督')
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m






for index in range(len(PARA)):
    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    # target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile')

    for index_2 in range(len(files_json)):

        filename=join(target_path,files_json[index_2])
        print(filename)
        # f = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'a+',encoding='utf-8',newline='' "")
        # ff = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'r',encoding='utf-8',newline='' "")
        # l=len(ff.readlines())
        # ff.close()





        with open(filename) as f_obj:
            final_features=json.load(f_obj)
        train_data=final_features['train']
        test_data=final_features['test']
        print('ok******************')


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


        ########################333
        # dataset_labels=test_labels+train_labels
        dataset_labels=[np.array(train_labels),np.array(test_labels)]

        ###########################
        X = [np.array(train_data),np.array(test_data)]


        # print(X.shape)

        metric=RetMetric(X,dataset_labels)
        print(metric.recall_k(1))
        # print(metric.recall_k(2))
        # print(metric.recall_k(4))
        # print(metric.recall_k(8))
        # print(metric.recall_k(16))
        # print(metric.recall_k(32))

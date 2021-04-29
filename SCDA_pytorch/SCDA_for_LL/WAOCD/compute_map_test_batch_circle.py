import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KDTree
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from os.path import join
import os
import csv


PARA=[

    'vgg16-397923af.pth',
]




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







for index in range(len(PARA)):
    PARA_I = PARA[index].replace('.', '_')  # 避免路径出错
    # target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile', PARA_I)
    target_path = join(os.path.abspath(os.path.dirname(os.getcwd())), 'datafile')

    for index_2 in range(len(files_json)):

        filename=join(target_path,files_json[index_2])
        print(filename)
        f = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'a+',encoding='utf-8',newline='' "")
        ff = open(join(os.path.abspath(os.path.dirname(os.getcwd())),'result',results_csv[index_2]),'r',encoding='utf-8',newline='' "")
        l=len(ff.readlines())
        ff.close()





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
        dataset_labels=train_labels

        #############################


        print(KDTree.valid_metrics)
        X = np.array(train_data)
        print(X.shape)



        ###########################################3
        # X1=np.array(train_data)
        # X=np.vstack((X,X1))
        # print(X.shape)
        #########################################




        query=np.array(test_data)
        # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        kdt = KDTree(X, leaf_size=40, metric='l2')
        # s = pickle.dumps(kdt)                     # doctest: +SKIP
        # kdt = pickle.loads(s)
        result=kdt.query(query, k=200, return_distance=False)
        #大概就是这样子的，有K个query,最终返回的a矩阵就有K行，每一行都有九个元素，代表
        #我们以第一行为例吧，第一行的九个元素。就代表距离第一个query距离最近的九张图片的索引（这九张图片来自训练集），我们通过这个索引
        #就可以获知这九张图片的具体的类别（如果我们已知了query的类别，既可以据此来计算mAP），
        # 以及对应的存储路径（在图像检索系统中进行最后的检索结果的输出）
        print(result[0:5])
        print(result.shape)
        query_result_labels=np.zeros(result.shape)
        h,w=result.shape
        for i in range(h):
            for j in range(w):
                query_result_labels[i,j]=dataset_labels[result[i,j]]
                # query_result_labels[i,j]=test_labels[result[i,j]]

        print(query_result_labels[0:5])


        # query_result_paths=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths.append([])
        #     for j in range(w):
        #         query_result_paths[i].append(train_paths[result[i,j]])



        result_top_1=result[:,0:1]
        print(result_top_1.shape)
        result_top_1_labels=query_result_labels[:,0:1]
        print(result_top_1_labels.shape)

        ###################################################33
        result_top_2=result[:,0:2]
        print(result_top_2.shape)
        result_top_2_labels=query_result_labels[:,0:2]
        print(result_top_2_labels.shape)


        result_top_4=result[:,0:4]
        print(result_top_4.shape)
        result_top_4_labels=query_result_labels[:,0:4]
        print(result_top_4_labels.shape)


        result_top_8=result[:,0:8]
        print(result_top_8.shape)
        result_top_8_labels=query_result_labels[:,0:8]
        print(result_top_8_labels.shape)

        result_top_16=result[:,0:16]
        print(result_top_16.shape)
        result_top_16_labels=query_result_labels[:,0:16]
        print(result_top_16_labels.shape)

        result_top_32=result[:,0:32]
        print(result_top_32.shape)
        result_top_32_labels=query_result_labels[:,0:32]
        print(result_top_32_labels.shape)
        #####################################################
        # query_result_paths_top_1=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths_top_1.append([])
        #     for j in range(1):
        #         query_result_paths_top_1[i].append(train_paths[result[i,j]])

        result_top_5=result[:,0:5]
        print(result_top_5.shape)
        result_top_5_labels=query_result_labels[:,0:5]
        print(result_top_5_labels[0:10,:])
        print(result_top_5_labels.shape)
        # query_result_paths_top_5=[]
        # h,w=result.shape
        # for i in range(h):
        #     query_result_paths_top_5.append([])
        #     for j in range(5):
        #         query_result_paths_top_5[i].append(train_paths[result[i,j]])
        result_top_10=result[:,0:10]
        result_top_10_labels=query_result_labels[:,0:10]

        result_top_15=result[:,0:15]
        result_top_15_labels=query_result_labels[:,0:15]

        result_top_20=result[:,0:20]
        result_top_20_labels=query_result_labels[:,0:20]


        result_top_30=result[:,0:30]
        result_top_30_labels=query_result_labels[:,0:30]

        result_top_40=result[:,0:40]
        result_top_40_labels=query_result_labels[:,0:40]

        result_top_50=result[:,0:50]
        result_top_50_labels=query_result_labels[:,0:50]

        result_top_60=result[:,0:60]
        result_top_60_labels=query_result_labels[:,0:60]

        result_top_70=result[:,0:70]
        result_top_70_labels=query_result_labels[:,0:70]

        result_top_80=result[:,0:80]
        result_top_80_labels=query_result_labels[:,0:80]

        result_top_90=result[:,0:90]
        result_top_90_labels=query_result_labels[:,0:90]

        result_top_100=result[:,0:100]
        result_top_100_labels=query_result_labels[:,0:100]

        result_top_200=result[:,0:200]
        result_top_200_labels=query_result_labels[:,0:200]

        # result_top_500=result[:,1:501]
        # result_top_500_labels=query_result_labels[:,1:501]

        def compute_mAP(querys_results,querys_labels,querys_results_labels):
            '''query_results:[5794,9]的array
                query_results_labels:[5794,9]的array
                query_labels:[5794,]的list
                   pos_list:数据库中与查询图像相似的结果'''
            querys_labels=np.array(querys_labels)
            one_precision = []
            intersect_size = 0
            precision_all=[]
            for i in range(querys_results.shape[0]):#5794
                intersect_size = 0
                one_precision = []
                for j in range(querys_results.shape[1]):#9
                    if querys_results_labels[i,j] == querys_labels[i]:
                        intersect_size =intersect_size + 1
                        # precision = intersect_size - 1 / (j)
                        precision = intersect_size / (j + 1 )

                        one_precision.append(precision)
                if len(one_precision)==0:
                    precision_all.append(0)
                else:
                    # precision_all.append(np.mean(np.array(one_precision)).tolist())
                    precision_all.append(np.mean(np.array(one_precision)))

            return np.mean(np.array(precision_all))


        #Recall@K
        def compute_mAP_new(querys_results,querys_labels,querys_results_labels):
            '''query_results:[5794,9]的array
                query_results_labels:[5794,9]的array
                query_labels:[5794,]的list
                   pos_list:数据库中与查询图像相似的结果'''
            querys_labels=np.array(querys_labels)
            score=0
            for i in range(querys_results.shape[0]):#5794
                if querys_labels[i].tolist() in  querys_results_labels[i,:].tolist():
                    score=score+1

            return  score/querys_results.shape[0]


        print("top_1 mAP:")
        top_1=compute_mAP(result_top_1,test_labels,result_top_1_labels)
        print(top_1)

        print("top_5 mAP:")
        top_5=compute_mAP(result_top_5,test_labels,result_top_5_labels)
        print(top_5)

        print("top_10 mAP:")
        top_10=compute_mAP(result_top_10,test_labels,result_top_10_labels)
        print(top_10)

        print("top_15 mAP:")
        top_15=compute_mAP(result_top_15,test_labels,result_top_15_labels)
        print(top_15)

        print("top_20 mAP:")
        top_20=compute_mAP(result_top_20,test_labels,result_top_20_labels)
        print(top_20)

        print("top_30 mAP:")
        top_30=compute_mAP(result_top_30,test_labels,result_top_30_labels)
        print(top_30)

        print("top_40 mAP:")
        top_40=compute_mAP(result_top_40,test_labels,result_top_40_labels)
        print(top_40)

        print("top_50 mAP:")
        top_50=compute_mAP(result_top_50,test_labels,result_top_50_labels)
        print(top_50)

        print("top_60 mAP:")
        top_60=compute_mAP(result_top_60,test_labels,result_top_60_labels)
        print(top_60)

        print("top_70 mAP:")
        top_70=compute_mAP(result_top_70,test_labels,result_top_70_labels)
        print(top_70)

        print("top_80 mAP:")
        top_80=compute_mAP(result_top_80,test_labels,result_top_80_labels)
        print(top_80)

        print("top_90 mAP:")
        top_90=compute_mAP(result_top_90,test_labels,result_top_90_labels)
        print(top_90)

        print("top_100 mAP:")
        top_100=compute_mAP(result_top_100,test_labels,result_top_100_labels)
        print(top_100)

        print("top_200 mAP:")
        top_200=compute_mAP(result_top_200,test_labels,result_top_200_labels)
        print(top_200)

        # print("top_500 mAP:")
        # print(compute_mAP(result_top_500,test_labels,result_top_500_labels))
        print("*-*-*-*-*-*--*-*-*-*-*--*-*--*--*--*-*--*--*--*--*--*-*-*---*-*-*-")
        # print("top_1 mAP_new:")
        # print(compute_mAP_new(result_top_1,test_labels,result_top_1_labels))
        # print("top_5 mAP_new:")
        # print(compute_mAP_new(result_top_5,test_labels,result_top_5_labels))
        print("Recall@1:")
        R_1=compute_mAP_new(result_top_1,test_labels,result_top_1_labels)
        print(R_1)
        print("Recall@2:")
        R_2=compute_mAP_new(result_top_2,test_labels,result_top_2_labels)
        print(R_2)
        print("Recall@4:")
        R_4=compute_mAP_new(result_top_4,test_labels,result_top_4_labels)
        print(R_4)
        print("Recall@8:")
        R_8=compute_mAP_new(result_top_8,test_labels,result_top_8_labels)
        print(R_8)
        print("Recall@16:")
        R_16=compute_mAP_new(result_top_16,test_labels,result_top_16_labels)
        print(R_16)
        print("Recall@32:")
        R_32=compute_mAP_new(result_top_32,test_labels,result_top_32_labels)
        print(R_32)

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        csv_writer = csv.writer(f)
        if l==0:
            csv_writer.writerow(["top_1","top_5","top_10","top_15","top_20","top_30","top_40","top_50","top_60","top_70","top_80","top_90","top_100","top_200",'R@1','R@2','R@4','R@8','R@16','R@32'])
        csv_writer.writerow([top_1,top_5,top_10,top_15,top_20,top_30,top_40,top_50,top_60,top_70,top_80,top_90,top_100,top_200,R_1,R_2,R_4,R_8,R_16,R_32])

        f.close()


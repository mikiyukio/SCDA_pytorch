from skimage.measure import label
import numpy as np

def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of a binary image
    Parameters:
    ---
    bw_img: ndarray
        binary image
	Returns:
	---
	lcc: ndarray
		largest connect component.
    Example:
    ---
        # >>> lcc = largestConnectComponent(bw_img)
    '''
    # labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)

    # plt.figure(), plt.imshow(labeled_img, 'gray')

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc


def largestConnectComponent_many(bw_img, ):
    '''
    compute largest Connect component of a binary image
    Parameters:
    ---
    bw_img: ndarray
        binary image
	Returns:
	---
	lcc: ndarray
		largest connect component.
    Example:
    ---
        # >>> lcc = largestConnectComponent(bw_img)
    '''
    # labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)

    # plt.figure(), plt.imshow(labeled_img, 'gray')

    # max_label = 0
    # max_num = 0
    # for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
    #     if np.sum(labeled_img == i) > max_num:
    #         max_num = np.sum(labeled_img == i)
    #         max_label = i
    # lcc = (labeled_img == max_label)
    lcc=[]
    nums=[]
    for i in range(1, num + 1):
        lcc.append((labeled_img == i)+0)
        nums.append(np.sum(labeled_img == i))
    nums=np.array(nums)
    sorted_num = -np.sort(-nums)  # 0指代的是batch_size为1的情况
    # print(sorted_output)
    sorted_num_index = np.argsort(-nums)
    sorted_lcc=[]
    for i in range(len(sorted_num_index)):
        sorted_lcc.append(lcc[sorted_num_index[i]])
    return sorted_lcc


def largestConnectComponent_heat(bw_img,bw_img_heat ):
    '''
    compute largest Connect component of a binary image
    Parameters:
    ---
    bw_img: ndarray
        binary image
    bw_img_heat: ndarray
        heat_map:bw_img=heat_map>np.mean(heat_map)
	Returns:
	---
	lcc: ndarray
		selected component . but maybe not the largest connect component
    Example:
    ---
        # >>> lcc = largestConnectComponent(bw_img,,heatmap)
    '''
    # labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)

    # plt.figure(), plt.imshow(labeled_img, 'gray')

    # max_label = 0
    # max_num = 0
    # for i in range(1, num + 1):  # 这里从1开始，防止将背景设置为最大连通域
    #     if np.sum(labeled_img == i) > max_num:
    #         max_num = np.sum(labeled_img == i)
    #         max_label = i
    # lcc = (labeled_img == max_label)
    lcc=[]
    nums=[]
    lcc_heat=[]
    # lcc_heat_mean=[]
    lcc_heat_max=[]

    for i in range(1, num + 1):
        lcc.append((labeled_img == i)+0)
        nums.append(np.sum(labeled_img == i))
        lcc_heat.append(bw_img_heat*lcc[i-1])
        # lcc_heat_mean.append(np.sum(bw_img_heat*lcc[i-1])/np.sum(lcc[i-1]))
        lcc_heat_max.append(np.max(bw_img_heat*lcc[i-1]))
    nums=np.array(nums)
    sorted_num = -np.sort(-nums)
    # print(sorted_output)
    sorted_num_index = np.argsort(-nums)
    sorted_lcc=[]
    sorted_lcc_heat=[]
    # sorted_lcc_heat_mean=[]
    sorted_lcc_heat_max=[]

    for i in range(len(sorted_num_index)):
        sorted_lcc.append(lcc[sorted_num_index[i]])
        sorted_lcc_heat.append(lcc_heat[sorted_num_index[i]])
        # sorted_lcc_heat_mean.append(lcc_heat_mean[sorted_num_index[i]])
        sorted_lcc_heat_max.append(lcc_heat_max[sorted_num_index[i]])

    # flag=(sorted_lcc_heat_mean>=np.mean(sorted_lcc_heat_mean))+0
    flag_1=(sorted_lcc_heat_max>=np.mean(sorted_lcc_heat_max))+0
    #还是flag_1更为合适，因为flag对应的mean方式
    #因为假设有这样一种情况，热图有两个子区域；这两个子区域的最大值是相同的，
    #但是其中一个子区域较大，这个子区域中有很多较小的值，那样这两个子区域的平均值就会相差很多，
    #flag就无法将两个区域同时筛选出来。这时就需要max

    # out=np.zeros_like(bw_img)
    # for i in range(len(flag)):
    #     out=out+sorted_lcc[i]*flag[i]
    out_1=np.zeros_like(bw_img)
    for i in range(len(flag_1)):
        out_1=out_1+sorted_lcc[i]*flag_1[i]

    # return out,out_1
    return out_1


# x_heat=np.array([[1,0.4,0.3,0.2,0.1],[0.6,1,8,8,8],[0.09,0.4,1,1,8],[8,0,0,1,1]])
# # #是否需要进行孤立离群值的检测呢
# # #，我觉得是没有必要的，因为heatmap已经是归一化了的
# x=np.array([[1,0,0,0,0],[0,1,1,1,1],[0,0,1,1,1],[1,0,0,1,1]])
# print('highlight')
# print(x)
# print("heatmap")
# print(x_heat)
# b=largestConnectComponent_heat(x,x_heat)
# # print('mean')
# # print(a)
# print('max')
# print(b)
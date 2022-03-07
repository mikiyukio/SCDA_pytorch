from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import json
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be loaded.", default='checkpoint_3999_8.4847_cos.pth')
args = parser.parse_args()
# 传入图像的embedding特征和对应的图像的名字

test_paths_name=join(os.path.abspath('./datafile/test_paths.json'))
with open(test_paths_name) as miki:
    test_paths = json.load(miki)

# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/',args.model_name.replace('.', '_'),'embedding_max_avg.json')
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/',args.model_name.replace('.', '_'),'embedding_max_avg.json')
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/',args.model_name.replace('.', '_'),'embedding_max_avg.json')
# filename=join(os.path.abspath(os.path.dirname(os.getcwd())),'./datafile/',args.model_name.replace('.', '_'),'embedding_max_avg.json')
filename=join(os.path.abspath(os.path.dirname(os.getcwd())),"SCDA_for_LL/datafile/final_representation_4096.json")

with open(filename) as f_obj:
    final_features=json.load(f_obj)
test_data=final_features['test']

print(len(test_paths))
print(np.array(test_data).shape)



def draw_tsne(features, imgs):
    print(f">>> t-SNE fitting")
    # 初始化一个TSNE模型，这里的参数设置可以查看SKlearn的官网
    tsne = TSNE(n_components=2, init='pca', perplexity=30)
    Y = tsne.fit_transform(features)
    print(f"<<< fitting over")

    fig, ax = plt.subplots()
    fig.set_size_inches(21.6, 14.4)
    plt.axis('off')
    print(f">>> plotting images")
    imscatter(Y[:, 0], Y[:, 1], imgs, zoom=0.1, ax=ax)
    print(f"<<< plot over")
    plt.savefig(fname='figure.eps', format='eps')
    plt.show()


def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        # im = cv2.imread(image)
        miki_1 = Image.open(image).convert('RGB')
        im = cv2.cvtColor(np.asarray(miki_1), cv2.COLOR_RGB2BGR)
        # print(im.size())
        im = cv2.resize(im, (30, 30))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


draw_tsne(np.array(test_data),test_paths)

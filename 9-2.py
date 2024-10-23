# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:05:05 2024

@author: toshiki
"""

import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sklearn

#k-meansを使うためのインポート
from sklearn.cluster import KMeans
#データ取得のためのインポート
from sklearn.datasets import make_blobs

#サンプルデータの生成
#注意：make＿blobsは二つの値を返すため、一方は使用しない「＿」で受け取る
X,_ = make_blobs(random_state=10)

plt.title('9-2-2-1')
plt.scatter(X[:,0], X[:,1], color = 'black')

#KMeansクラスの初期化
kmeans = KMeans(init = 'random', n_clusters=3)

#クラスターの重心を計算
kmeans.fit(X)

#クラスター番号を予測
y_pred = kmeans.predict(X)

plt.title('9-2-2-2')
plt.scatter(X[:,0], X[:,1], color = 'black')
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:31:55 2024

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

#webからデータを取得したリ、zipファイルを扱うためのライブラリ
import requests, zipfile
import io

zip_file_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'

r = requests.get(zip_file_url, stream = True)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall(path = './chap9')

#対象データの読み込み
bank = pd.read_csv('./chap9/bank-full.csv', sep = ';')
bank.head()


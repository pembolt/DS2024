# データの加工・処理・分析ライブラリ
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 機械学習ライブラリ
import sklearn

# データ読み込み
import requests, zipfile
import io

# データを取得
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
res = requests.get(url).content

# 取得したデータをDataFrameオブジェクトとして読み込み
mushroom = pd.read_csv(io.StringIO(res.decode('utf-8')), header=None, sep=',')

# カラム名の設定
mushroom.columns = ['classes', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                    'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape',
                    'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring',
                    'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color',
                    'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']

# 先頭5行を表示
print(mushroom.head())


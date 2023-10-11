import zipfile
import os
import subprocess
import csv
import numpy as np
import json
import pickle
import math
import sklearn
import collections
import pandas as pd
from collections import Counter
from sklearn.svm import SVC
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

def df_se(data):
    c = collections.Counter([element for sublist in data for element in sublist])
    # 指定最小的出現次數閾值
    min_frequency = 10

    # 過濾出現次數大於或等於閾值的元素
    remaining_data = [
        [element for element in sublist if element and c[element] >= min_frequency]
        for sublist in data
    ]
    # 創建一個新的列表來存儲過濾後的元素
    filtered_elements = []

    # 過濾出現次數大於或等於閾值的元素並添加到新列表中
    for element in c:
        if element and c[element] >= min_frequency:
            filtered_elements.append(element)
    filtered_elements=np.array(filtered_elements)

    return remaining_data,filtered_elements

#slf
def slf(strings,max_length):
    bin_size = math.log(max_length) / 50
    slf_arr=[0]*50
    for string in strings:
        bin_index = int(math.floor(math.log(len(string)) / bin_size))
        slf_arr[bin_index-1] += 1
    slf_arr=np.array(slf_arr)
    return slf_arr

#psi
def psi(remaining_data):
    #psi
    # 收集所有出現在整個列表中的字串
    all_strings = list(set(element for sublist in remaining_data for element in sublist))

    # 初始化 PSI 矩陣
    psi_matrix = np.zeros((len(remaining_data), len(all_strings)), dtype=int)

    # 創建一個字串到索引的映射
    string_to_index = {string: index for index, string in enumerate(all_strings)}
    idx=0
    # 填充 PSI 矩陣
    for i, row in enumerate(remaining_data):
        for string in row:
            if string in string_to_index:
                j = string_to_index[string]
                psi_matrix[i, j] = 1
    psi_vector=np.array(psi_matrix)
    return psi_vector

#dfrank
def DFrank(vector,label):
    #yi
    unique_categories, category_indices = np.unique(label[:,1], return_inverse=True)
    index_by_category = [np.where(label[:,1] == category)[0] for category in unique_categories]
    yi=[]
    for i in index_by_category:
        yi.append(np.sum(vector[i],axis=0))
    yi = np.transpose(yi)
    #hyi
    H_yi=[]
    for i in yi:
        p_yi_j=[]
        for j in i:
            p_yi_j.append(j/np.sum(i))
        hyi=np.array(p_yi_j*np.log(p_yi_j))
        H_yi.append(-np.nansum(hyi))
    #dfi,k
    unique_categories, category_indices = np.unique(label[:,0], return_inverse=True)
    index_by_category = [np.where(label[:,0] == category)[0] for category in unique_categories]
    df_i_k=[]
    for i in index_by_category:
        df_i_k.append(np.sum(vector[i],axis=0))
    df_i_k = np.transpose(df_i_k)
    #dfrank score
    Dfrank=[]
    for i in range(0,len(H_yi)):
        cat=[]
        Dfrank.append(H_yi[i]*df_i_k[i]/np.nansum(df_i_k[i]))
    Dfrank=np.array(Dfrank)
    return Dfrank

def se(df_vector,psi_vector,filtered_elements):
    # 創建一個空的列表，用於存儲每個類別中選擇的前 20000 個特徵的索引
    selected_feature_indices = []
    selected_features_names=[]
    # 遍歷每個類別
    for class_scores in df_vector.T:  # 使用 .T 轉置矩陣以遍歷每個類別的分數
        # 根據 DFrank 分數對特徵進行排序，取前 20000 個特徵的索引
        top_20000_indices = np.argsort(class_scores)[-20000:]
        selected_feature_indices.append(top_20000_indices)
        selected_features_names.append(filtered_elements[top_20000_indices])#20000*9個特徵的名字
    # 合併每個類別中選擇的特徵索引
    selected_indices = np.concatenate(selected_feature_indices)
    selected_names=np.concatenate(selected_features_names)
    # 從原始的 PSIVector 中選擇選擇的特徵
    selected_features = psi_vector[:, selected_indices]
    # selected_features 現在包含了每個類別中選擇的前 20000 個特徵

    unique_data, unique_indices = np.unique(selected_names, return_index=True)#找出重複的data
    sele_vector=selected_features[:,unique_indices]#往回找對應的欄位
    return unique_data,sele_vector

def rfe(sele_vector,label,unique_data):
    #rfe
    # 加載數據集
    traindata = np.array(sele_vector.tolist())
    y = np.array(label.tolist())
    traindata, y = shuffle(traindata, y, random_state=42)
    # 創建SVM模型
    svm_model = SVC(kernel='linear')
    # 訓練SVM模型   
    svm_model.fit(traindata, y[:,0])
    # 獲取特徵權重
    feature_weights = svm_model.coef_[0]

    # 根據權重給特徵排序，選擇前 2000 個特徵
    num_selected_features = 2000
    selected_features_indices = np.argsort(np.abs(feature_weights))[::-1][:num_selected_features]
    # 選擇特徵
    X_selected = sele_vector[:, selected_features_indices]
    X_selected_names=unique_data[selected_features_indices]#RFE篩出的string
    # 現在 X_selected 只包含選擇的特徵
    return X_selected,X_selected_names
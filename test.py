from object_oriented_detector import GraphTheoryDetector
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

clf_test = GraphTheoryDetector(
    model_name_or_path='svm.pkl', task_type='classification')
string = clf_test.extractFeature(fpath_1='/home/yishan/psdetector/testdata/')
vector = clf_test.vectorize(string)
result = clf_test.predict(vector)
print(result)

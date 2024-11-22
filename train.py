from object_oriented_detector import GraphTheoryDetector
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

clf = GraphTheoryDetector(model_name_or_path='svm.pkl',
                          task_type='classification', training=True)

feature, label = clf.extractFeature(fpath="/home/yishan/psd/dataset",
                                    labelpath='/home/yishan/psd/dataset.csv')
    
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)

clf.train(feature=X_train, label=y_train, model=svm.SVC(kernel='linear'))
clf.save_model()
y_pred= clf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

import json
import vet
import numpy as np
import pickle
import r2pipe
from joblib import load, dump
import joblib
from malwareDetector.detector import detector
import os
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class GraphTheoryDetector(detector):
    def __init__(self, model_name_or_path: str = '', task_type: str = 'detection', training: bool = False, max_length=783265) -> None:
        self.model_name_or_path = model_name_or_path
        self.task_type = task_type
        self.max_length = max_length
        self.training = training

        if training:
            self.model = None
            self.trained = False
            return

        if task_type != 'detection' and task_type != 'classification':
            raise ValueError(
                'task_type must be either detection or classification')

        # load model
        self.model = None
        if '/' in model_name_or_path or os.path.exists(model_name_or_path):
            # specify path to the model
            self.model = load(model_name_or_path)
            '''with open(model_name_or_path, 'rb') as pickle_file:
                self.model = pickle.load(pickle_file)'''

    def extractFeature(self, fpath_1: str = '', fpath_2: str = '', labelpath: str = '') -> np.array:
        '''
        Extract features from a binary file using the radare2 library.

        Args:
            fpath (str): The path to the file to be processed.

        Returns:
            np.data: A tw-dim string array.
        '''
        data = []
        filenum_1 = 0
        filenum_2 = 0
        for root, _, files in os.walk(fpath_1):
            file_list = [f for f in os.listdir(
                root) if os.path.isfile(os.path.join(root, f))]
            for file_name in file_list:
                if filenum_1 == 5000:
                    break
                file_path = os.path.join(root, file_name)
                r2 = r2pipe.open(file_path)
                result = r2.cmd("izzj")
                string = []
                string.append(file_name)
                json_obj = json.loads(result)
                for jsonfile in json_obj:
                    string.append(jsonfile["string"])
                data.append(string)
                filenum_1 = filenum_1+1
        for root, _, files in os.walk(fpath_2):
            file_list = [f for f in os.listdir(
                root) if os.path.isfile(os.path.join(root, f))]
            for file_name in file_list:
                if filenum_2 == 5000:
                    break
                file_path = os.path.join(root, file_name)
                r2 = r2pipe.open(file_path)
                result = r2.cmd("izzj")
                string = []
                string.append(file_name)
                json_obj = json.loads(result)
                for jsonfile in json_obj:
                    string.append(jsonfile["string"])
                data.append(string)
                filenum_2 = filenum_2+1
        # print(self.trained)
        if self.training == False:
            return data
        if self.training:
            if self.trained == False:
                dataset = pd.read_csv(labelpath)
                newdata = []
                label = []
                for i in data:
                    newdata.append(i[1:])
                    label.append(dataset[(dataset["filename"] == i[0])][[
                        'label', 'CPU Architecture']])
                label = np.array(label)
                label = label.reshape(len(newdata), 2)
                remaining_data, filtered_elements = vet.df_se(newdata)
                self.max_length = max(len(s) for s in remaining_data)
                slf_vector = []
                for string in remaining_data:
                    slf_vector.append(vet.slf(string, self.max_length))
                slf_vector = np.array(slf_vector)
                psi_vector = vet.psi(remaining_data)
                df_vector = vet.DFrank(psi_vector, label)
                unique_data, sele_vector = vet.se(
                    df_vector, psi_vector, filtered_elements)
                X_selected, X_selected_names = vet.rfe(
                    sele_vector, label, unique_data)
                feature = np.column_stack((X_selected, slf_vector))
                with open("filtered_elements.pickle", 'wb') as pickle_file:
                    pickle.dump(filtered_elements, pickle_file)
                with open("X_selected_names.pickle", 'wb') as pickle_file:
                    pickle.dump(X_selected_names, pickle_file)
                return np.array(feature), label[:, 0]
            if self.trained == True:
                return data

    def vectorize(self, data: np.array, labelpath: str = '') -> np.array:
        '''
        veterize string

        labelpath:the path of label to get Architecture and path

        Args:
        Returns:
        '''
        if not self.training or self.trained == True:
            with open("filtered_elements.pickle", 'rb') as pickle_file:
                filtered_elements = pickle.load(pickle_file)
            with open("X_selected_names.pickle", 'rb') as pickle_file:
                X_selected_names = pickle.load(pickle_file)
            feature = []
            for i in data:
                i = i[1:]
                matching_mask = np.isin(i, filtered_elements)
                matching_indices = np.where(matching_mask)[0]
                rdata = np.array(i)[matching_indices]
                psi_vecor = np.array(
                    [1 if item in rdata else 0 for item in X_selected_names], dtype=int)
                slf_vector = np.array(vet.slf(rdata, self.max_length))
                all_vector = np.concatenate((psi_vecor, slf_vector))
                feature.append(all_vector)
        return np.array(feature)

    def predict(self, feature_vector: np.array) -> np.array:
        return self.model.predict(feature_vector.tolist())

    def train(self, feature: np.array, label: np.array, model: object) -> None:
        print('Start trianing')
        start = time.time()
        self.model = model.fit(feature, label)
        end = time.time()
        print(f'Time cost: {end-start} sec')
        self.trained = True

    def save_model(self) -> None:
        dump(self.model, self.model_name_or_path)
        '''with open(self.model_name_or_path, 'wb') as pickle_file:
            pickle.dump(self.model, pickle_file)'''

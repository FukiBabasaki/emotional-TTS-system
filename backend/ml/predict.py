import os
import glob
import csv
import pandas as pd  
import numpy as np  
import pickle
import sys
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


class Predictor:
    def __init__(self, predictor, emotion, speaker='male2'):
        self.emotion = emotion
        self.predictor = predictor
        self.speaker = speaker


    def predict(self):
        predicted = os.path.relpath(os.path.join('data', 'predicted.csv'))
        #Initialise with columns
        with open(predicted, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['speaker', 'emotion', 'aa_predicted', 'ap_predicted', 'dur_acc_predicted', 'ton', 'toff', 'dur_phr_predicted', 'fmin_predicted'])
            
        # Get the input dataset.
        predictors = pd.read_csv(self.predictor)

        # Get set of required features
        feat = pd.read_csv(os.path.join('data', 'set_of_features.csv'), sep=r'\t', engine='python').columns

        #Dropping columns that are not required
        predictors = predictors.loc[:, feat]

        #Converting categorical data into numbers with Pandas and Scikit-learn
        predictors = pd.get_dummies(predictors)
        np.where(np.isnan(predictors))
        predictors = predictors.dropna(how='any')

        # Get models for the emotion
        rf_models, ada_models = self.getModels(self.emotion)

        predictions = list()

        for x in range(5):
            rf_predict = rf_models[x].predict(predictors) 
            ada_predict = ada_models[x].predict(predictors)

            predictions.append((rf_predict + ada_predict)/2)
        
        self.predictions = predictions

        self.append_to_csv(predicted)
        return predicted

    def getModels(self, emotion):
        """
        This function takes emotion as an input and returns
        lists of adaboost and random forest models in a folder.
        """
        pathToModels = os.path.join('ml', 'trained_models')

        rf_models = list()
        ada_models = list()

        for para in ['abs_aa', 'ap_new', 'dur_acc', 'dur_phr_new', 'fmin']:
            path_to_ada_model = os.path.join(pathToModels, 'ada_' + emotion + '_' + para + '.pickle')
            path_to_rf_model = os.path.join(pathToModels, 'rf_' + emotion + '_' + para + '.pickle')

            pickle_ada = open(path_to_ada_model, 'rb')
            pickle_rf = open(path_to_rf_model, 'rb')

            ada_models.append(pickle.load(pickle_ada))
            rf_models.append(pickle.load(pickle_rf))

        return rf_models, ada_models

    def append_to_csv(self, path):
        with open(path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            ton = 0
            toff= 0
            
            # number of phones in the sentence which is the number of rows in the csv file
            num = len(self.predictions[0])

            for i in range(num):
                aa_predicted = self.predictions[0][i]
                ap_predicted = self.predictions[1][i]
                dur_acc_predicted = self.predictions[2][i]
                dur_phr_predicted = self.predictions[3][i]
                fmin_predicted = self.predictions[4][i]
                csv_writer.writerow([self.speaker, self.emotion, aa_predicted, ap_predicted, dur_acc_predicted, ton, toff, dur_phr_predicted, fmin_predicted])
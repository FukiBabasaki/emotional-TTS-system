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

# Get the path to this python file.
cwd = os.path.dirname(os.path.abspath(__file__))

# Change the working directory
os.chdir(os.path.join('..', 'files'))

# Initialise csv file with columns
with open('predictions_for_all_robot_dialogs_withfeats.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['speaker', 'emotion', 'sentence', 'aa_predicted', 'ap_predicted', 'dur_acc_predicted', 'ton', 'toff', 'dur_phr_predicted', 'fmin_predicted'])

def main():
    # loop over all csv files.
    for csv in glob.glob("*.csv"):

        # Get the input dataset.
        dataset = pd.read_csv(csv)

        # Get the strings for each attribute
        basename = os.path.basename(csv)

        try:
            speaker,emotion,context,session = basename.split("_")
        except ValueError:
            # When this exception is thrown, the csv is not related to the fujisaki features.
            continue

        context,a = context.split("a")

        #Converting categorical data into numbers with Pandas and Scikit-learn
        dataset = pd.get_dummies(dataset)
        np.where(np.isnan(dataset))
        dataset = dataset.dropna(how='any')


        # Get models for the emotion
        rf_models, ada_models = getModels(emotion)

        predictions = list()

        for x in range(5):
            rf_predict = rf_models[x].predict(dataset) 
            ada_predict = ada_models[x].predict(dataset)

            predictions.append((rf_predict + ada_predict)/2)
            
        append_to_csv(predictions, speaker, emotion, context)
        


def getModels(emotion):
    """
    This function takes emotion as an input and returns
    lists of adaboost and random forest models in a folder.
    """
    pathToModels = os.path.join(cwd, 'modelfiles')

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

def append_to_csv(prediction_list,speaker, emotion, sentence):
    with open('predictions_for_all_robot_dialogs_withfeats.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        ton = 0
        toff= 0
        
        # number of phones in the sentence which is the number of rows in the csv file
        num = len(prediction_list[0])

        for i in range(num):
            aa_predicted = prediction_list[0][i]
            ap_predicted = prediction_list[1][i]
            dur_acc_predicted = prediction_list[2][i]
            dur_phr_predicted = prediction_list[3][i]
            fmin_predicted = prediction_list[4][i]
            csv_writer.writerow([speaker, emotion, sentence, aa_predicted, ap_predicted, dur_acc_predicted, ton, toff, dur_phr_predicted, fmin_predicted])
        
main()
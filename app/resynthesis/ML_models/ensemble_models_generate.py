# -*- coding: utf-8 -*-
"""
This file trains and saves models to predict fujisaki parameters.

@author: jjam194
""" 

import os
import pandas as pd  
import numpy as np  
import pickle
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


# Get the path to this file.
cwd = os.path.dirname(os.path.abspath(__file__))

# Get the input dataset.
dataset = pd.read_csv(os.path.join(cwd, 'fuji_feats_synthesised_emotional_male_with_sentence_number.csv'))  

#Converting categorical data into numbers with Pandas and Scikit-learn
# One-hot encode the data using pandas get_dummies
dataset = pd.get_dummies(dataset)
np.where(np.isnan(dataset))
dataset = dataset.dropna(how='any')

print('The shape of our features is:', dataset.shape)

#Preparing Data For Training - splitting data into 20% (3 sentences) and 80% (12 sentences)

all_sentence_numbers = list(range(1, 16)) 
test_sentence_numbers = [8,11,13]
train_sentence_numbers = list(set(all_sentence_numbers) - set(test_sentence_numbers))

#splitting data into training and testing
trainset = dataset[dataset['sentence_number'].isin(train_sentence_numbers)]
testset = dataset[dataset['sentence_number'].isin(test_sentence_numbers)]

#settings for hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


#Determine Performance Metrics

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mae = round(np.mean(errors), 2)
        
    mse1 = errors*errors
    mse = round(np.mean(mse1), 2)
        
    mape = 100 * (errors / (test_labels + predictions))
    accuracy = round(np.mean(100 - mape),2)
        
    corr = np.corrcoef(predictions, test_labels)[0, 1]
    corr = round(corr, 3)

    return mae,mse,accuracy, corr, predictions

def evaluate_avg_model(predictions_avg, test_labels):
    errors = abs(predictions_avg - test_labels)
    mae = round(np.mean(errors), 2)
        
    mse1 = errors*errors
    mse = round(np.mean(mse1), 2)
        
    mape = 100 * (errors / (test_labels + predictions_avg))
    accuracy = round(np.mean(100 - mape),2)
        
    corr = np.corrcoef(predictions_avg, test_labels)[0, 1]
    corr = round(corr, 3)
        
    return mae,mse,accuracy, corr

def save_model(model, model_type, emotion, para):
    """
    save_model takes model, model name, and emotion as inputs and
    output pickle file
    """

    with open(os.path.join('modelfiles',  model_type + '_' + emotion  + '_'+ para + '.pickle'), 'wb') as dump_var:
                pickle.dump(model, dump_var)


def create_model(para, emotion):
    """
    This function takes parameter name and emotion to create each model.
    Each model is saved in pickle files
    """
    if emotion == 'angry':
        trainset_emotion = trainset[trainset.emotion_angry == 1] 
        testset_emotion = testset[testset.emotion_angry == 1]
                
    elif emotion == 'anxious':
        trainset_emotion = trainset[trainset.emotion_anxious == 1] 
        testset_emotion = testset[testset.emotion_anxious == 1]

    elif emotion == 'apologetic':
        trainset_emotion = trainset[trainset.emotion_apologetic == 1] 
        testset_emotion = testset[testset.emotion_apologetic == 1]

    elif emotion == 'enthusiastic':
        trainset_emotion = trainset[trainset.emotion_enthusiastic == 1] 
        testset_emotion = testset[testset.emotion_enthusiastic == 1]
                
    elif emotion == 'excited':
        trainset_emotion = trainset[trainset.emotion_excited == 1] 
        testset_emotion = testset[testset.emotion_excited == 1]

    elif emotion == 'happy':
        trainset_emotion = trainset[trainset.emotion_happy == 1] 
        testset_emotion = testset[testset.emotion_happy == 1]
                
    elif emotion == 'neutral':
        trainset_emotion = trainset[trainset.emotion_neutral == 1] 
        testset_emotion = testset[testset.emotion_neutral == 1]

    elif emotion == 'pensive':
        trainset_emotion = trainset[trainset.emotion_pensive == 1] 
        testset_emotion = testset[testset.emotion_pensive == 1]

    elif emotion == 'sad':
        trainset_emotion = trainset[trainset.emotion_sad == 1] 
        testset_emotion = testset[testset.emotion_sad == 1]

                
    elif emotion == 'worried':
        trainset_emotion = trainset[trainset.emotion_worried == 1] 
        testset_emotion = testset[testset.emotion_worried == 1]

    else :
        trainset_emotion = None
        testset_emotion = None
        
    
    train_labels_emotion = np.array(trainset_emotion[para])

    # Remove the labels from the features
    # axis 1 refers to the columns
    trainset_emotion = trainset_emotion.drop(['sentence_number','fmin','abs_aa','ap_new','dur_phr_new','dur_acc', 'emotion_angry', 'emotion_anxious', 'emotion_apologetic', 'emotion_enthusiastic', 'emotion_excited', 'emotion_happy', 'emotion_pensive', 'emotion_sad', 'emotion_worried', 'emotion_neutral'], axis = 1)
    testset_emotion = testset_emotion.drop(['sentence_number','fmin','abs_aa','ap_new','dur_phr_new','dur_acc', 'emotion_angry', 'emotion_anxious', 'emotion_apologetic', 'emotion_enthusiastic', 'emotion_excited', 'emotion_happy', 'emotion_pensive', 'emotion_sad', 'emotion_worried', 'emotion_neutral'], axis = 1)

    # Also remove all unrequired dummy columns
    trainset_emotion = trainset_emotion.loc[:, :'words_to_next_punctuation']
    testset_emotion = testset_emotion.loc[:, :'words_to_next_punctuation']

    # Convert to numpy array
    train_features = np.array(trainset_emotion)
    test_features = np.array(testset_emotion)

    #hyperparameter and modelling tuning for RF

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(train_features, train_labels_emotion)

    #baseline model default features
    base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
    base_model.fit(train_features, train_labels_emotion)

    #hyperparameter tuning and modelling for adaboost
    crossvalidation=KFold(n_splits=10,shuffle=True,random_state=1)
    tree_regressor = None
    for depth in range (1,10):
            tree_regressor=tree.DecisionTreeRegressor(max_depth=depth,random_state=1)
            if tree_regressor.fit(trainset_emotion, train_labels_emotion).tree_.max_depth<depth:
                    break
    
    score=np.mean(cross_val_score(tree_regressor,trainset_emotion,train_labels_emotion,scoring='neg_mean_squared_error', cv=crossvalidation,n_jobs=1))

    ada=AdaBoostRegressor()
    search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'random_state':[1]}
    ada=GridSearchCV(estimator=ada,param_grid=search_grid,scoring='neg_mean_squared_error',n_jobs=1,cv=crossvalidation)
    ada.fit(trainset_emotion,train_labels_emotion)

    #hyperparameter tuned model
    best_random = rf_random.best_estimator_

    # save the models into pickle files
    save_model(best_random, 'rf', emotion, para)
    save_model(ada, 'ada', emotion, para)

    print('models for ' + emotion + ' saved as pkl files')


for para in ['abs_aa', 'ap_new', 'dur_acc', 'dur_phr_new', 'fmin'] :
    create_model(para, 'angry')

    create_model(para, 'anxious')

    create_model(para, 'apologetic')

    create_model(para, 'enthusiastic')

    create_model(para, 'excited')

    create_model(para, 'happy')

    create_model(para, 'neutral')

    create_model(para, 'pensive')

    create_model(para, 'sad')

    create_model(para, 'worried')

print("Program completed")

#predicting at sentence level#####################################################################
def predict(model, test_features):
    predictions = model.predict(test_features)
    
    return predictions

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:39:49 2019

#### Wifi Challenge
# Predict location

@author: Aline Barbosa Alves
"""
# Import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, \
                            classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import plotly.express as px
import numpy as np
from plotly.offline import plot
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

"""Import Data"""
# Save filepath to variable for easier access
training_data_path = '/home/aline/Documentos/Ubiqum/IoT Analytics/WifiLocationing/data/trainingData.csv'
validation_data_path = '/home/aline/Documentos/Ubiqum/IoT Analytics/WifiLocationing/data/validationData.csv'


# read the data and store data in DataFrame
training_data = pd.read_csv(training_data_path) 
validation_data = pd.read_csv(validation_data_path)

"""Get to know the data"""
# Print a summary of the data
training_data.describe()
validation_data.describe()

# Columns
training_data.columns
validation_data.columns

# Missing data
training_data.isnull().any().sum()
validation_data.isnull().any().sum()

"""Pre processing"""
# Feature selection
training = training_data.iloc[:,0:520]
validation = validation_data.iloc[:,0:520]

# Targets
latitude = training_data.LATITUDE
longitude = training_data.LONGITUDE
floor = training_data.FLOOR
building = training_data.BUILDINGID

# Set floor and building as categorical data
floor = pd.Series(floor, dtype="category")
building = pd.Series(building, dtype="category")

"""Cross-validation"""
def get_score_rfr(n_estimators):
    my_pipeline = Pipeline(steps=[
#        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, 
                                        random_state=0))
    ])
#    scores = []
    scores = -1 * cross_val_score(my_pipeline, training, latitude,
                                      cv=3,
                                      scoring='neg_mean_absolute_error')
    return scores.mean()

number_of_trees = [50, 100, 150, 200, 250, 300, 350, 400] # 200

results = {}

for i in number_of_trees:
    results.update({i:get_score_rfr(i)})

"""Normalize"""
# Transform data - Put 'not found' signals (100) as the min
for i in range(0,len(training.columns)):
    if (training.iloc[:,[i]].min() == 100).bool():
        training.iloc[:,[i]] = (training.min()).min()
    else:
        training.iloc[:,[i]] = \
        training.iloc[:,[i]].replace(100, training.iloc[:,[i]].min())

training.values

# Do the same with the validation data
for i in range(0,len(validation.columns)):
    if (validation.iloc[:,[i]].min() == 100).bool():
        validation.iloc[:,[i]] = (validation.min()).min()
    else:
        validation.iloc[:,[i]] = \
        validation.iloc[:,[i]].replace(100, validation.iloc[:,[i]].min())

validation.values


for i in range(0,len(training.columns)):
    if (training.iloc[:,[i]].min()==training.iloc[:,[i]].max()).bool():
        training.iloc[:,[i]] = 0
    else:
        training.iloc[:,[i]] = \
        (training.iloc[:,[i]] - training.iloc[:,[i]].min())/ \
        (training.iloc[:,[i]].max() - training.iloc[:,[i]].min())

# Create function to transform putting 'not found' signals (100) as 
# the min and normalize data
def normalize(df):
    for i in range(0,len(df.columns)):
        # If its only 'not found', put zero
        if (df.iloc[:,[i]].min()==df.iloc[:,[i]].max()).bool():
            df.iloc[:,[i]] = 0
        else:
            # Replace 'not found' with the min
            df.iloc[:,[i]] = \
            df.iloc[:,[i]].replace(100, df.iloc[:,[i]].min())

            # Normalize data            
            if (df.iloc[:,[i]].min()==df.iloc[:,[i]].max()).bool():
                df.iloc[:,[i]] = 0
            else:
                df.iloc[:,[i]] = \
                (df.iloc[:,[i]] - df.iloc[:,[i]].min())/ \
                (df.iloc[:,[i]].max() - df.iloc[:,[i]].min())
    return df

# Apply function on training and validation data
training_normalized = normalize(training)
validation_normalized = normalize(validation)

"""Plots"""
scatter_training = px.scatter_3d(training_data, 
                    x="LONGITUDE", 
                    y="LATITUDE", 
                    z="FLOOR", 
                    color="BUILDINGID")
plot(scatter_training)

scatter_validation = px.scatter_3d(validation_data, 
                    x="LONGITUDE", 
                    y="LATITUDE", 
                    z="FLOOR", 
                    color="BUILDINGID")
plot(scatter_validation)

"""Standardize"""
#Robust Scaler
r_scaler = preprocessing.RobustScaler()
training_rscaler = r_scaler.fit_transform(training)

training_rscaler = pd.DataFrame(training_rscaler, 
                                columns=list(training.columns))

# Standard Scaler
s_scaler = preprocessing.StandardScaler()
training_sscaler = s_scaler.fit_transform(training)

training_sscaler = pd.DataFrame(training_sscaler, 
                                columns=list(training.columns))

# Function to standardize
def standardize(df):
    for i in range(0,len(df)):
        # Replace 'not found' with the min
        df.iloc[i,:] = \
        df.iloc[i,:].replace(100, df.iloc[i,:].min())

    return df.sub(df.mean(1), axis=0).div(df.std(1), axis=0)

# Apply the function
training_sscaler_row = standardize(training)
# Replace NaN with 0
training_row_replace = training_sscaler_row.replace(np.nan,0)
# Replace NaN with the min
training_row_replace_min = training_sscaler_row.replace(np.nan,training_sscaler_row.min())
# Check the data
training_row_replace_min.isnull().any().sum()
len(training_row_replace_min)
# Check index from rows with NaN
index_training = training_sscaler_row.iloc[:,0].index[training_sscaler_row.iloc[:,0].apply(np.isnan)]
len(index_training)
training_data.iloc[index_training,:]
# Drop rows with NaN
training_row_drop = training_sscaler_row.drop(index_training, axis=0)

# Apply the function
validation_sscaler_row = standardize(validation)
# Replace NaN with 0
validation_row_replace = validation_sscaler_row.replace(np.nan,0)
# Replace NaN with the min
validation_row_replace_min = validation_sscaler_row.replace(np.nan,validation_sscaler_row.min())
# Check the data
validation_row_replace_min.isnull().any().sum()
len(validation_row_replace_min)
# Check index from rows with NaN
index_validation = validation_sscaler_row.iloc[:,0].index[validation_sscaler_row.iloc[:,0].apply(np.isnan)]
len(index_validation)
validation_data.iloc[index_validation,:]
# Drop rows with NaN
validation_row_drop = validation_sscaler_row.drop(index_validation, axis=0)

# Drop same rows from the original data
building_drop = building.drop(index_training, axis=0)
floor_drop = floor.drop(index_training, axis=0)
latitude_drop = latitude.drop(index_training, axis=0)
longitude_drop = longitude.drop(index_training, axis=0)
validation_data_drop = validation_data.drop(index_validation, axis=0)

"""Models"""
#Random Forest Classifier - Cross validation
def get_score_rfc(n_estimators):
    my_pipeline = Pipeline(steps=[
        ('model', RandomForestClassifier(n_estimators=n_estimators))
    ])
    scores = cross_val_score(my_pipeline, training_sscaler_row, building,
                                      cv=3,
                                      scoring='accuracy')
    return scores.mean()

results = {}

for i in number_of_trees:
    results.update({i:get_score_rfc(i)})

#Random Forest Classifier - Building
rf_classifier = RandomForestClassifier(n_estimators=400)
rf_classifier.fit(training_row_replace,building)
rf_classifier_predictions = rf_classifier.predict(validation_row_replace)
accuracy_score(validation_data.BUILDINGID, rf_classifier_predictions)
confusion_matrix(validation_data.BUILDINGID, rf_classifier_predictions)
classification_report(validation_data.BUILDINGID, rf_classifier_predictions)

#SVM SVC - Building
svc_model = SVC()
svc_model.fit(training_row_replace, building)
svc_predictions = svc_model.predict(validation_row_replace)
accuracy_score(validation_data.BUILDINGID, svc_predictions)
confusion_matrix(validation_data.BUILDINGID, svc_predictions)
classification_report(validation_data.BUILDINGID, svc_predictions)

# KNN - Building
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(training_row_replace, building)
knn_predictions = knn_model.predict(validation_row_replace)
accuracy_score(validation_data.BUILDINGID, knn_predictions)

# Random Forest Classifier - Floor
# Create new columns with the building in both data set
validation_building = validation_row_replace
validation_building['BUILDINGID'] = svc_predictions 
training_building = training_row_replace
training_building['BUILDINGID'] = building 
# Fit the model
rf_classifier.fit(training_building,floor)
rfc_predictions_floor = rf_classifier.predict(validation_building)
accuracy_score(validation_data.FLOOR, rfc_predictions_floor)

#SVM SVC - Floor
svc_model.fit(training_building,floor)
svc_predictions_floor = svc_model.predict(validation_building)
accuracy_score(validation_data.FLOOR, svc_predictions_floor)

# KNN - Floor
knn_model.fit(training_building,floor)
knn_predictions_floor = knn_model.predict(validation_building)
accuracy_score(validation_data.FLOOR, knn_predictions_floor)

""""""
t = training_floor.iloc[:,0:520]
t[t < 0] = 0
t['BUILDINGID'] = building
t['FLOOR'] = floor 

v = validation_floor.iloc[:,0:520]
v[v < 0] = 0
v['BUILDINGID'] = svc_predictions
v['FLOOR'] = svc_predictions_floor 
""""""
#Random Forest Regressor - Latitude
# Create new columns with the building in both data set
validation_floor = validation_building
validation_floor['FLOOR'] = svc_predictions_floor 
training_floor = training_building
training_floor['FLOOR'] = floor 
# Fit the model
rf_model = RandomForestRegressor(random_state = 2)
rf_model.fit(t,latitude)
rf_predictions_latitude = rf_model.predict(v)
mean_absolute_error(validation_data.LATITUDE, rf_predictions_latitude)

#XGBoost - Latitude
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_model.fit(training_floor, latitude, 
              early_stopping_rounds=5,
              eval_set=[(validation_floor, validation_data.LATITUDE)],
              verbose=False)
xgb_predictions_latitude = xgb_model.predict(validation_floor)
mean_absolute_error(validation_data.LATITUDE, xgb_predictions_latitude)

#Random Forest Regressor - Longitude
# Create new columns with the building in both data set
validation_latitude = validation_floor
validation_latitude['LATITUDE'] = rf_predictions_latitude 
training_latitude = training_floor
training_latitude['LATITUDE'] = latitude 
# Fit the model
rf_model.fit(training_latitude, longitude)
rf_predictions_longitude = rf_model.predict(validation_latitude)
mean_absolute_error(validation_data.LONGITUDE, rf_predictions_longitude)

#XGBoost - Longitude
xgb_model.fit(training_latitude, longitude, 
              early_stopping_rounds=5,
              eval_set=[(validation_latitude, validation_data.LONGITUDE)],
              verbose=False)
xgb_predictions_longitude = xgb_model.predict(validation_latitude)
mean_absolute_error(validation_data.LONGITUDE, xgb_predictions_longitude)

# Try to know H2O to parallelize the processing
# Try to learn about PCA
# pipeline - ok
# Cross-validation - ok
# Boosted gradient - ok
# hyper parameters and random_state
# normalize WAPs (remove 100) - ok
# standarlize WAPs (remove 100) per row - ok
# pick the high values for each row
# Use floor and building as categorical then use RandomForestClassifier - ok
# data-leakage
# plot errors
# api google maps to fit the predictions
# check errors with other features (device, user, etc)
# try new models (knn, svm, etc) - ok
# try neural network
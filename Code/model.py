"""
An initial XGBoost model based on
https://github.com/dmlc/xgboost/blob/master/demo/guide-python/multioutput_regression.py

"""
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
from utils import loadModelData, loadRSSModelData, loadCovariateData, loadSections, pointToSection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle, utm
from multilat import gps_solve
from kmlInterface import HabitatMap, Habitat
from plot import plotGridWithPoints
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf 

def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()], optimizer='adam')
	return model

def simpleKeras(useCovariate: bool =False, sectionThreshold: int =50, plotError: bool = True, isAllDists: bool =False,isRSS: bool = False):
    X, y = loadRSSModelData(month="June",includeCovariatePred=useCovariate, isAllDists=isAllDists, isRSS=isRSS)
    n_inputs = X.shape[1]
    n_outputs = y.shape[1]

    results = []
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
    for train_ix, test_ix in cv.split(X):
        # prepare data
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]
        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        model.fit(X_train, y_train, verbose=1, epochs=10000)
        # evaluate model on test set
        vals = model.evaluate(X_test, y_test, verbose=0)
        # store result
        print(vals)
        results.append(vals)
    print(results)

def rmseModel(useCovariate: bool =False, sectionThreshold: int =50, isErrorData: bool =True, useErrorBars: bool = False, useColorScale: bool = True, plotError: bool = True, isAllDists: bool =False,isRSS: bool = False, useThresholdError: bool = False, sameNodeColor: bool = False):
    """ Root mean squared error XGBoost trained on the june data 
            useCovariate : bool, refers to whether or not to include the predicted habitat with determining
            the location
            sectionThreshold: int, refers to the dist in m threshold to include where the data appears in which sections 
            isErrorData : bool, refers to the type of data that will be taken as input (will either be predicting the 
            offset of the multilat or all the locations)
    """
    # Load in the data by the inputted boolean 
    if isErrorData:
        X, y = loadModelData(month="June",threshold=-96, verbose=False, includeCovariatePred=useCovariate)
    else:
        X, y = loadRSSModelData(month="June",includeCovariatePred=useCovariate, isAllDists=isAllDists, isRSS=isRSS)
    startIdx = 0
    if useCovariate:
        startIdx = 1
    
    # Split the data into train, test, and validation sets
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=0.8, random_state=101)
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining,y_remaining, test_size=0.5)

    # Train a regressor on it
    reg = xgb.XGBRegressor(tree_method="hist", n_estimators=32)
    reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    # predict the whole dataset
    yPred = reg.predict(X)
    # find the average error for the whole dataset
    avgError = [0,0]

    for idx in range(len(yPred)):
        gt = np.array([np.float64(X[idx][0]+y[idx][0]), np.float64(X[idx][1]+y[idx][1])])
        predicted = np.array([np.float64(yPred[idx][0]+X[idx][0]),np.float64(yPred[idx][1]+X[idx][1])])
        dist = np.linalg.norm(predicted-gt)
        avgError[0]+=dist
        avgError[1]+=1

    # print the result
    print(len(X_valid),len(X_test),len(X_train),len(X))
    print("The total average error is {}m".format(avgError[0]/avgError[1]))

    # Now we evaluate on the test set
    # predict on unseen test set
    yTestPred = reg.predict(X_test)
    
    # we want to find all the error values as well
    # as their respective locations and habitats
    allErrors = []
    freqErrorSections = {}
    freqErrorHabitats = {}
    # want to load in the section nodes
    grid, sections, nodes = loadSections()
    # load in the habitat
    habitatMap = HabitatMap()
    # determine the different node locations in the order derived from list(nodes.keys()) - order matters
    nodeLocs = []
    for node in list(nodes.keys()):
        nodeLocs.append([nodes[node]["NodeUTMx"],nodes[node]["NodeUTMy"]])
    
    # errorLocs will contain all the locations of the errors to plot
    errorLocs = []
    # errorDirections will have the error in both the x and y direction
    errorDirections = []

    # assign the section ids and outside grid value
    for i in range(len(sections)):
        freqErrorSections[i] = []
    freqErrorSections[-1] = []

    # assign the habitat values and outside grid value
    habitat_titles = habitatMap.getHabitats()
    for habitat_title in habitat_titles:
        freqErrorHabitats[habitat_title] = []
    freqErrorHabitats[-1] = []

    for idx in range(len(yTestPred)):
        gt = None
        predicted = None
        # determine the ground truth and predicted values according to the data provided to the model
        if isErrorData:
            gt = np.array([np.float64(X_test[idx][0]+y_test[idx][0]), np.float64(X_test[idx][1]+y_test[idx][1])])
            predicted = np.array([np.float64(yTestPred[idx][0]+X_test[idx][0]),np.float64(yTestPred[idx][1]+X_test[idx][1])])
        else:
            gt = gps_solve(y_test[idx],np.array(list(nodeLocs)))
            predicted = gps_solve(yTestPred[idx],np.array(list(nodeLocs)))
        dist = np.linalg.norm(predicted-gt)
        errorDirections.append([predicted[0]-gt[0],predicted[1]-gt[1]])
        allErrors.append(dist)
        # if we're not using the threshold or if the distance is >= threshold, 
        # collect the section and habitat frequencies
        if not useThresholdError or dist>=sectionThreshold:
            # store the values in section & habitat and error location again based on the format of the data
            if isErrorData:
                sec = pointToSection(X_test[idx][0]+y_test[idx][0], X_test[idx][1]+y_test[idx][1], sections)
                errorLocs.append([X_test[idx][0]+y_test[idx][0], X_test[idx][1]+y_test[idx][1]])
                habIdx, habitatName = habitatMap.whichHabitat(X_test[idx][0]+y_test[idx][0], X_test[idx][1]+y_test[idx][1])
            else:
                sec = pointToSection(gt[0],gt[1], sections)
                errorLocs.append([gt[0],gt[1]])
                habIdx, habitatName = habitatMap.whichHabitat(gt[0], gt[1])
            if habIdx!=-1:
                freqErrorHabitats[habitatName].append(dist)
            else:
                freqErrorHabitats[habIdx].append(dist)
            freqErrorSections[sec].append(dist)
    
    # Display the section frequency errors
    if useThresholdError:
        print("The test errors over {} m had the following section distribution: ".format(sectionThreshold))
    else:
        print("The test errors had the following section distribution: ")
    
    for secKey in freqErrorSections.keys():
        print("{} : {}".format(secKey,freqErrorSections[secKey]))
    # Display the habitat frequency errors
    if useThresholdError:
        print("The test errors over {} m had the following habitat distribution: ".format(sectionThreshold))
    else:
        print("The test errors had the following habitat distribution: ")
    
    for habKey in freqErrorHabitats.keys():
        print("{} : {}".format(habKey,freqErrorHabitats[habKey]))
    
    # And print the average test error
    print("The test average error is {}m with a maximum of {}".format(sum(allErrors)/len(allErrors), max(allErrors)))
    # determine whether to plot all the error locations
    if plotError:
        if useErrorBars:
            plotGridWithPoints(errorLocs,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=True, errors=errorDirections, sameNodeColor=sameNodeColor)
        elif useColorScale:
            plotGridWithPoints(errorLocs,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=False, colorScale = True, errors=allErrors, sameNodeColor=sameNodeColor)
        else:
            plotGridWithPoints(errorLocs,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=False, colorScale = False, errors=None, sameNodeColor=sameNodeColor)
    

def covariateTrain(saveModel=False):
    """ Creates a Classifier to determine the habitat given the signals in a particular location,
        based on march data"""
    # load in the data for the covariate training
    X, y = loadCovariateData()
    # Split the data for evluatoin
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    # Has hyperparameters to be tuned
    clf = RandomForestClassifier(bootstrap=True, random_state=101, oob_score=True, n_jobs=-1)
    #clf = xgb.XGBClassifier()
    # train on the data
    clf.fit(X_train,y_train)
    # predict on the test set
    y_pred = clf.predict(X_test)
    # show the classification metrics for the test data
    print("On all the test data")
    getClassificationMetrics(y_test, y_pred)
    # predict on all the data
    newYPred = clf.predict(X)
    # show the classification metrics for all the data
    print("On all the data")
    getClassificationMetrics(y, newYPred)

    # Save the model 
    if saveModel:
        pickle.dump(clf, open("models/covariateClf", 'wb'))
        print("Saved the model")
    
def getClassificationMetrics(y_test, y_pred):
    """ Prints some metrics for the results of classification predictions """
    # View the accuracy 
    print(accuracy_score(y_test, y_pred))
    # View confusion matrix for test data and predictions
    print(confusion_matrix(y_test, y_pred))
    # View the classification report for test data and predictions
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    rmseModel(useCovariate=True,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)
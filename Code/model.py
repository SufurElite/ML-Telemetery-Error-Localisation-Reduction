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
from plot import plotGridWithPoints

def rmseModel(useCovariate=False, sectionThreshold=50, isErrorData=True):
    """ Root mean squared error XGBoost trained on the june data"""
    nodeLocs = None
    y_vals = None
    if isErrorData:
        X, y = loadModelData(month="June",threshold=-96, verbose=False, includeCovariatePred=useCovariate)
    else:
        X, y = loadRSSModelData(month="June",includeCovariatePred=True)
    print(X.shape)
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=0.8, random_state=101)
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining,y_remaining, test_size=0.5)

    # Train a regressor on it
    reg = xgb.XGBRegressor(tree_method="hist", n_estimators=32)
    reg.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

    yPred = reg.predict(X)
    avgError = [0,0]

    for idx in range(len(yPred)):
        gt = np.array([np.float64(X[idx][0]+y[idx][0]), np.float64(X[idx][1]+y[idx][1])])
        predicted = np.array([np.float64(yPred[idx][0]+X[idx][0]),np.float64(yPred[idx][1]+X[idx][1])])
        dist = np.linalg.norm(predicted-gt)
        avgError[0]+=dist
        avgError[1]+=1

    yTestPred = reg.predict(X_test)
    allErrors = []
    freqErrorSections = {}
    grid, sections, nodes = loadSections()
    
    nodeLocs = []
    for node in list(nodes.keys()):
        nodeLocs.append([nodes[node]["NodeUTMx"],nodes[node]["NodeUTMy"]])

    freqErrorSections[-1] = []
    errorLocs = []
    print(len(X_valid),len(X_test),len(X_train),len(X))
    print("The total average error is {}m".format(avgError[0]/avgError[1]))

    for i in range(len(sections)):
        freqErrorSections[i] = []
    for idx in range(len(yTestPred)):
        gt = None
        predicted = None
        if isErrorData:
            gt = np.array([np.float64(X_test[idx][0]+y_test[idx][0]), np.float64(X_test[idx][1]+y_test[idx][1])])
            predicted = np.array([np.float64(yTestPred[idx][0]+X_test[idx][0]),np.float64(yTestPred[idx][1]+X_test[idx][1])])
        else:
            gt = gps_solve(y_test[idx],np.array(list(nodeLocs)))
            predicted = gps_solve(yTestPred[idx],np.array(list(nodeLocs)))
        dist = np.linalg.norm(predicted-gt)
        allErrors.append(dist)
        if dist>=sectionThreshold:
            if isErrorData:
                sec = pointToSection(X_test[idx][0]+y_test[idx][0], X_test[idx][1]+y_test[idx][1], sections)
                errorLocs.append([X_test[idx][0]+y_test[idx][0], X_test[idx][1]+y_test[idx][1]])
            else:
                sec = pointToSection(gt[0],gt[1], sections)
                errorLocs.append([gt[0],gt[1]])
            freqErrorSections[sec].append(dist)
            


    print("The Errors over {} m had the following section distribution: ".format(sectionThreshold))
    for secKey in freqErrorSections.keys():
        print("{} : {}".format(secKey,freqErrorSections[secKey]))
    print("The test average error is {}m with a maximum of {}".format(sum(allErrors)/len(allErrors), max(allErrors)))
    #plotGridWithPoints(errorLocs)
    

def covariateTrain(saveModel=False):
    """ Random Forest Classifier to determine the habitat given the signals in a particular location, \
        based on march data"""
    # https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56
    X, y, origX, orig_y = loadCovariateData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    print(X.shape)
    print(X_train.shape)
    print(X_test.shape)
    # Has hyperparameters to be tuned
    #clf = RandomForestClassifier(bootstrap=True, random_state=101, oob_score=True, n_jobs=-1)
    clf = xgb.XGBClassifier()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print("On all the test data")
    getClassificationMetrics(y_test, y_pred)
    if saveModel:
        pickle.dump(clf, open("models/covariateClf", 'wb'))
        print("Saved the model")
    newYPred = clf.predict(X)
    print("On all the data")
    getClassificationMetrics(y, newYPred)
    
def getClassificationMetrics(y_test, y_pred):
    """"""
    print(accuracy_score(y_test, y_pred))
    # View confusion matrix for test data and predictions
    print(confusion_matrix(y_test, y_pred))
    # View the classification report for test data and predictions
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    #rmseModel(False)
    rmseModel(useCovariate=False,isErrorData=True)

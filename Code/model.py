"""
An initial XGBoost model based on
https://github.com/dmlc/xgboost/blob/master/demo/guide-python/multioutput_regression.py

"""
import numpy as np
from matplotlib import pyplot as plt
import xgboost as xgb
from utils import loadModelData
from sklearn.model_selection import train_test_split

def rmseModel():
    """ Root mean squared error XGBoost trained on the june data"""
    X, y = loadModelData(month="June",threshold=-102, verbose=False)
    #XTest, YTest = loadModelData(month="March", threshold=-86, verbose=False)


    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=0.8, random_state=101)
    X_valid, X_test, y_valid, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5)
    #X_valid, X_test, y_valid, y_test = train_test_split(XTest,YTest, test_size=0.5)

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
    for idx in range(len(yTestPred)):
        gt = np.array([np.float64(X_test[idx][0]+y_test[idx][0]), np.float64(X_test[idx][1]+y_test[idx][1])])
        predicted = np.array([np.float64(yTestPred[idx][0]+X_test[idx][0]),np.float64(yTestPred[idx][1]+X_test[idx][1])])
        dist = np.linalg.norm(predicted-gt)
        allErrors.append(dist)

    print(len(X_valid),len(X_test),len(X_train),len(X))
    print("The total average error is {}m".format(avgError[0]/avgError[1]))
    print("The test average error is {}m with a maximum of {}".format(sum(allErrors)/len(allErrors), max(allErrors)))
    print(allErrors)

if __name__ == "__main__":
    rmseModel()

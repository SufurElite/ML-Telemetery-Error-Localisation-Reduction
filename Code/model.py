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
    X, y = loadModelData(month="June",threshold=-86, verbose=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    # Train a regressor on it
    reg = xgb.XGBRegressor(tree_method="hist", n_estimators=64)
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    yPred = reg.predict(X)
    avgError = [0,0]
    
    for idx in range(len(yPred)):
        gt = np.array([np.float64(X[idx][0]+y[idx][0]), np.float64(X[idx][1]+y[idx][1])])
        predicted = np.array([np.float64(yPred[idx][0]+X[idx][0]),np.float64(yPred[idx][1]+X[idx][1])])
        dist = np.linalg.norm(predicted-gt)
        avgError[0]+=dist
        avgError[1]+=1

    yTestPred = reg.predict(X_test)
    testError = [0,0]
    
    for idx in range(len(yTestPred)):
        gt = np.array([np.float64(X_test[idx][0]+y_test[idx][0]), np.float64(X_test[idx][1]+y_test[idx][1])])
        predicted = np.array([np.float64(yTestPred[idx][0]+X_test[idx][0]),np.float64(yTestPred[idx][1]+X_test[idx][1])])
        dist = np.linalg.norm(predicted-gt)
        testError[0]+=dist
        testError[1]+=1

    print("The total average error is {}m".format(avgError[0]/avgError[1]))
    print("The test average error is {}m".format(testError[0]/testError[1]))

if __name__ == "__main__":
    rmseModel()
    
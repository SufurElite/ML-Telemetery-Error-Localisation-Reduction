import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils import loadModelData, loadRSSModelData, loadCovariateData, loadSections, pointToSection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from multilat import gps_solve

def maxDist(arr):
    """ Determine the maximum index of the array to remove """
    idx = 0
    for i in range(1,len(arr)):
        if arr[i][0]>arr[idx][0]:
            idx = i
    return idx

def knn(X_train, x_example, y_example = [], y_test = [], k:int = 3):
    """ Find the indices of the smallest k signal distances between the
        x_example and X_train, return them accordingly. 
        If y_test and y_example are present, similarly populate them with 
        the indices of the actual closest values"""
    
    if len(X_train)<k: return [[-1, None] * k]
    smallestIndices = []
    y_closest = None

    if len(y_example)!=0 and len(y_test)!=0:
        y_closet = []
    
    currMax = -1
    currMaxIdx = -1

    currYMax = -1
    currYMaxIdx = -1
    x_example = np.array(x_example)
    for i in range(len(X_train)):
        cur_rss = np.array(X_train[i])
        sig_dist = np.linalg.norm(cur_rss-x_example)
        input(y_test[i])
        if len(smallestIndices)<k:
            smallestIndices.append([sig_dist, i])
            if len(smallestIndices)==k:
                currMaxIdx = maxDist(smallestIndices)
                currMax = smallestIndices[currMaxIdx][0]
        elif sig_dist<currMax:
            smallestIndices[currMaxIdx][0] = sig_dist
            smallestIndices[currMaxIdx][1] = i
            currMaxIdx = maxDist(smallestIndices)
            currMax = smallestIndices[currMaxIdx][0]
            print(smallestIndices)
            input()

        # repeat now but for the y-values
        if len(y_closet)<k:
            y_closet.append([actual_dist, i])
            if len(y_closet)==k:
                currYMaxIdx = maxDist(y_closet)
                currYMax = smallestIndices[currYMaxIdx][0]
        elif sig_dist<currMax:
            smallestIndices[currMaxIdx][0] = sig_dist
            smallestIndices[currMaxIdx][1] = i
            currMaxIdx = maxDist(smallestIndices)
            currMax = smallestIndices[currMaxIdx][0]
            print(smallestIndices)
            input()

        pass
    pass

def kValueExplored(X_train, X_test, y_train, y_test,k:int = 3):
    pass

def specificRSSkNN(X_train, X_test, y_train, y_test, k:int = 3):
    
    knn(X_train=X_train,x_example=X_test[0], y_example=y_test[0], y_test=y_test)
    # now iterate over the x_test to find the k-closest values
    pass

def main(knnExploration:bool = False, isSample:bool = False):

    X, y = loadRSSModelData(month="June",includeCovariatePred=False)
    print(X[0])
    # with sampling we create an even distribution in the sections
    if isSample:
        pass
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=101)

    if knnExploration:
        kValueExplored()
    else:
        specificRSSkNN(X_train, X_test, y_train, y_test)
    

if __name__=="__main__":
    main()
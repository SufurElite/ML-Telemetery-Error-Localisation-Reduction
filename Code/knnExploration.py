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

def knn(X_train, x_example, y_example = [], y_train = [], nodeLocs = [], k:int = 3, j: int = -1):
    """ Find the indices of the smallest k signal distances between the
        x_example and X_train, return them accordingly. 
        If y_test and y_example are present, similarly populate them with 
        the indices of the actual closest values"""
    
    if len(X_train)<k: return [[-1, None] * k]
    smallestIndices = []
    y_closest = None
    y_loc = None
    
    if len(y_example)!=0 and len(y_train)!=0 and len(nodeLocs)!=0:
        y_closest = []
        y_loc = gps_solve(y_example,np.array(list(nodeLocs)))
    
    currMax = -1
    currMaxIdx = -1

    currYMax = -1
    currYMaxIdx = -1
    x_example = np.array(x_example)
    
    for i in range(len(X_train)):
        if j!=-1:
            print("{} {} : {}".format(j, i,len(X_train)))
        cur_rss = np.array(X_train[i])
        sig_dist = np.linalg.norm(cur_rss-x_example)
        x_loc = gps_solve(y_train[i], np.array(list(nodeLocs)))
        actual_dist = np.linalg.norm(y_loc-x_loc)
    
        if len(smallestIndices)<k:
            smallestIndices.append([sig_dist, actual_dist, i])
            if len(smallestIndices)==k:
                currMaxIdx = maxDist(smallestIndices)
                currMax = smallestIndices[currMaxIdx][0]
        elif sig_dist<currMax:
            smallestIndices[currMaxIdx][0] = sig_dist
            smallestIndices[currMaxIdx][1] = actual_dist
            smallestIndices[currMaxIdx][2] = i
            currMaxIdx = maxDist(smallestIndices)
            currMax = smallestIndices[currMaxIdx][0]
            #print(smallestIndices)
            #input()

        # repeat now but for the y-values
        if len(y_closest)<k:
            y_closest.append([actual_dist, i])
            if len(y_closest)==k:
                currYMaxIdx = maxDist(y_closest)
                currYMax = y_closest[currYMaxIdx][0]
        elif actual_dist<currYMax:
            y_closest[currYMaxIdx][0] = actual_dist
            y_closest[currYMaxIdx][1] = i
            currYMaxIdx = maxDist(y_closest)
            currYMax = y_closest[currYMaxIdx][0]
            #print(y_closest)
            #input()
    print(smallestIndices)
    print(y_closest)
    simpleAverage = [0,0]
    for i in range(k):
        tmp_y_loc = gps_solve(y_train[smallestIndices[i][2]], np.array(list(nodeLocs)))
        simpleAverage[0]+=tmp_y_loc[0]
        simpleAverage[1]+=tmp_y_loc[1]
    simpleAverage[0]/=k
    simpleAverage[1]/=k
    simp_loc = np.array(simpleAverage)
    print(simpleAverage)
    print(y_example)
    error_dist = np.linalg.norm(y_loc-simp_loc)
    print(error_dist)
    return error_dist
    

def kValueExplored(X_train, X_test, y_train, y_test,k:int = 3):
    pass

def specificRSSkNN(X_train, X_test, y_train, y_test, k:int = 3):
    # want to load in the section nodes
    grid, sections, nodes = loadSections()
    # load in the habitat
  
    nodeLocs = []
    for node in list(nodes.keys()):
        nodeLocs.append([nodes[node]["NodeUTMx"],nodes[node]["NodeUTMy"]])
    err = 0 
    for i in range(len(X_test)):
        err+=knn(X_train=X_train,x_example=X_test[i], y_example=y_test[i], y_train=y_train, nodeLocs = nodeLocs, j = i)
    err/=len(X_test)
    print("Total test error : {}".format(err))
    # now iterate over the x_test to find the k-closest values
    pass

def main(knnExploration:bool = False, isSample:bool = False):

    X, y = loadRSSModelData(month="June",includeCovariatePred=False)
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

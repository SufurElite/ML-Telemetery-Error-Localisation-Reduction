"""
    A file that contains functions that might be useful across different files,
    or to perform specific utility operations once (e.g. associating data)
"""
import pandas as pd
import numpy as np
import json, utm
import datetime
import multilat
import math
from vincenty import vincenty
from scipy.optimize import curve_fit
import pickle
from kmlInterface import HabitatMap, Habitat

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from tensorflow import keras
from sklearn.model_selection import train_test_split
import random

def loadNodes(rewriteUTM=False):
    """ Loads in the nodes from the CSV and returns it as a dictionary where the key is its NodeId"""
    # Load in the data from the CSV using pandas
    df = pd.read_csv(r'../Data/Nodes.csv')
    # Convert the dataframe to a list of dictionaries, where each row is a dictionary
    nodesData = df.to_dict('records')
    nodes = {}
    # Reformat the data to a dictionary where the NodeId is the key of the dictionary
    # rather than an element of a dictionary in a list
    for i in range(len(nodesData)):
        if rewriteUTM:
            utmVals = utm.from_latlon(nodesData[i]["Latitude"], nodesData[i]["Longitude"])
            nodesData[i]["NodeUTMx"] = utmVals[0]
            nodesData[i]["NodeUTMy"] = utmVals[1]
        nodes[nodesData[i]["NodeId"]] = nodesData[i]
        del nodes[nodesData[i]["NodeId"]]["NodeId"]
    return nodes
def distBetweenNodes(node1, node2, Nodes):
    """ Given 2 node ids and a dictionary of nodes this will calculate the distance between the two """
    node1Loc = np.array([np.float64(Nodes[node1]["NodeUTMx"]), np.float64(Nodes[node1]["NodeUTMy"])])
    node2Loc = np.array([np.float64(Nodes[node2]["NodeUTMx"]),np.float64(Nodes[node2]["NodeUTMy"])])
    dist = np.linalg.norm(node1Loc-node2Loc)
    return dist

def loadSections():
    """ Will create a dictionary of section # to coordinates of
        each square formed within the grid """

    nodes = loadNodes(True)

    # for right now, assuming nodes will always be in a square shape
    length = int(math.sqrt(len(nodes.keys())))
    # the number of sections is (length-1)^2
    sections = {}
    for i in range((length-1)**2):
        # the first 0,0 is the X min max,
        # the second 0,0 is the Y min max
        sections[i] = [(0,0),(0,0)]

    # the first number after the id will be the distance to the right, then
    # the left, below, then above
    grid = [[["3290ef",0,0],["37a200",0,0],["377bc2",0,0],["37917d",0,0]],
            [["37a5e9",0,0],["376095",0,0],["3275dd",0,0],["328b9b",0,0]],
            [["3775bf",0,0],["37774c",0,0],["3774a7",0,0],["379611",0,0]],
            [["3288e6",0,0],["328840",0,0],["377905",0,0],["32820a",0,0]]]
    # Was initially automating the prepopulation of the above grid,
    # but due to the grid being slanted it requires a bit more work
    # so I did the ids manually and populated distances below

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # get the current node id
            curNode = grid[i][j][0]
            distRight = 0
            distAbove = 0
            # if we're not at the edges, get the distance to the right and above
            if j!=len(grid[j])-1:
                nodeRight = grid[i][j+1][0]
                distRight = distBetweenNodes(curNode, nodeRight, nodes)
            if i!=0:
                nodeAbove = grid[i-1][j][0]
                distAbove = distBetweenNodes(curNode, nodeAbove, nodes)
            grid[i][j][1] = distRight
            grid[i][j][2] = distAbove

    # go through every 'section' in the grid and find the bounds for the sections
    for i in range(length-1):
        for j in range(length-1):
            # initialize the minimum and maximum values and then
            # just take the min and max for each the nodes to the right and below
            minX = nodes[grid[i][j][0]]["NodeUTMx"]
            minY = nodes[grid[i+1][j][0]]["NodeUTMy"]

            maxX = nodes[grid[i][j+1][0]]["NodeUTMx"]
            maxY = nodes[grid[i][j][0]]["NodeUTMy"]
            # compare the values to the only other smallest/largest on the same level
            minX = min(minX,nodes[grid[i+1][j][0]]["NodeUTMx"])
            minY = min(minY,nodes[grid[i+1][j+1][0]]["NodeUTMy"])
            maxX = max(maxX,nodes[grid[i+1][j+1][0]]["NodeUTMx"])
            maxY = max(maxY,nodes[grid[i][j+1][0]]["NodeUTMy"])

            sections[i*3+j] = [(minX,minY),(maxX, maxY)]

    return grid, sections, nodes

def pointToSection(dataPointX, dataPointY, sections):
    """ Given a UTM X, Y value and the section dictionary with the bounds defined, return which x and y value it falls in """
    # go over all the sections and if the X & Y values lie wtihin the sections bound, then return i
    for i in range(len(sections)):
        if (sections[i][0][0]<=dataPointX and sections[i][0][1]<=dataPointY) and (dataPointX<=sections[i][1][0] and dataPointY<=sections[i][1][1]):
            return i
    # otherwise return -1 (it's out of bounds)
    return -1

def convertOldUtm(oldUTMx,oldUTMy, oldNodes=[], newNodes=[]):
    """ This function will take in a March TestInfo UTMx and convert it
        to a UTM relative to the new UTM nodes"""

    oldUTMs = np.array([np.float64(oldUTMx),np.float64(oldUTMy)])

    # load in the new and old nodes
    if oldNodes==[]:
        oldNodes = loadNodes()
    if newNodes == []:
        newNodes = loadNodes(True)

    # calculate the relative distance to each of the old nodes, but save
    # the location from the new nodes, and then calculate the new utmx, utmy
    oldNodeDists = []
    newNodeLocs = []

    # for every node calculate the distance between the given UTMx,UTMy and the old nodes' locations
    # and then add the same node's location but with the new utms to a list
    for node in oldNodes.keys():
        npNodeLoc = np.array([np.float64(oldNodes[node]["NodeUTMx"]),np.float64(oldNodes[node]["NodeUTMy"])])
        dist = np.linalg.norm(oldUTMs-npNodeLoc)
        newNodeLocs.append([newNodes[node]["NodeUTMx"],newNodes[node]["NodeUTMy"]])
        oldNodeDists.append(dist)
    # once we have this data, we can calculate the location of the old utmx, utmy from the march data
    # to a new one
    newUTMx,newUTMy = multilat.gps_solve(oldNodeDists, list(np.array(newNodeLocs)))
    return newUTMx,newUTMy

def loadBoundaries(redoUtm=False):
    """ Will return the boundaries of the node system """
    nodes = loadNodes(rewriteUTM=redoUtm)
    minX = nodes[list(nodes.keys())[0]]["NodeUTMx"]
    maxX = nodes[list(nodes.keys())[0]]["NodeUTMx"]
    minY = nodes[list(nodes.keys())[0]]["NodeUTMy"]
    maxY = nodes[list(nodes.keys())[0]]["NodeUTMy"]
    for node in nodes.keys():
        if nodes[node]["NodeUTMx"]>maxX:
            maxX = nodes[node]["NodeUTMx"]
        if nodes[node]["NodeUTMx"]<minX:
            minX = nodes[node]["NodeUTMx"]
        if nodes[node]["NodeUTMy"]>maxY:
            maxY = nodes[node]["NodeUTMy"]
        if nodes[node]["NodeUTMy"]<minY:
            minY = nodes[node]["NodeUTMy"]
    return minX,maxX,minY,maxY

def loadCovariateModel():
    """ Used to load in the Classifier for the prediction model """
    loadedModel = pickle.load(open("models/covariateClf", 'rb'))
    return loadedModel

def loadData(month="March", pruned=False, combined = False):
    """ Loads the data that has been stored from associateTestData,
        month specifies the month the data was taken happened (March or June)
        by default is March """
    pathName = "../Data/"+month+"/associatedTestData"
    if pruned:
        pathName+="Pruned"
    pathName+=".json"

    #Delete after usage.
    if month == "June":
        pathName = "../Data/"+month+"/trilatData.json"

    #Don't want to ruin the other function just yet; so another parameter if we run the combined June-March prediction
    if combined == True:
        if month == "March":
            pathName = "../Data/"+month+"/associatedMarchData_2.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    return data
def loadCovariateData():
    """ Using the additional features from the march data,
        initially we can try using ordinal number values for the habitats """

    # load in the habitat map
    habitatMap = HabitatMap()
    data = loadData("March")
    # Load old and new node data for converting the old UTMs to new
    oldNodes = loadNodes()
    newNodes = loadNodes(True)
    """ Will create a dictionary of the locations from the march associated Test Data gt
        and then any points from june that falls into a radius of 10m (arbitrarily chosen)
            - if there are multiple points within this radius, the closest one will be selected as
            the same habitat type
    """
    nodes = list(newNodes.keys())
    habitatDist = {}

    X = []
    y = []

    for key in data.keys():
        # convert the old utm values to new
        convertedUTMx, convertedUTMy = convertOldUtm(data[key]["GroundTruth"]["TestUTMx"],data[key]["GroundTruth"]["TestUTMy"],oldNodes,newNodes)
        # use the habitat index to determine the value
        habIdx, habitatName = habitatMap.whichHabitat(convertedUTMx, convertedUTMy)
        if habIdx==-1: continue
        tmp_x = [[0,0] for i in range(len(nodes))]
        for dataEntry in data[key]["Data"]:
            nodeKey = dataEntry["NodeId"]
            if nodeKey=="3288000000": nodeKey="3288e6"
            nodeIdx = nodes.index(nodeKey)
            assert(nodeKey==nodes[nodeIdx])
            tmp_x[nodeIdx][0]+=dataEntry["TagRSSI"]
            tmp_x[nodeIdx][1]+=1
        # then average the values in this 2 minute time span
        x = [0 for i in range(len(nodes))]
        for i in range(len(tmp_x)):
            if tmp_x[i][1]==0:continue
            x[i]=tmp_x[i][0]/tmp_x[i][1]

        y.append(habIdx)
        X.append(x)
        if habitatName not in habitatDist:
            habitatDist[habitatName] = 0
        habitatDist[habitatName]+=1

    assert(len(X)==len(y))
    print("From the march data, X has {} and y has {} values".format(len(X),len(y)))
    # now load in the June data
    juneData = loadData(month="June")
    juneX = juneData["X"]
    june_y = juneData["y"]
    assert(len(juneX)==len(june_y))

    for i in range(len(juneX)):
        # get the june utm values
        juneUTM = utm.from_latlon(june_y[i][0], june_y[i][1])
        # use the habitat index to determine the value
        habIdx, habitatName = habitatMap.whichHabitat(juneUTM[0], juneUTM[1])

        if habIdx==-1: continue

        tmp_x = [0 for i in range(len(nodes))]
        for nodeEntry in juneX[i]["data"].keys():
            nodeKey = nodeEntry
            if nodeKey=="3288000000": nodeKey="3288e6"
            nodeIdx = nodes.index(nodeKey)
            assert(nodeKey==nodes[nodeIdx])
            tmp_x[nodeIdx]=juneX[i]["data"][nodeEntry]
        X.append(tmp_x)
        y.append(habIdx)
        if habitatName not in habitatDist:
            habitatDist[habitatName] = 0
        habitatDist[habitatName]+=1

    print("After applying the covariate habitat map, X has {} and y has {} values".format(len(X),len(y)))

    print("The distribution of habitat values is ")
    for habKey in habitatDist.keys():
        print(habKey,habitatDist[habKey])

    X = np.array(X)
    y = np.array(y)
    assert(len(X)==len(y))

    return X, y

def loadRSSModelData(month="June",includeCovariatePred = False):
    """ This is similar to loading model data, but the y values, instead of being offsets to correct
        the error derived from multilat, are the distances to each node"""

    # if we want to include the habitat as a factor, then load in the classification
    covariateModel = None
    habitatMap = HabitatMap()
    if includeCovariatePred:
        covariateModel = loadCovariateModel()
    # load in the data
    data = loadData(month)
    X_vals = data["X"]
    y_vals = data["y"]
    assert(len(X_vals)==len(y_vals))
    # load in the nodes
    nodes = loadNodes(rewriteUTM=True)
    nodeKeys = list(nodes.keys())
    xInputNum = len(nodeKeys)
    # starting idx is either 0 or 1 if we include the habitat,
    # but this has the opportunity to be increased later if we want to have multiple
    # features
    startIdx = 0
    if includeCovariatePred:
        xInputNum+=1
        startIdx+=1
    X = []
    y = []
    for i in range(len(X_vals)):
        tmp_x = [0 for i in range(xInputNum)]
        # signal_X is to be able to predict the habitat from signals
        signal_X = [0 for i in range(len(nodeKeys))]
        tmp_y = [0 for i in range(len(nodeKeys))]
        # ground truth location
        tagGt = utm.from_latlon(y_vals[i][0], y_vals[i][1])

        for nodeNum in range(len(nodeKeys)):
            nodeKey = nodeKeys[nodeNum]
            nodeLoc = np.array([np.float64(nodes[nodeKey]["NodeUTMx"]), np.float64(nodes[nodeKey]["NodeUTMy"])])
            tagLoc = np.array([np.float64(tagGt[0]),np.float64(tagGt[1])])
            tmp_y[nodeNum] = np.linalg.norm(nodeLoc-tagLoc)
            if nodeKey not in X_vals[i]["data"] or X_vals[i]["data"][nodeKey]<-102:
                # and then calculate this node's relative distance to the tag
                continue
            else:
                # otherwise set the relative x value equivalent to the distance
                tmp_x[startIdx+nodeNum] = calculateDist_2(X_vals[i]["data"][nodeKey])
                signal_X[nodeNum] = X_vals[i]["data"][nodeKey]

        if includeCovariatePred:
            # will start with just ordinal but may switch to different encoding,
            # model dependent
            #habitatPred, habitat_title = habitatMap.whichHabitat(tagGt[0], tagGt[1])
            #if habitatPred==-1:
            habitatPred = covariateModel.predict(np.array([signal_X]))[0]

            tmp_x[0] = habitatPred
        X.append(tmp_x)
        y.append(tmp_y)
    X = np.array(X)
    y = np.array(y)

    print(len(X), len(y))
    assert(len(X)==len(y))

    return X,y

def loadModelData(month="June", modelType="initial", threshold=-102, includeCovariatePred = False, verbose=True):
    """ Unlike the regular loadData function, this one presents the information in a format
        specifically for a model to train on. The way the data will differ depends on if it's initial
        (where we'll be trying to predict offsets of the calculated values) or if it's sequential, given
        n steps predict the current step"""
    covariateModel = None
    if includeCovariatePred:
        covariateModel = loadCovariateModel()
    if month=="June":
        res = multilat.predictions(threshold,keepNodeIds=True,month="June")
    else:
        res = multilat.predictions(threshold,keepNodeIds=True,month="March")

    rewriteUtm = False
    if month=="June":
        rewriteUtm = True

    nodes = loadNodes(rewriteUTM=rewriteUtm)
    notUsed, sections, _ = loadSections()

    # the input to the model is going to be the calculated distance to each node
    # plus the projected output
    nodeKeys = list(nodes.keys())

    xInputNum = len(nodeKeys)+2
    startIdx = 2
    if includeCovariatePred:
        xInputNum+=1
        startIdx+=1
    X = []
    y = []

    counter = 1
    numberOfVals = len(list(res.keys()))
    for entry in res.keys():
        if verbose:
            print("{}/{}".format(counter, numberOfVals))
        x = [0 for i in range(xInputNum)]
        x[0] = res[entry]["res"][0]
        x[1] = res[entry]["res"][1]
        tmp_y = res[entry]["gt"]-res[entry]["res"]

        tmp_y[0] = round(tmp_y[0],1)
        tmp_y[1] = round(tmp_y[1],1)

        #If it is a March run, it would not have a nodeIds keys
        #Figured it is not that importnat for March rn.
        for nodeNum in range(len(nodeKeys)):
            # if the node id is not one of the ones the tag contacted skip

            if nodeKeys[nodeNum] not in res[entry]["nodeIds"]: continue
            # otherwise set the relative x value equivalent to the distance
            nodeIdx = res[entry]["nodeIds"].index(nodeKeys[nodeNum])

            x[startIdx+nodeNum] = res[entry]["nodeDists"][nodeIdx]
        if includeCovariatePred:
            # will start with just ordinal but may switch to different encoding,
            # model dependent
            habitatPred = covariateModel.predict(np.array([x[startIdx:]]))
            x[2] = habitatPred[0]

        x = np.array(x)
        tmp_y = np.array(tmp_y)
        X.append(x)
        y.append(tmp_y)
        counter+=1

    X = np.array(X)
    y = np.array(y)
    print(len(X), len(y))
    assert(len(X)==len(y))

    return X, y

def associateJuneData(newData = False):
    """ This function currently only works for June but ideally will be refactored with the March function.
        It associates the TestIds and its data with the rows

        Of the beepdata, currently uses 10,927 rows for 707 batches ~ 15.4 rows a batch
        seems like there must be an error
        """

    if(newData == True):
        #print("Running the new data!")
        nodeLocations = loadNodes(rewriteUTM=True)
    # Load the June BeepData in
    beepData = pd.read_csv(r'../Data/June/BeepData.csv')
    # sort the data by tag and time, so that as we increment we can group them together in ~2 second intervals
    beepData.sort_values(by = ['TagId', 'Time.local'], axis=0, ascending=[False, True], inplace=True, ignore_index=True, key=None)
    # Load all the flight data into dataframes, and then combine into one dataframe
    flights = ['19522D2A_flight1.csv', '19522D2A_flight2.csv', '19522D2A_flight3.csv','5552664C_flight1.csv']

    flightDataList = []

    for flight in flights:
        df = pd.read_csv('../Data/June/'+flight, index_col=None, header=0)
        # get the name of the file right before the _
        df['TagId'] = flight[:-12]
        flightDataList.append(df)

    flightDF = pd.concat(flightDataList, axis=0, ignore_index=True)
    flightDF['datetime(utc)'] = pd.to_datetime(flightDF['datetime(utc)'])
    """
    General idea:
        The beep data (input values) are sorted by tag and by time, so increasing over them should
        be data that belongs to either the same batch (of 2 secs) or to the next batch

        Approach -
            Increment over the beepdata and check if the row is either over 2 seconds ahead of the base value
            in which case reset the base value, or include it in the batch of data (assuming the data )
            Only include the batch if there are at least 3 nodes included and there is a corresponding flightdata
    """
    # X will be composed of batches of data
    X = []
    # y will be composed of X items corresponding GT flight values
    y = []
    batch = {}
    baseTime = '0'
    tag = ''
    errorDist = [0,0,0,0]
    avgNums = [0,0]
    missedTheCut = 0
    missedthecutbadSort = 0
    missedTheCutTooFew = 0
    for index, row in enumerate(beepData.iterrows()):
        #Converting it already
        row[1]['Time.local'] = datetime.datetime.strptime(row[1]['Time.local'], "%Y-%m-%dT%H:%M:%SZ")
        currDate = row[1]['Time.local']
        #Changing NodeId - because NewData needs to find the NodeID
        if row[1]['NodeId']=="3288000000": row[1]['NodeId']="3288e6"
        #print(index)
        if baseTime == '0':
            batch = {}
            baseTime = currDate
            tag = row[1]['TagId']
        elif tag!=row[1]['TagId'] or (currDate-baseTime>datetime.timedelta(0,2)):
            # Look for flight data th the same time
            upperBound = baseTime+datetime.timedelta(0,2)
            #print(baseTime,currDate)
            timeSort = flightDF[flightDF['datetime(utc)'].between(baseTime.strftime("%Y-%m-%d %H:%M:%S"),upperBound.strftime("%Y-%m-%d %H:%M:%S"))]
            # then sort the above timeSorted data by the relevant test tag id
            tagSort = timeSort[timeSort['TagId'].str.contains(tag)]


            if len(batch.keys())>=3 and len(tagSort)!=0:
                data = {}
                data["time"] = baseTime.strftime("%Y-%m-%dT%H:%M:%SZ")
                data["tag"] = tag
                data["data"]={}
                rowVals = 0
                for i in batch.keys():
                    # average the tag-Node RSSi value
                    if newData == True:
                        data["data"][i] = batch[i]
                    else:
                        data["data"][i] = batch[i][1]/batch[i][0]
                    rowVals+=batch[i][0]
                print(data["data"])
                X.append(data)
                if newData == True:
                    avgAlt = tagSort["altitude_above_seaLevel(feet)"].mean()
                # get the average latitude over the 2 seconds
                avgLat = tagSort["latitude"].mean()
                # get the average longitude over the 2 seconds
                avgLong = tagSort["longitude"].mean()

                #If newData is true; then we add the altitude as well
                if newData == True:
                    y.append([avgLat,avgLong,avgAlt])
                else:
                    y.append([avgLat,avgLong])

                avgNums[0]+=1
                avgNums[1]+=rowVals
            else:
                if tag=="5552664C":
                    errorDist[0]+=1
                elif index<16518:
                    errorDist[1]+=1
                elif index<21253:
                    errorDist[2]+=1
                else:
                    errorDist[3]+=1
                if(len(batch.keys())<3):
                    missedTheCutTooFew+=1
                elif len(tagSort)==0:
                    missedthecutbadSort+=1
                missedTheCut+=len(batch.keys())
            # Reset the variables
            batch = {}
            baseTime = currDate
            tag = row[1]['TagId']
        #If the newData is true; then gather different info
        if  newData == True:
            if row[1]['NodeId'] not in batch.keys():
                batch[row[1]['NodeId']]=[0,0,[]]
                batch[row[1]['NodeId']][0] +=1
                name = row[1]['NodeId']
                batch[row[1]['NodeId']][1] = [nodeLocations[name]['Latitude'],nodeLocations[name]['Longitude']]
                batch[row[1]['NodeId']][2].append(row[1]['TagRSSI'])
            elif row[1]['NodeId'] in batch.keys():
                batch[row[1]['NodeId']][0] +=1
                batch[row[1]['NodeId']][2].append(row[1]['TagRSSI'])
        #Otherwise, gather the same information
        else:
            if row[1]['NodeId'] not in batch.keys():
                batch[row[1]['NodeId']]=[0,0]
            batch[row[1]['NodeId']][0]+=1
            batch[row[1]['NodeId']][1]+=row[1]['TagRSSI']
    # every x should have a corresponding y
    assert(len(X)==len(y))
    finalData = {}
    finalData['X']=X
    finalData['y']=y
    if(newData == True):
        with open("../Data/June/newData.json","w+") as f:
            json.dump(finalData, f)
    else:
        with open("../Data/June/associatedTestData.json","w+") as f:
            json.dump(finalData, f)
    print(missedTheCut,missedTheCutTooFew,missedthecutbadSort)
    print(errorDist)
    print("There were {} rows,\nof which there were {} 2 second intervals/batches with relevant data,\naveraging {} rows a batch".format(len(beepData),avgNums[0],(avgNums[1]/avgNums[0])))



def associateMarchData(month="March"):
    """ This function currently only works for March but ideally will be refactored to be generic
        for March or for the June data given a month as a parameter, and it associates the TestIds
        and its data with the rows """

    # Load in both the test info and beepData into a dataframe
    testInfo = pd.read_csv(r'../Data/'+month+'/TestInfo.csv')
    beepData = pd.read_csv(r'../Data/'+month+'/BeepData.csv')

    # data dictionary is going to be the associated data we return
    # where each key is a test and its values are the related test data
    data = {}

    testsWithoutDataIds = []
    # rows are the total rows of test data that are used
    rows = 0
    # runningIdx is the index of the current row in the CSV file so if there's
    # an issue we can inspect it closely
    runningIdx = 1
    # testsWith/Without Data are variables to keep track of how many tests did/didn't
    # appear to have associated data
    testsWithData = 0
    testsWithoutData = 0

    # Loop over all the tests
    for index, row in enumerate(testInfo.iterrows()):
        runningIdx+=1

        # get the relevant date info from the current testInfo row
        # splitDate = [month, day, year]
        splitDate = row[1]["Date"].split("/")
        #add the relevant year
        dateTime = "20"+splitDate[2]+"-"
        # if the month is single digit first add a 0
        if len(splitDate[0])<2:
            dateTime+="0"
        dateTime+=splitDate[0]+"-"
        # if the day is single digit first add a 0
        if len(splitDate[1])<2:
            dateTime+="0"
        dateTime+=splitDate[1]+"T"

        # If the hour (+5 for Bogota->UTC) is single digit, add a 0
        if len(str(row[1]["Hour"]+5))<2:
            dateTime+="0"
        dateTime+=str(row[1]["Hour"]+5)+":"

        # If the min is single digit, add a 0
        if len(str(row[1]["Min"]))<2:
            dateTime+="0"
        dateTime+=str(row[1]["Min"])

        # sort the beepData if the time contains the dateTime formatted above
        timeSort = beepData[beepData['Time.local'].str.contains(dateTime)]
        # then sort the above timeSorted data by the relevant test tag id
        tagSort = timeSort[timeSort['TagId'].str.contains(row[1]["TagId"])]

        # If tagSort isn't empty, we can add the data to this testId in the dictionary
        if not tagSort.empty:
            # gt = Ground Truth of the test
            #   - i.e. all the relevant info from the testInfo as a dictionary
            gt = row[1].to_dict()
            # tmp is the data for this given test
            tmp = {"GroundTruth":gt, "Data":tagSort.to_dict('records')}
            # create a key based on the the tagid, testid, and date for the associated data
            # and set it equal to what we found above
            data[row[1]["TagId"]+row[1]["TestId"]+dateTime] = tmp
            # add the total number of rows to keep track
            rows+=len(tagSort.to_dict('records'))
            # and increment the number of tests that had data
            testsWithData+=1
        else:
            # if tagSort was empty, there was no relevant data found
            # increment the number of test without data by 1
            testsWithoutData+=1
            # and add this particular test id and row index for further exploration if desired
            testsWithoutDataIds.append(str(runningIdx)+" "+row[1]["TestId"]+"\n")

    # print the relevant logging/debugging info from above
    print("""The number of rows with data was {}\nThe number of tests with data was {}\nThe number of tests without data was {}\n And the average amount of data for each test is {}""".format(rows, testsWithData, testsWithoutData, str(float(rows/testsWithData))))

    # Write the resulting data into its relevant folder
    with open("../Data/"+month+"/TestsNoData.txt", "w+") as f:
        f.writelines(testsWithoutDataIds)

    with open("../Data/"+month+"/associatedTestDataPruned.json","w+") as f:
        json.dump(data, f)
    return data
def rewriteMarchData(month="March"):
    pathName = "../Data/"+month+"/associatedTestData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    X = []
    Y = []
    for key in data.keys():
        for id in data[key]:
            #print(data[key][id])
            if id == "GroundTruth":
                date = data[key][id]["Date"]
                date = date[::-1]
                timeS = data[key][id]["Start.Time"]
                timeE = data[key][id]["Stop.Time"]
                dateT = date+"-"+timeS+"-"+timeE
                tag = data[key][id]["TagId"]
                test = data[key][id]["TestId"]
                posX = data[key][id]["TestUTMx"]
                posY = data[key][id]["TestUTMy"]
            if id == "Data":
                information = {}
                for item in data[key][id]:
                    if item['NodeId']=="3288000000": item['NodeId']="3288e6"
                    if item["NodeId"] not in information.keys():
                        information[item["NodeId"]] = [0,1]
                        information[item["NodeId"]][0] = item["TagRSSI"]
                    else:
                        information[item["NodeId"]][0] += item["TagRSSI"]
                        information[item["NodeId"]][1] += 1
                for idX in information.keys():
                    sum = information[idX][0]
                    count = information[idX][1]
                    avg = round(sum/count,2)
                    information[idX] = avg
        X.append({    "time" : dateT,
                        "tag" : tag,
                        #"testId" = test
                        "data" : information
        })
        Y.append([posX,posY])
    assert(len(X) == len(Y))
    finalData = {}
    finalData['X']=X
    finalData['y']=Y
    with open("../Data/"+month+"/associatedMarchData_2.json","w+") as f:
        json.dump(finalData, f)


def LoadSeqData(month="June"):
    '''
        Idea is that if we give the machine the distances it is away from other nodes and the signal strenght, we would be able to predict the distance it is away from
        the actual node.
        So, basically trying to reproduce the 'proximity' filter, with a neural network -> thus we would get an even better distance result, that would count the
        noise / vegetation in as well. Since, the distance ~ signalStr equation in itself is not enough to make the prediction to the actual distance.
    '''
    if(month =="June"):
        # Load in the June JSON
        pathName = "../Data/"+month+"/newData.json"
        with open(pathName,"r") as f:
            data = json.load(f)
        nodes = loadNodes(rewriteUTM=True)

    X = data["X"]
    y = data["y"]
    newX = []
    newY = []
    newZ = []
    #Going to gather the new data pairs in the form of:
    #Elements of newX will look like: [
    #                                   [signalStrength of the current node in the batch],
    #                                   [all relative node distances in the same batch, that also have relevant signal],
    #                                   [all relative node signals in the same batch, that are relevant]
    #                                   [batch_number]]
    #Elements of newY will look like: [distance between current node and the drone]
    n = 0
    for index, item in enumerate(data["X"]):

        #Keys
        keys = item["data"].keys()
        #Get the drone position in lat,long
        dronePos = [data["y"][index][0],data["y"][index][1]]
        #Convert those positions to utm coordinates
        utmDronePos = utm.from_latlon(dronePos[0],dronePos[1])
        utmDronePosArr = np.array([np.float64(utmDronePos[0]),np.float64(utmDronePos[1])])

        #getNodedistances returns two lists. One of them contains [[relative node distances in the same batch],[position of current node], [signal of current node]]
        #The skipKeys will have all the NodeIDs of the nodes that need to be skipped because they are
        #outside of the range for the signals
        replace, skipKeys = getNodedistances(item["data"])

        #Gets all the signals in the batch
        signals = getSignals(item["data"])

        for key in keys:
            if(key == "3288000000"): key="3288e6"
            #If the key has wrong outside of range signal, we skip it
            if key in skipKeys:continue

            #If the drone position is not recorded well, we skip it
            if(dronePos == (0,0) or dronePos == (0.754225181062586, -12.295563892977972) or dronePos == (4.22356791831445, -68.85519277364834)): continue

            #We gather all the signals that are in the batch for the current node
            #and skip the outside of the range signals
            sSignals = []
            for newKey in keys:
                if(newKey == "3288000000"): key="3288e6"
                if newKey == key: continue
                if newKey in skipKeys:continue
                sSignals.append(signals[newKey])
            #Need at least 3 nodes, so we take out everything else - lost about 48 data points from the 4600.
            if len(sSignals) < 2: continue
            #current node position in latlon, then convert it into utm, then get the distance between them
            sNodePos = item["data"][key][1]
            utmsNodePos = utm.from_latlon(sNodePos[0],sNodePos[1])
            utmsNodePosArr = np.array([np.float64(utmsNodePos[0]),np.float64(utmsNodePos[1])])
            actualDist = np.linalg.norm(utmsNodePosArr-utmDronePosArr)

            #Eventually add these data pairs into the newX and newY, which will be saved for later training and modelling.
            newX.append([item['data'][key][2],item['data'][key][0], sSignals, index, key])
            newY.append(actualDist)
            newZ.append(dronePos)
    assert(len(newX) == len(newY))
    finalData = {}
    finalData['X']=newX
    finalData['y']=newY
    finalData['Z']=newZ

    #Save the file in a json, that will be used for modelling / training
    with open("../Data/"+month+"/distanceNNData.json","w+") as f:
            json.dump(finalData, f)

def FunctionalModel():
    X, y = loadANNData_3()
    #print(y)
    y = np.array(y, dtype="float64")
    print(y)
    i1 = np.array(X,dtype="float64")
    print(i1[0])
    input()
    #i2 = np.array(X[1], dtype="float64")
    #print(i2[0])
    #input()
    #i3 = np.array(X[2], dtype=object)
    #print(i3[0])
    #input()

    print(i1[0].shape)
    print(i1.shape)
    input()
    print(i1)
    print("\n")
    input()
    '''
    for i in range(0,600):
        print(i1[i].shape,i2[i].shape,i3[i].shape)
    input()
    '''
    #print(X[0])
    #input()
    #input1 = Input(shape=(16,))
    '''
    input2 = Input(shape=(17,))
    #input0 = Concatenate()([input1, input2])
    x = Dense(2)(input2)
    hidden1 = Dense(10,activation='relu')(x)
    hidden2 = Dense(8,activation='relu')(hidden1)
    hidden3 = Dense(6, activation='relu')(hidden2)
    output = Dense(1,activation='relu')(hidden3)
    model = Model(inputs=input2, outputs=output)
    model.summary()
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_absolute_error"]
    )
    '''
    model = Sequential()
    model.add(Dense(1, input_dim=1, activation="relu"))
    model.add(Dense(1, activation="linear"))
    opt = SGD(learning_rate=0.01,momentum=0.9)
    model.compile(
        loss="mean_squared_logarithmic_error",
        optimizer=opt,
        metrics=["mse"]

    )
    x_train, x_test, y_train, y_test = train_test_split(i1, y, train_size=0.8, random_state=101)
    print(x_train)
    print(y_train)
    input()
    #print(x1_train[0],x2_train[0], y_train[0])
    #input()
    history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_split=0.2)
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print(test_scores)
    predicted = model.predict(x_test)

    print(predicted)
    print(y_test)
    for i in range(0,len(y_test)):
        print(predicted[i], "and",y_test[i], "and", x_test[i])
        input()

def error_calculation(arr, identified):
    '''
        This function is similar to the check_dissonance function, but is different from it to an extent.
        check_dissonance can be deleted, once this one is working properly.
        Jesus it's such a mindfuck to explain what this is doing, but I will try.
        So, this function in simple terms going to calculate the error likliness for each signal in a batch.
        Obviously if there are more 'faulty' signals, then the other 'good' signal values are going to increase
        as well. We don't want that. So, this function is going to be called recursively after the highest
        error is going to be removed, and perhaps changed into a better representation of the signal. OR
        this function could be used just to identify the best signals for trilateration.

        The way it calcualtes the error is by calculating the otherNode locations to the node that the current signal
        is received by. By the assumption that the lower the signal, the higher the 'drone' is and that means the otherNodes are
        also higher, meaning it should have the highest nodeDistanceMean whatsoever.
        Then we would check how much percentage is it needed to get to below and above the certain value, based on whether the signal
        is lower or higher than the current one.
        And then punishment / reward numerator is added, then it is all divided by a size (the highest probability it can get)-
        Thus, giving us a cool value for error. But as said before, if there is a wrong signal in the batch, that is going to increase the
        other signals' errors as well.
    '''
    #print(arr)
    currSig = arr[0]
    currDistS = arr[1]
    currOtherSig = arr[2]
    currOtherDistS = arr[3]
    size = (len(currOtherSig))
    numerator = 0
    endPercent = 50
    errorDistances = []
    for i in range(0,len(currOtherSig)):
        compare = currOtherDistS[i]
        val0 = currDistS * 1.0
        val1 = currDistS
        val2 = currDistS
        percent = 0
        percent2 = 0
        print("\n")
        print(i,". Run")
        print("#################################")

        for j in range(0,endPercent):
            if(val0 *(1+(j/100)) < compare):
                val1 = val0 *(1+((j+1)/100))
                percent = j+1
            if(val0 *(1-(j/100)) > compare):
                val2 = val0 *(1-((j+1)/100))
                percent2 = j+1
        if(currOtherSig[i] > currSig):

            print("\n")
            print(currOtherSig[i], "is bigger than ", currSig)
            print(currOtherSig[i], currSig)
            print(val0, compare)
            print(val1, compare, percent)
            print(val2, compare, percent2)
            print("\n")


            if(val0 < compare):
                if(insideIdentified_2(identified, compare) == False):
                    addNum = (0.1*(percent)/endPercent)
                    numerator += addNum
                    print("Possible dissonance")
                    print(val0, " is less than ", compare, "so we add", addNum)
                else:
                    print("Possible dissonance, but ")
                    print("Inside the identified error signals")
                    addNum = (0.025*(endPercent-percent)/endPercent)
                    numerator += addNum
                    print(val0, "is less than", compare, "but", currOtherSig[i], "is in identified error signals")
                    print("So we add", addNum)
            else:
                addNum = (0.05*(endPercent-percent2)/endPercent)
                numerator += addNum
                print("Good for first")
                print(val0, " is bigger than ", compare, "so we add", addNum)

        if(currOtherSig[i] < currSig):

            print("\n")
            print(currOtherSig[i], "is less than ", currSig)
            print(currOtherSig[i], currSig)
            print(val0, compare)
            print(val1, compare, percent)
            print(val2, compare, percent2)
            print("\n")


            if(val0 > compare):
                if(insideIdentified_2(identified, compare) == False):
                    addNum = (0.1*(percent2)/endPercent)
                    numerator += addNum
                    print("Possible dissonance")
                    print(val0, " is bigger than ", compare, "so we add", addNum)
                else:
                    print("Possible dissonance, but ")
                    print("Inside the identified error signals")
                    addNum = (0.025*(endPercent-percent2)/endPercent)
                    numerator += addNum
                    print(val0, "is bigger than", compare, "but", currOtherSig[i], "is in identified error signals")
                    print("So we add", addNum)
            else:
                addNum = (0.05 *(endPercent-percent)/endPercent)
                numerator += addNum

                print("Good for first")
                print(val0, " is less than ", compare, "so we add", addNum)

        if(currOtherSig[i] == currSig):
            print("\n")
            print(currOtherSig[i], "is equal to ", currSig)
            print(currOtherSig[i], currSig)
            print(val0, compare)
            print(val1, compare, percent)
            print(val2, compare, percent2)
            print("\n")

            if(insideIdentified_2(identified, compare) == False):
                if(val0 > compare):
                    addNum = (0.1*(percent2)/endPercent)
                    numerator += addNum
                else:
                    addNum = (0.1 *(percent)/endPercent)
                    numerator += addNum
            else:
                if(val0 > compare):
                    addNum = (0.05*(percent2)/endPercent)
                    numerator += addNum
                else:
                    addNum = (0.05*(percent)/endPercent)
                    numerator += addNum

        #input()
        print("#################################")
        print("\n")

        '''
        if(addNum != 0):
            dist1 = calculateDist_2(currSig)
            dist2 = calculateDist_2(currOtherSig[i])
            err = abs(dist1-dist2)
            if(currOtherSig[i] < currSig):
                err = err*(percent2/endPercent)
            elif(currOtherSig[i] > currSig):
                err = -err*(percent/endPercent)
            else:
                if(percent2 > percent):
                    err = err*(percent2/endPercent)
                else:
                    err = -err*(percent/endPercent)
            #errorDistances.append([addNum, err, currOtherSig[i])
            if(len(errorDistances) == 0):
                errorDistances.append([currSig, arr[len(arr)-2]])
        '''
    '''
        The error distances need to compare each signal error % in a bacth, to successfully indentify
        the signal(s) that are wrong. After identifying those signals, we can make a proper guess at the
        error distances. Otherwise, it is going to just mess it up.
    '''
    '''
        Another idea, is to recursively run this code, getting the highest error signal out of the batch,
        calculating the percentage for error again (then the likely 'big' error node will not affect the others)
        and then do it until we have only 3 signals left. Those 3 signals will be used for trilateration.
    '''


    #print(numerator)
    #print(size)

    #print(errorDistances)
    #print(round(numerator/size*100,2), "%")

    return [arr, round(numerator/size*100,2)]

def removeHighestError(arr, arr2, identified):
    arr2_sorted = sorted(arr2)
    print("\n")
    print("removeErrorFunction")
    print("\n")
    lowB = deviation(arr2, arr2_sorted[0])
    uppB = deviation(arr2, arr2_sorted[len(arr2_sorted)-1])
    if(len(identified) == 0):
        if(lowB > uppB):
            index = findIndex(arr2, arr2_sorted[0])
            print(arr2[index])
            print(arr[index])
            identified.append([arr[index], index])
        else:
            index = findIndex(arr2, arr2_sorted[len(arr2)-1])
            #print(arr[index])
            #print(arr2[index])
            identified.append([arr[index], index])
        print("\n")
        print(arr)
        print(arr2)
        print(identified)
        print("\n")
        return arr, identified
    elif(len(arr) - len(identified) == 3):
        print("\n")
        print(arr)
        print("\n")
        print(identified)
        for ident in identified:
            arr.remove(ident[0])

        return arr, identified
    else:
        inside = []
        outside = []
        lowest = 100
        highest = -1
        for err in range(0,len(arr2)):

            if(insideIdentified(identified, err)):
                continue
            else:
                if(arr2[err] > highest):
                    highest = arr2[err]
                    indexHigh = err
                if(arr2[err] < lowest):
                    lowest = arr2[err]
                    indexLow = err
        print(lowest, highest)
        print(indexLow, indexHigh)
        lowB = deviation(arr2, lowest)
        uppB = deviation(arr2, highest)

        #Finding which one to take in
        print(arr2[indexLow])
        print(arr[indexLow])
        print(arr[indexHigh])
        print(arr2[indexHigh])
        print(lowB, uppB)
        print(arr2_sorted)
        if(lowB > uppB):
            print(arr2[indexLow])
            print(arr[indexLow])
            identified.append([arr[indexLow], indexLow])
        else:
            print(arr[indexHigh])
            print(arr2[indexHigh])
            identified.append([arr[indexHigh], indexHigh])
        print("\n")
        print(arr)
        print(arr2)
        print(identified)
        print("\n")
        return arr, identified

    #input()
def deviation(arr, value):
    total = 0
    for i in range(0, len(arr)):
        total += pow(arr[i]-value,2)
    result = sqrt(total/(len(arr)-1))
    return result

def insideIdentified(arr, index):
    for i in range(0, len(arr)):
        if(arr[i][1] == index):
            return True
    return False

def insideIdentified_2(arr, value):
    for i in range(0, len(arr)):
        if(arr[i][0][1] == value):
            return True
    return False

def findIndex(arr, value):
    for i in range(0, len(arr)):
        if(arr[i] == value):
            return i
def testLowest(arr):
    err = []
    for i in range(0,len(arr)):
        err.append(arr[i][6])
    ident = []
    while(len(arr) - len(ident) != 3):
        highest = -1
        highestIndex = None
        for j in range(0,len(err)):
            if(abs(err[j]) > highest and j not in ident):
                highest = abs(err[j])
                highestIndex = j
        ident.append(highestIndex)
    newImportant = []
    for k in range(0,len(arr)):
        if(k not in ident):
            newImportant.append(arr[k])
    return newImportant




def loadANNData_3():
    '''
        Idea here would be gather the signal and then the calculated distance, then get what the error is
        then train the signal and the error on the model. Model would try to predict the error that is
        contributed to the signal~distance.
    '''
    grid, sections, nodes = loadSections()
    month = "June"
    pathName = "../Data/"+month+"/distanceNNData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    iput1 = []
    iput2 = []
    iput3 = []
    output = []
    batchN = 0
    a = 0
    distanceSum = []
    actualDist = []
    allSignals = []
    importantDs = []
    goodCount = 0
    badCount = 0
    goodCount2 = 0
    badCount2 = 0
    dronePos = None
    minimals = []
    fileX = []
    fileY = []
    for index, i in enumerate(data["X"]):
        actual = data['y'][index]
        predicted = calculateDist_2(i[0])
        err = actual - predicted
        err = np.float64(err)
        #print("SIGNAL VALUE TO BE MANUPULATED")
        #print(i[0], i[1], i[2], actual, predicted, err, i[3])
        #input()
        #print(err)
        importantD = [i[0], i[1], i[2], actual, predicted, err, i[3], i[4]]

        importantD[1] = sum(importantD[1])/len(importantD[1])
        if(dronePos != data["Z"][index]):
            keptDronePos = dronePos
            dronePos = data["Z"][index]
        if(i[3] == batchN):
            total = 0
            for number in i[1]:
                total += number
            distanceSum.append([total/len(i[1]), a])
            actualDist.append(actual)
            allSignals.append(i[0])
            importantDs.append(importantD)
            a += 1


        else:
            for e in range(0,len(importantDs)):
                addTotals = []
                for j in range(0,len(distanceSum)):
                    if(e == distanceSum[j][1]): continue
                    addTotals.append(distanceSum[j][0])
                importantDs[e].insert(3, addTotals)
            #print(distanceSum, actualDist, allSignals)
            # This is where we calculate the value between that will be between -1 to 1, that will tell us
            # how big the error could be
            # The next value then will tell us how likely the error is
            # Those two values are going to be used by the machine along with the signal, to predict the error
            # Then the error is going to be added to the calcualted distance, giving a better result for the RSSI ~ Distance
            utmCord = utm.from_latlon(keptDronePos[0],keptDronePos[1])
            inside = pointToSection(utmCord[0], utmCord[1], sections)
            if(inside != -1):
                print("\n")
                print("This is the data before selection: ")
                print(importantDs)
                print("\n")
                identified = []
                if(len(importantDs) < 3 or len(importantDs[0][3])== 0):
                    canCont = True
                else:
                    canCont = False
                while canCont == False:
                    #print("Started again, mate!")
                    errorCalcs = []
                    newImportants = []
                    for important in importantDs:
                        newImportant = error_calculation(important, identified)
                        errorCalcs.append(newImportant[1])
                        newImportants.append(newImportant[0])
                    #print("\n")
                    #print(newImportants)
                    print(errorCalcs)
                    #input()
                    print("\n")
                    if(len(newImportants) == 3):
                        canCont = True
                    else:

                        newImportants, identified = removeHighestError(newImportants, errorCalcs, identified)
                        #Remove the commenting to from the testLowest, to test what would be the value,
                        #if we chose all the best values for trilateration, turns out the function is working, just
                        #trilateriation is not that powerful haha.

                        #newImportants = testLowest(newImportants)

                        #print("\n")
                        #print("This is new Importants! After removing the highest error!")
                        #print(newImportants)
                        #print(identified)
                        #print("\n")
                        #input()
                        importantDs = newImportants
                        #input()
                        '''
                        for toPop in maxIndices:
                            print("This is ",toPop)
                            popped = newImportants.pop(toPop)
                            popped = popped[1]
                        importantDs = newImportants
                        '''
                        #popped = newImportants.pop(maximumErrorIndex)
                        #popped = popped[1]
                        #print(popped)
                        #importantDs = removeHighestError(newImportants, popped)
                print("So this is going to be used for Trilateration: ")

                '''
                    Gathering the data for a trilateration json ahhahaha
                '''
                fileData = {}
                for fileE in importantDs:
                    fileData[fileE[len(fileE)-1]] = fileE[0]
                #print(fileData)
                outerFileData = {}
                outerFileData["time"] = random.randint(0,1000000)
                outerFileData["tag"] = random.randint(0,1000000)
                outerFileData["data"] = fileData

                fileX.append(outerFileData)
                fileY.append(keptDronePos)
                print(importantDs)
                print("\n")
                #input()
            #Decide if the idea is good or not here, batch data gathered
            maxi2 = 0
            mini2 = 10000000
            for d in range(0, len(actualDist)):
                if(maxi2 < actualDist[d]):
                    maxi2 = actualDist[d]
                    maxi2Index = d
                if(mini2 > actualDist[d]):
                    mini2 = actualDist[d]
                    mini2Index = d
            #Playing around with the maximum a bit
            maxi = distanceSum[maxi2Index][0] * 1.35
            maxiIndex = maxi2Index
            mini = distanceSum[mini2Index][0] * 0.65
            miniIndex = mini2Index

            for n in range(0,len(distanceSum)):
                if(maxi < distanceSum[n][0]):
                    maxi = distanceSum[n][0]
                    maxiIndex = n
                if(mini > distanceSum[n][0]):
                    mini = distanceSum[n][0]
                    miniIndex = n
            if(maxi2Index == maxiIndex):
                #print("OMG TRUE!")
                goodCount += 1

            else:
                #print("You son of a bitch that is not true")
                badCount += 1
            if(mini2Index == miniIndex):
                goodCount2 +=1
            else:
                badCount2 += 1
            #Reset the batch data, start with new batch
            a = 0
            batchN = i[3]
            distanceSum = []
            actualDist = []
            allSignals = []
            importantDs = []
            total = 0
            for number in i[1]:
                total += number
            distanceSum.append([total/len(i[1]),a])
            actualDist.append(actual)
            allSignals.append(i[0])
            importantDs.append(importantD)
            a += 1

        iput1.append(np.array(i[0]))
        output.append(np.array(err))
    print(goodCount, badCount)
    print(goodCount2, badCount2)
    input()
    print("\n")
    print("\n")
    for i in minimals:
        print(i)
        input()

    assert(len(fileX) == len(fileY))
    finalData = {}
    finalData['X']=fileX
    finalData['y']=fileY

    #Save the file in a json, that will be used for modelling / training
    with open("../Data/"+month+"/trilatData.json","w+") as f:
            json.dump(finalData, f)
    return iput1, output


def loadANNData_2():
    '''
        Contains the data in format that the machine's gonna learn on.
    '''
    month = "June"
    pathName = "../Data/"+month+"/distanceNNData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    iput1 = []
    iput2 = []
    iput3 = []
    output = []
    maxi = 0
    for index, i in enumerate(data["X"]):
        #temp0 = []
        temp0_ind = 0
        temp1_ind = 0
        '''
        while len(temp0) != 15:
            if(temp0_ind == 0):
                temp0.append(i[0])
                temp0_ind += 1
            else:
                temp0.append(np.float64(0))
        '''
        if(i[0] < -102): continue
        #print(i[0], "and the actual distance away is: ", data['y'][index])
        #print("\n")
        #print(i[1])
        #print("\n")
        #print(i[2])
        #input()
        while len(i[1]) != 16:
            i[1].append(np.float64(0))
        while len(i[2]) != 17:
            if(temp1_ind == 0):
                i[2].insert(0,i[0])
                temp1_ind +=1
            else:
                i[2].append(np.float64(0))
        i[1].insert(0,np.float64(i[0]))
        iput1.append(np.array(i[0]))
        iput2.append(np.array(i[1]))
        iput3.append(np.array(i[2]))
        output.append(np.array(data['y'][index]))
    return iput1, output
def deviation(signals, signal):
    total = 0
    for i in range(0,len(signals)):
        total = total + pow((signals[i]-signal),2)
    return math.sqrt(total/len(signals))

def getSignals(arr):
    keys = arr.keys()
    returned = {}
    for key in keys:
        returned[key] = arr[key][2]
    return returned


def getNodedistances(arr):
    nodes = loadNodes(rewriteUTM=True)
    keys = arr.keys()
    skipKeys = []
    #Adds the NodeIDs that has signals out of the equation range -103<RSSI<-72
    for keyS in keys:
        if(keyS == "3288000000"): key="3288e6"
        currentNode = arr[keyS]
        if(arr[keyS][2][0] < -101):
            skipKeys.append(keyS)
        if(arr[keyS][2][0] > -73):
            skipKeys.append(keyS)
    #This loop gathers the data so that the loadANNData functions can use it
    #Returns a list, in the form [[Nodes distances away from the current node],[position of the current node],[signal corresponding to the node]]
    #Also returns the list that contains the taken out NodeIDs
    for key in keys:
        if(key == "3288000000"): key="3288e6"
        if(key in skipKeys):continue
        currentNode = arr[key]
        nodeDistances=[]
        currentNodePos = [nodes[key]['Latitude'],nodes[key]['Longitude']]
        utmCurrentNodePos = utm.from_latlon(currentNodePos[0], currentNodePos[1])
        utmCurrentNodePosArr = np.array([np.float64(utmCurrentNodePos[0]),np.float64(utmCurrentNodePos[1])])
        for node in nodes:
            if(node == key): continue
            if(node not in keys): continue
            if(node in skipKeys): continue
            nodePos = [nodes[node]['Latitude'],nodes[node]['Longitude']]
            utmNodePos = utm.from_latlon(nodePos[0], nodePos[1])
            utmNodePosArr = np.array([np.float64(utmNodePos[0]),np.float64(utmNodePos[1])])
            aDist = np.linalg.norm(utmNodePosArr-utmCurrentNodePosArr)
            nodeDistances.append(aDist)
        arr[key][0] = nodeDistances
        arr[key][2] = arr[key][2][0]
    return arr, skipKeys


def LoadMLPData(month="June"):
    #This data will be loading RSS values and coordinates, so that the machine can 'learn' on it
    pathName = "../Data/"+month+"/newData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    batchSignals = []
    targetPos = []

    for index, item in enumerate(data["X"]):

        #Keys please
        keys = item["data"].keys()
        signals= []
        newSignals = []
        #Loop through each item in X data set
        for key in keys:
            dronePos = (data["y"][index][0],data["y"][index][1])
            #print(item["data"][key][2])
            #input()
            if(dronePos == (0,0) or dronePos == (0.754225181062586, -12.295563892977972) or dronePos == (4.22356791831445, -68.85519277364834)): continue
            signal = item["data"][key][2][0]
            signals.append(signal)
            convertedPos = utm.from_latlon(data["y"][index][0],data["y"][index][1])
            dronePos = [np.float64(convertedPos[0]),np.float64(convertedPos[1])]
        if len(signals) > 3:
            for i in range(0,3):
                maxi = max(signals)
                newSignals.append(np.float64(maxi))
                signals.remove(maxi)

            #newSignals = np.array(newSignals)
            #dronePos = np.array(dronePos)
            batchSignals.append(newSignals)
            targetPos.append(dronePos)
    assert(len(targetPos) == len(batchSignals))
    #finalData = {}
    #finalData['X']=batchSignals
    #finalData['Y']=targetPos
    #with open("../Data/June/MLPData.json","w+") as f:
    #    json.dump(finalData, f)
    #batchSignals = np.array(batchSignals)
    #targetPos = np.array(targetPos)
    return batchSignals, targetPos


def loadANNData(month="June"):
    pathName = "../Data/"+month+"/newEquationData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    values=[]
    distances=[]
    count = 0
    for i in range(0,len(data["Y"])):
        if(data["Y"][i] < -102):
            count += 1
            continue
        values.append(np.float64([data["Y"][i]]))
        distances.append(np.float64(data["X"][i]))

    print("This much was taken out:", count)
    X = np.array(values)
    Y = np.array(distances)
    #Y = Y.astype(int)
    return X, Y





def newEquationData(month="June"):
    #I believe it would be nice to have it like this
    #We can derive other equations from other months as well if needed
    pathName = "../Data/"+month+"/newData.json"
    with open(pathName,"r") as f:
        data = json.load(f)

    #signal strength -> y values
    signalStr = []
    #drone distance from nodes -> x values
    distances = []

    #Loop through the items in data set X
    for index, item in enumerate(data["X"]):

        #Get the keys
        keys = item["data"].keys()

        #Loop through each item in X data set
        for key in keys:

            #Set pointA to be the location of the drone
            pointA = (data["y"][index][0],data["y"][index][1])

            #Loop through the each node, to get the signal and distance
            for i in range(0,len(item["data"][key][2])):

                pointB = (item["data"][key][1][0],item["data"][key][1][1])

                #Using vincenty alorithm to calculate the distance between them in km
                #It is said to be quite accurate, metre specific accuracy
                distance = vincenty(pointA, pointB)
                #Exclude the useless data points
                if(pointA == (0,0) or pointA == (0.754225181062586, -12.295563892977972) or pointA == (4.22356791831445, -68.85519277364834)): continue
                #Otherwise we convert into meters
                distance = distance*1000

                #Set the signal
                signal = item["data"][key][2][i]

                #Add the signal value to the y values
                signalStr.append(signal)

                #Add distance to the x values
                distances.append(distance)

    #Make sure that the values are the same length; suitable for plotting
    assert(len(signalStr) == len(distances))
    finalData = {}
    finalData['X']=distances
    finalData['Y']=signalStr
    with open("../Data/June/newEquationData.json","w+") as f:
        json.dump(finalData, f)

def deriveEquation(month="June"):

    #Open the newEquationData file, rune newEquationData to get it
    pathName = "../Data/"+month+"/newEquationData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    #Create a data frame, which will be used to make a csv file
    #and to optimize an equation
    dataFrame = {}
    dataFrame["Y"] = data["Y"]
    dataFrame["X"] = data["X"]

    #Creating csv file so that we can monitor it if wanted to
    df = pd.DataFrame(dataFrame)
    df.to_csv("../Data/June/newEqData.csv")

    #Setting values to optimize the equation
    df = pd.DataFrame({"x": data["X"], "y": data["Y"]})
    x = np.array(df["x"])
    y = np.array(df["y"])

    # Have an initial guess as to what the values of the parameters are
    a_guess = 47.23
    b_guess = -0.005
    c_guess = -105.16

    # Fit the function a * np.exp(b * t) + c to x and y
    popt, pcov = curve_fit(
        lambda t, a, b, c: a * np.exp(b * t) + c,
        x, y, p0=(a_guess, b_guess, c_guess)
    )

    # The optimised values of the parameters are
    a = popt[0]
    b = popt[1]
    c = popt[2]

    return([a,b,c])
'''
    Part for the algorithm that tries to optimize the signal strength values - currently not working properly

'''

'''
    Main function for rewriting the values. This one is a recursive function, which updates a list's distances+signalsternght if the condition holds.
    What I found out that 1 signal strenght is roughly 1 signal difference, so that is why I tried to find it till 200m roughly.
    There is a problem with this - if all values are way off, then this function will just worsen the situation. Again, might be good to improve this somehow, or
    find another approach.
'''
def rewrite(values, valuesUpdated):
    #Base case
    if(len(values) == 1):
        return valuesUpdated
    mini, index = value_finder(values)

    #If there is disonance, then does the swapping
    if(check_dissonance(values, index) == True):
        #print("Dissonance alert!")
        return_index, closestDist = closest_to(values, index)
        if(abs(values[index][1]-values[return_index][1]) > 3):
            #print("Dissonance for sure!")
            closestDist = abs(values[index][0] - values[return_index][0])
            #print("ClosestDistance:", closestDist)
            negative = bool((values[index][0] - values[return_index][0]) < 0)
            if(closestDist < 50):
                values[index][1] = values[return_index][1]
            elif(closestDist< 100 and closestDist>=50):
                if(negative == True):
                    values[index][1] = values[return_index][1] - 1
                else:
                    values[index][1] = values[return_index][1] + 1
            elif(closestDist< 150 and closestDist>=100):
                if(negative == True):
                    values[index][1] = values[return_index][1] - 2
                else:
                    values[index][1] = values[return_index][1] + 2
            elif(closestDist < 200 and closestDist>=150):
                if(negative == True):
                    values[index][1] = values[return_index][1] - 3
                else:
                    values[index][1] = values[return_index][1] + 3
            else:
                if(negative == True):
                    values[index][1] = values[return_index][1] - 4
                else:
                    values[index][1] = values[return_index][1] + 4
            if(values[index][1] < -102): values[index][1] = -102
            if(values[index][1] > -73): values[index][1] = -73
            for i in range(0,len(valuesUpdated)):
                if(values[index][0] == valuesUpdated[i][0]):
                    valuesUpdated[i][1] = values[index][1]
                    valuesUpdated[i][2] = calculateDist_2(valuesUpdated[i][1])

    values.remove(values[index])
    #print("This is values updated! ",valuesUpdated)
    #print("Values: ", values)
    #input()

    return rewrite(values, valuesUpdated)

'''
    This function checks whether there is 'dissonance' or not. Currently dissonance is defined as the following:
        -If the signal value that is the lowest, does not have the highest sum of distances (sum of the dist that the node is away from the other nodes)
         then there is probably a dissonance happening. Since the assumption is that if the node is the farthest away from the other nodes, then it should
         have the lowest signal. Obviously there are cases, when this might not be true - thus there is another conditon, which is if the signal strenght
         difference is more than 2, then we can conclude that it's probably a dissonance. Idea is currently not working, on the one hand for some reason,
         even though the distances are optimzed and are closer to the actual distances the nodes are away from the drone, it sometimes gives an error that
         is way off. Perhaps, gps_solve optimization does not work properly, then?? If these two methods cancel each other, then need to have a look at another
         idea. (Currently it seems that they do cancel each other out.)
'''
def check_dissonance(values, index):
    maxi = value_finder_2(values, index)
    #print("This is maxi!", maxi)
    #print("This is current value: ", values[index][0])
    if(maxi > values[index][0]):
        diss = True
    else:
        diss = False
    return diss
'''
    This function finds the highest distance (distance means sum of the distances that the node is away from the other nodes) and returns that value
'''
def value_finder_2(values, index):
    maxi = float('-inf')
    for i in range(0,len(values)):
        if(i == index): continue
        if(maxi < values[i][0]):
            maxi = values[i][0]
    return maxi

'''
    This function finds the lowest signal value in a list and returns its index and the value
'''
def value_finder(values):
    index = None
    mini = float('inf')
    for i in range(0,len(values)):
        if(values[i][1] < mini):
            mini = values[i][1]
            index = i
    return mini, index

def closest_to(values, index):
    keepIt = []
    for i in range(0, len(values)):
        if(i == index): continue
        keepIt.append([i, abs(values[i][0]-values[index][0])])
    mini = float('inf')
    return_index = None
    for j in range(0, len(keepIt)):
        if(keepIt[j][1] < mini):
            mini = keepIt[j][1]
            return_index = keepIt[j][0]
    return return_index, mini

'''
    End of the Part for the new type of approach.
'''
def calculateDist_3(RSSI):
    model = pickle.load(open('anndistance.sav','rb'))
    y = model.predict([[np.float64(RSSI)]])
    return y[0]
def calculateRSSI(distance):
    rssi = (np.exp((distance*-0.00568791789392432))*29.797940785785794)-102.48720932300988
    return rssi

def calculateDist_2(RSSI):
    # formula from our data
    #New values
    #29.797940785785794, -0.00568791789392432, -102.48720932300988
    #Old values
    #29.856227966632954, -0.0054824231026629686, -102.84339991053513
    #values = deriveEquation()
    #dist = np.log((RSSI-values[2])/values[0])/values[1]
    dist = np.log((RSSI+102.48720932300988)/29.797940785785794)/-0.00568791789392432
    return dist


def calculateDist(RSSI):
    """ Given the RSS, plug it into the exponential formula """
    # formula from Paxton et al
    dist = np.log((RSSI+105.16)/47.23)/-.005
    return dist


if __name__=="__main__":
    #associateJuneData(newData=True)
    #deriveEquation()
    #loadModelData()
    #rewriteMarchData()
    #print(multilat.predictions(rssiThreshold=-102,keepNodeIds=True, isTriLat = False, month="June"))
    #associateJuneData(newData=True)
    #LoadMLPData()
    #loadANNData()
    #LoadSeqData()
    #loadANNData_2()
    loadANNData_3()
    #FunctionalModel()

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
    #Currently using another json, so we can test the function, that was written.
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
    Part for the algorithm that tries to optimize the signal strength values - working to an extent - but I don't think this is the path forward.
    We can keep this for negative results.

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

'''
    Function that can be used to calculate the signal from distance. This will be relevant for the proximity kind of model.
'''
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
    print(multilat.predictions(rssiThreshold=-102,keepNodeIds=True, isTriLat = False, month="June"))

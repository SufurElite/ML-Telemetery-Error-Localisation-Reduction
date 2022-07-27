"""
    A file that contains functions related to exploratory data analysis.
"""
import numpy as np
import pandas as pd
import math
import utils
import json, utm
import datetime
import multilat
from vincenty import vincenty
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

def closestNodeCount(month="March"):
    """ This function goes over the data, specified by month, and then compares the
        Node that was closest by the RSS to Distance approximation versus the actual closest node
        (the closest node that received a signal). We can determine this by going over all the associated test data
        (the signals to each node for a given test) and the location's ground truth, and find the smallest RSS distance
        and the smallest actual distance to the node (since we know the nodes' locations and the ground truth). And then
        at the end compare the number that were correctly determined, miscatgeroized as each other, etc.
        """
    # Load in the associated test data and the nodes data
    data = utils.loadData()
    nodes = utils.loadNodes()

    # wrongCount is a counter for the number of nodes that were not the closest
    # when estiamted to be
    wrongCount = 0
    # approximatedCount is a counter for the number of nodes correctly predicted
    approximatedCount = 0
    # wasSecondClosest is a counter of the number of nodes that were second closest predicted by RSS Estimate
    wasSecondClosest = 0

    # frequencyConfused is a map of the frequencies that particular nodes are confused with specific other nodes
    frequencyConfused = {}

    for i in data.keys():
        # Get the ground truth for the UTMx and UTMy
        dataUTMx = data[i]["GroundTruth"]["TestUTMx"]
        dataUTMy = data[i]["GroundTruth"]["TestUTMy"]

        # format for pair1, secondPredicted, and pair2 [distance, gtDistance, nodeId]
        # pair 1 will hold the smallest the RSS estimated distance (and secondPredicted will
        # hold the second smallest RSS estimated distance)
        pair1 = [1200, 0, ""]
        secondPredicted = [1200, 0, ""]
        # pair 2 will hold the actual smallest between the test location and node
        pair2 = [1200, 0, ""]

        # go over all the test's associated data to find the smallest
        for j in range(len(data[i]["Data"])):
            # filter out signals that won't work with the formula
            if data[i]["Data"][j]["TagRSSI"]<=-105.16: continue
            if data[i]["Data"][j]["NodeId"] not in nodes.keys(): continue

            nodeUTMx = nodes[data[i]["Data"][j]["NodeId"]]["NodeUTMx"]
            nodeUTMy = nodes[data[i]["Data"][j]["NodeId"]]["NodeUTMy"]
            # calculate RSS distance
            rssDist = utils.calculateDist(data[i]["Data"][j]["TagRSSI"])
            # ground truth distance
            nodeUTM = np.vstack((nodeUTMx, nodeUTMy)).T
            tagUTM = np.vstack((dataUTMx, dataUTMy)).T
            gtDist = np.linalg.norm(nodeUTM-tagUTM)

            # check if the estimated distance is less than current
            # and reassign values
            if rssDist<pair1[0]:
                secondPredicted[0] = pair1[0]
                secondPredicted[1] = pair1[1]
                secondPredicted[2] = pair1[2]
                pair1[0] = rssDist
                pair1[1] = gtDist
                pair1[2] = data[i]["Data"][j]["NodeId"]
            # check if the actual distance is less than current
            # and reassign values
            if gtDist<pair2[0]:
                pair2[0] = gtDist
                pair2[1] = rssDist
                pair2[2] = data[i]["Data"][j]["NodeId"]
        # after all the associated test data, check if the pairs aren't the same
        if pair1[0]!=pair2[1] and pair2[0]!=pair1[1] and pair1[2]!=pair2[2]:
            # if the nodes aren't not in the frequencyConfused dictionary add them
            if pair1[2] not in frequencyConfused.keys():
                frequencyConfused[pair1[2]] = {}
            if pair2[2] not in frequencyConfused.keys():
                frequencyConfused[pair2[2]] = {}
            # if they aren't in each other's dictionary, then add them
            if pair2[2] not in frequencyConfused[pair1[2]].keys():
                frequencyConfused[pair1[2]][pair2[2]] = 0
            if pair1[2] not in frequencyConfused[pair2[2]].keys():
                frequencyConfused[pair2[2]][pair1[2]] = 0

            # regardless, update the frequency by 1
            frequencyConfused[pair1[2]][pair2[2]]+=1
            frequencyConfused[pair2[2]][pair1[2]]+=1

            # add 1 to the number of incorrectly guessed
            wrongCount+=1
            # if the actual one was the second closest, increment counter
            if secondPredicted[2] == pair2[2]:
                wasSecondClosest+=1
        else:
            # otherwise the estimation was right
            approximatedCount+=1

    print("""The number incorrectly determined by RSS->distance was {}\nThe number correctly determined by RSS->distance was {}\nThe number of second closest determined by RSS->distance was {}.\n""".format(wrongCount, approximatedCount, wasSecondClosest))
    print("Here is the dictionary of each node that was misclassified and the number of misclassifications they had with other nodes:")
    print(frequencyConfused)
def checkBatches():
    pathName = "../Data/"+month+"/newData.json"
    with open(pathName,"r") as f:
        data = json.load(f)

def compareDistanceCalculator(distanceFunction,rssiThreshold=-105.16):
    """
        This function will evaluate the overall accuracy of a provided RSSI->distance function.
        The values that are collected here should indicate the overall distance error compared to reality,
        as well as the for each individual node.

        (Right now just works with June)
    """
    data = utils.loadData("June")
    nodes = utils.loadNodes(True)
    X = data["X"]
    y = data["y"]


    totalDistanceError = [0,0]
    nodeDistErrorFreq = {}
    for node in list(nodes.keys()):
        nodeDistErrorFreq[node] = [0,0]

    for i in range(len(X)):
        """ determine the actual distance from the tag to each node
            and then the approximated one by the provided function"""
        # get the utm location of the drone
        utmGT = utm.from_latlon(y[i][0], y[i][1])
        gt = np.array([np.float64(utmGT[0]),np.float64(utmGT[1])])
        for dataEntry in X[i]["data"].keys():
            nodeId = dataEntry
            if X[i]["data"][dataEntry] <=rssiThreshold: continue
            if dataEntry=="3288000000": nodeId="3288e6"
            # find the current nodes location and estimated value
            nodeLoc = np.array([np.float64(nodes[nodeId]['NodeUTMx']),np.float64(nodes[nodeId]['NodeUTMy'])])
            actualDist = np.linalg.norm(nodeLoc-gt)
            # find the approximated distance by the function provided
            approxDist = distanceFunction(X[i]["data"][dataEntry])
            nodeDistErrorFreq[nodeId][0]+=(abs(actualDist-approxDist))
            nodeDistErrorFreq[nodeId][1]+=1

            totalDistanceError[0]+=(abs(actualDist-approxDist))
            totalDistanceError[1]+=1

    print("With an RSSI threshold of {}".format(rssiThreshold))
    print("and given the provided RSSI -> distance function, the following results were obtained:")
    for node in nodeDistErrorFreq.keys():
        if nodeDistErrorFreq[node][0]==0: continue
        print("\t{} had an average error of {}m".format(node, (nodeDistErrorFreq[node][0]/nodeDistErrorFreq[node][1])))
    print("Thus the function had a total overall average error of {} m".format((totalDistanceError[0]/totalDistanceError[1])))

if __name__=="__main__":
    #closestNodeCount()
    #random()
    #dataObt()
    print(newEquation())
    #plotEquation()
    #compareDistanceCalculator(distanceFunction,rssiThreshold=-105.16)
    #checkBatches()

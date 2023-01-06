"""
    A file that contains functions related to exploratory data analysis.
"""
import numpy as np
import utils
import json, utm
import multilat
from vincenty import vincenty

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

def orderedStrengths(saveFilteredDist=False, month="June"):
    """
        Using the new data, we bucket the errors to see how far off 
        the distances are between [0,20], (20,50], (50,75], (75,100), (100, inf)
        And we also save the distances and the signal strengths to graph on an x,y cartesian plot
    """
    pathName = "../Data/"+month+"/newData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    X = []
    y = data["y"]
    #signal strength -> y values
    signalStr = []
    #drone distance from nodes -> x values
    distances = []
    #Bucket the errors - 0-20m, 21m-50m, 51m-75m, 75m-100m, 100m<
    errors = [0,0,0,0,0]
    #Loop through the items in data set X
    for index, item in enumerate(data["X"]):
        #Get the keys
        keys = item["data"].keys()

        #Loop through each item in X data set
        newItems = {}
        for key in keys:
            if item["data"][key][2][0] < -102 or item["data"][key][2][0] > -73: continue
            #Set pointA to be the location of the drone
            pointA = (data["y"][index][0],data["y"][index][1])
            pointB = (item["data"][key][1][0],item["data"][key][1][1])

            #Exclude the useless data points
            if(pointA == (0,0) or pointA == (0.754225181062586, -12.295563892977972) or pointA == (4.22356791831445, -68.85519277364834)): continue

            # Using vincenty alorithm to calculate the distance between them in km
            distance = vincenty(pointA, pointB)
            #Convert into meters
            distance = distance*1000

            #Set the signal
            distance = round(distance,2)
            signal = item["data"][key][2][0]
            
            
            distance_2 = round(utils.calculateDist_2(signal),2)
            replace = [signal,distance,distance_2]
            newItems[key] = replace

            distances.append(distance)
            signalStr.append(signal)
        
        # Now bucket the errors
        sort_orders = sorted(newItems.items(), key=lambda x: x[1][1])
        for i in sort_orders:
            print(i[1])
            print(i[1][1], i[1][2])
            if(i[1][1] > i[1][2]):
                err = i[1][1] - i[1][2]
                if(i[1][0] > -73):
                    print(i)
                    print(sort_orders)
                    input()
            else:
                err = i[1][2] - i[1][1]
            if(err > 100):
                if(errors[4] == 0):
                    errors[4] = {}
                    errors[4][i[1][0]] = 1
                else:
                    if(i[1][0] not in errors[4].keys()):
                        errors[4][i[1][0]] = 1
                    else:
                        errors[4][i[1][0]] += 1
            elif(err <= 100 and err >75):
                if(errors[3] == 0):
                    errors[3] = {}
                    errors[3][i[1][0]] = 1
                else:
                    if(i[1][0] not in errors[3].keys()):
                        errors[3][i[1][0]] = 1
                    else:
                        errors[3][i[1][0]] += 1
            elif(err <=75 and err > 50):
                if(errors[2] == 0):
                    errors[2] = {}
                    errors[2][i[1][0]] = 1
                else:
                    if(i[1][0] not in errors[2].keys()):
                        errors[2][i[1][0]] = 1
                    else:
                        errors[2][i[1][0]] += 1
            elif(err <=50 and err > 25):
                if(errors[1] == 0):
                    errors[1] = {}
                    errors[1][i[1][0]] = 1
                else:
                    if(i[1][0] not in errors[1].keys()):
                        errors[1][i[1][0]] = 1
                    else:
                        errors[1][i[1][0]] += 1
            else:
                if(errors[0] == 0):
                    errors[0] = {}
                    errors[0][i[1][0]] = 1
                else:
                    if(i[1][0] not in errors[0].keys()):
                        errors[0][i[1][0]] = 1
                    else:
                        errors[0][i[1][0]] += 1
            print(err)
            print(errors)

    #Make sure that the values are the same length; suitable for plotting
    assert(len(signalStr) == len(distances))
    finalData = {}
    finalData['X']=distances
    finalData['Y']=signalStr
    if saveFilteredDist:
        with open("../Data/June/newDataDistanceFiltered.json","w+") as f:
            json.dump(finalData, f)

def closest_to(number, numbers):
    """ Helper function to find the two indices of the numbers closest to provided number """
    values = []
    for i in range(0,len(numbers)):
        values.append(abs(numbers[i] - number))
    mini_one = float('infinity')
    mini_two = float('infinity')
    one = 0
    two = 0
    for k in range(0,len(values)):
        if(values[k] < mini_one):
            mini_one = values[k]
            one = k
        if(values[k] < mini_two and mini_one != values[k]):
            mini_two = values[k]
            two = k
    return one, two

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

def dataGridSections(month="June"):
    """ Check the frequency of the data from a given month in particular sections of the grid """
    # load in the varying values
    grid, sections, nodes = utils.loadSections()
    # get the lat longs of the month
    latLongs = utils.loadData(month)["y"]
    # Initialize the frequency map for each key
    sectionFrequency = {}
    for i in range(len(sections.keys())):
        sectionFrequency[i] = 0
    # -1 will refer to if data point is outside the grid
    sectionFrequency[-1] = 0
    # go through all the lat longs and determine the section
    for latLong in latLongs:
        # Ignore these particular points
        if(latLong == [0,0] or latLong == [0.754225181062586, -12.295563892977972] or latLong == [4.22356791831445, -68.85519277364834]): continue
        # calculate the utm and then determine its section
        utmVals = utm.from_latlon(latLong[0], latLong[1])
        sectionNum = utils.pointToSection(utmVals[0], utmVals[1], sections)
        sectionFrequency[sectionNum]+=1
    print("The section to number of data points for the month of {} is as follows : ".format(month))
    for sectionKey in sectionFrequency.keys():
        print("\t{}:{}".format(sectionKey, sectionFrequency[sectionKey]))

def findAvgTrueError(verbose = False):
    """ Given all the actual distances to the nodes,
        calculate multilat and determine the error """
    # Loads in data where the X is the calculated dist to each node
    # and y is the actual dist to each node
    X, y = utils.loadRSSModelData(month = "June")
    # Generate the node locations in a list
    nodes = utils.loadNodes(True)
    nodeKeys = list(nodes.keys())
    nodeLocs = []
    for key in nodeKeys:
        nodeLocs.append([nodes[key]["NodeUTMx"],nodes[key]["NodeUTMy"]])

    errors = [0,0]
    for i in range(len(X)):
        # get a given row of data
        tmp_distances = X[i]
        for j in range(len(tmp_distances)):
            # if the node had data to a batch, set it to the actual distance
            if tmp_distances[j]!=0:
                tmp_distances[j] = y[i][j]
        # calculate the estimated & actual multilats
        estimated = np.array(multilat.gps_solve(tmp_distances, list(np.array(nodeLocs))))
        actual = np.array(multilat.gps_solve(y[i], list(np.array(nodeLocs))))
        # find the difference in the predicted location
        dist = np.linalg.norm(estimated-actual)
        errors[0]+=dist
        errors[1]+=1
        if verbose:
            print("{}, current error : {}, running average error : {}".format(i, dist, errors[0]/errors[1]))
    # show the average error
    print(errors[0]/errors[1])

if __name__=="__main__":
    dataGridSections()

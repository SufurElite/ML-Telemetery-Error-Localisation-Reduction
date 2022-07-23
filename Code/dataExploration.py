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
def dataObt(month="June"):
    #testInfo = pd.read_csv(r'../Data/'+month+'/TestInfo.csv')
    #beepData = pd.read_csv(r'../Data/'+month+'/BeepData.csv')

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
    nodeLocations = utils.loadNodes(rewriteUTM=True)
    '''
    for node in nodeLocations:
        print(node)
    print(nodeLocations["3288e6"]["NodeUTMx"])
    '''
    '''
    known = []
    for tag in beepData['TagId']:
        if tag not in known:
            known.append(tag)
    print(known)
    '''
    flightDF = pd.concat(flightDataList, axis=0, ignore_index=True)
    flightDF['datetime(utc)'] = pd.to_datetime(flightDF['datetime(utc)'])

    '''
    for time in flightDF['datetime(utc)']:
        print(time)
    '''
    # X will be composed of batches of data
    X = []
    # y will be composed of X items corresponding GT flight values
    Y = []

    batch = {}
    baseTime = '0'
    tag = ''

    for index, row in enumerate(beepData.iterrows()):
        #Don't want to deal with the conversion, so I convert it here
        row[1]['Time.local'] = datetime.datetime.strptime(row[1]['Time.local'], "%Y-%m-%dT%H:%M:%SZ")
        currDate = row[1]['Time.local']
        if row[1]['NodeId']=="3288000000": row[1]['NodeId']="3288e6"

        #print(index)
        #print(currDate)
        #print(row)

        #input()
        if baseTime == '0':
            baseTime = currDate
            tag = row[1]['TagId']
        elif tag!=row[1]['TagId'] or (currDate-baseTime>datetime.timedelta(0,2)):
            upperBound = baseTime+datetime.timedelta(0,2)
            #print("Hello!")
            #print(baseTime,currDate)
            timeSort = flightDF[flightDF['datetime(utc)'].between(baseTime,upperBound)]
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
                    data["data"][i] = batch[i]
                    #print(data["data"])
                    #input()
                X.append(data)
                # get the average latitude over the 2 seconds
                avgLat = tagSort["latitude"].mean()
                # get the average longitude over the 2 seconds
                avgLong = tagSort["longitude"].mean()
                avgAlt = tagSort["altitude_above_seaLevel(feet)"].mean()
                #coordinates = utm.from_latlon(avgLat,avgLong)
                #Y.append([coordinates[0],coordinates[1]])
                Y.append([avgLat,avgLong,avgAlt])
                #print(X, Y)
                #input()

            #Reset
            batch = {}
            baseTime = currDate
            tag = row[1]['TagId']
        if row[1]['NodeId'] not in batch.keys():
            batch[row[1]['NodeId']]=[0,0,[]]
            batch[row[1]['NodeId']][0] +=1
            name = row[1]['NodeId']
            batch[row[1]['NodeId']][1] = [nodeLocations[name]['Latitude'],nodeLocations[name]['Longitude']]
            batch[row[1]['NodeId']][2].append(row[1]['TagRSSI'])
        elif row[1]['NodeId'] in batch.keys():
            batch[row[1]['NodeId']][0] +=1
            batch[row[1]['NodeId']][2].append(row[1]['TagRSSI'])
        #print(batch)
        #print("The time interval: ", currDate-baseTime)
    assert(len(Y) == len(X))
    finalData = {}
    finalData['X']=X
    finalData['Y']=Y
    with open("../Data/June/newData.json","w+") as f:
        json.dump(finalData, f)
def newEquation(month="June"):
    pathName = "../Data/"+month+"/newData.json"
    with open(pathName,"r") as f:
        data = json.load(f)
    #print(data["X"][0]["data"])
    #print("")
    #print(data["Y"][0])
    #for item in data["X"]:
    signalStr = []
    distances = []

    for index, item in enumerate(data["X"]):
        #print(index)
        #print(item)
        keys = item["data"].keys()
        for key in keys:
            pointA = (data["Y"][index][0],data["Y"][index][1])
            for i in range(0,len(item["data"][key][2])):
                pointB = (item["data"][key][1][0],item["data"][key][1][1])
                #print("This is point A:",pointA)
                #print("This is point B:",pointB)
                #Calculates it in km
                distance = vincenty(pointA, pointB)
                distance = distance*1000
                signal = item["data"][key][2][i]
                #print(signal)
                signalStr.append(signal)
                #print(distance)
                distances.append(distance)
                #input()

                #print(item["data"][key][2][i])
                #print(data["Y"][index])
            #print(item["data"][key])
            #print(index)
            #input()
    assert(len(signalStr) == len(distances))

    #Kinda new to the plotting library, so I am testing whether I did not specify the scale properly
    dataFrame = {}
    dataFrame["Y"] = signalStr
    dataFrame["X"] = distances
    df = pd.DataFrame(dataFrame)
    df.to_csv("../Data/June/newEq.csv")

    df=pd.DataFrame({'x': distances, 'y': signalStr})

    #df_sample=df.sample(1000)
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

def plotEquation():

    # 100 linearly spaced numbers
    x = np.linspace(0,200,10000)
    values = newEquation()
    print(values)
    y = values[0]*np.exp(x*values[1])+values[2]


    # setting the axes at the centre
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    plt.plot(x,y, 'r')

    # show the plot
    plt.show()





def random():
    data = {"3775bf":[2,-107,-107],
            "37774c":[2,-96,-96],
            "379611":[2,-101,-101],
            "328840":[1,-95],
            "37a200":[2,-105,-105],
            "377905":[1,-103],
            "3774a7":[2,-103,-103],
            "328b9b":[2,-103,-103],
            "3290ef":[2,-108,-108],
            "3275dd":[2,-102,-102],
            }
    nodeTags = data.keys()
    print(nodeTags)
    for tag in nodeTags:
        print(tag)
    pos = [4.52534334789879,-73.77342545775063,10834.583218754]
    latD = pos[0]
    longD = pos[1]
    utmD = utm.from_latlon(latD,longD)
    utmDx = utmD[0]
    utmDy = utmD[1]
    nodes =utils.loadNodes(rewriteUTM=False)
    keys = nodes.keys()
    print(nodes['3775bf']['Latitude'])
    for key in keys:
        lat = nodes[key]['Latitude']
        long = nodes[key]['Longitude']
        utmP = utm.from_latlon(lat,long)
        utmx = utmP[0]
        utmy = utmP[1]
        distance = math.sqrt(pow((utmy-utmDy),2)+pow((utmx-utmDx),2))
        if(key in nodeTags):
            print("The distance of node: ", key," form the drone is: ", round(distance,2))



if __name__=="__main__":
    #closestNodeCount()
    #random()
    #dataObt()
    #newEquation()
    plotEquation()

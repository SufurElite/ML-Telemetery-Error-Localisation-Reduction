"""
    A file that contains functions that might be useful across different files, 
    or to perform specific utility operations once (e.g. associating data)
"""
import pandas as pd
import numpy as np
import json, utm

def loadNodes():
    """ Loads in the nodes from the CSV and returns it as a dictionary where the key is its NodeId"""
    # Load in the data from the CSV using pandas
    df = pd.read_csv(r'../Data/Nodes.csv')
    # Convert the dataframe to a list of dictionaries, where each row is a dictionary
    nodesData = df.to_dict('records')
    nodes = {}
    # Reformat the data to a dictionary where the NodeId is the key of the dictionary 
    # rather than an element of a dictionary in a list
    for i in range(len(nodesData)):
        nodes[nodesData[i]["NodeId"]] = nodesData[i]
        del nodes[nodesData[i]["NodeId"]]["NodeId"]
    return nodes

def loadBoundaries():
    """ Will return the boundaries of the node system """
    df = pd.read_csv(r'../Data/Nodes.csv')
    minX = min(df["NodeUTMx"])
    maxX = max(df["NodeUTMx"])
    minY = min(df["NodeUTMy"])
    maxY = max(df["NodeUTMy"])
    return minX,maxX,minY,maxY

def loadData(month="March", pruned=False):
    """ Loads the data that has been stored from associateTestData, 
        month specifies the month the data was taken happened (March or June)
        by default is March """
    pathName = "../Data/"+month+"/associatedTestData"
    if pruned:
        pathName+="Pruned"
    pathName+=".json"
    with open(pathName,"r") as f:
        data = json.load(f)
    return data

def associateTestData(month="March"):
    """ This function will associate the TestIds and its data with the rows
        Takes in a month parameter that specifies the month of the data, by default March"""
    
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

def calculateDist(RSSI):
    """ Given the RSS, plug it into the exponential formula """
    # formula from Paxton et al
    dist = np.log((RSSI+105.16)/47.23)/-.005
    return dist

if __name__=="__main__":
    associateTestData()
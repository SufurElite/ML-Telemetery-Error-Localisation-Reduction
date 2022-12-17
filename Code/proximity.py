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
import utils

from sklearn.model_selection import train_test_split
import random



def getSignals(arr):
    '''
        Function that is used by proximityDataManipulation function - to get all the signals in the specific list.
    '''
    keys = arr.keys()
    returned = {}
    for key in keys:
        returned[key] = arr[key][2]
    return returned


def getNodedistances(arr, month="June"):
    '''
        Function that is used by proximityDataManipulation function. It basically goes through all the signals in a batch
        for a specific signal. Exlcudes any signals that are out of the calcluate_distance range.
    '''
    if(month == "June"):
        nodes = utils.loadNodes(rewriteUTM=True)
    else:
        nodes = utils.loadNodes_46()
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




def proximityDataManipulation(month="June"):
    '''
        Idea is that if we give the machine the distances it is away from other nodes and the signal strength, we would be able to predict the distance it is away from
        the actual node.
        So, basically trying to reproduce the 'proximity' filter, with a possible neural network in the end.
        Thus we would get an even better distance result, that would count the noise / vegetation in as well.
        Since, the distance ~ signalStr equation in itself is not enough to make the prediction to the actual distance be completely accurate.
        This gathers data, from the newData.json

    '''
    if(month =="June"):
        # Load in the June JSON
        pathName = "../Data/"+month+"/newData.json"
        with open(pathName,"r") as f:
            data = json.load(f)
        nodes = utils.loadNodes(rewriteUTM=True)
    else:
        pathName = "../Data/"+month+"/newData.json"
        with open(pathName, "r") as f:
            data = json.load(f)
        nodes = utils.loadNodes_46()

    X = data["X"]
    y = data["y"]
    newX = []
    newY = []
    newZ = []
    '''
    #Going to gather the new data pairs in the form of:
    #Elements of newX will look like: [
    #                                   [signalStrength of the current node in the batch],
    #                                   [all relative node distances in the same batch, that also have relevant signal],
    #                                   [all relative node signals in the same batch, that are relevant]
    #                                   [batch_number]]
    #Elements of newY will look like: [distance between current node and the drone]
    '''
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
        replace, skipKeys = getNodedistances(item["data"], month)

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

def error_calculation(arr, identified):
    '''
        This function uses similar idea as the check_dissonance, but is an entirely idfferent approach.
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
        other signals' errors as well. So, that's why we are re-running this whole thing, after identifying one wrong signal.
        Currently, this function only works for choosing the lowest error producing signals etc.
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

    errorVal = []
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
        diff = currOtherSig[i]-currSig
        addErr = 0
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
                dissonant = True
                if(insideIdentified_2(identified, compare) == False):
                    addNum = (0.1*(percent)/endPercent)
                    numerator += addNum
                    print("Possible dissonance")
                    print(val0, " is less than ", compare, "so we add", addNum)
                    print("Difference between signals is:", diff)
                    addErr = diff*(percent)/endPercent
                else:
                    print("Possible dissonance, but ")
                    print("Inside the identified error signals")
                    addNum = (0.025*(endPercent-percent)/endPercent)
                    numerator += addNum
                    print(val0, "is less than", compare, "but", currOtherSig[i], "is in identified error signals")
                    print("So we add", addNum)
                    print("Difference between signals is:", diff)
                    addErr = diff*(endPercent-percent)/endPercent

            else:
                dissonant = False
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
                dissonant = True
                if(insideIdentified_2(identified, compare) == False):
                    addNum = (0.1*(percent2)/endPercent)
                    numerator += addNum
                    print("Possible dissonance")
                    print(val0, " is bigger than ", compare, "so we add", addNum)
                    print("Difference between signals is:", diff)
                    addErr = diff*(percent2)/endPercent
                else:
                    print("Possible dissonance, but ")
                    print("Inside the identified error signals")
                    addNum = (0.025*(endPercent-percent2)/endPercent)
                    numerator += addNum
                    print(val0, "is bigger than", compare, "but", currOtherSig[i], "is in identified error signals")
                    print("So we add", addNum)
                    print("Difference between signals is:", diff)
                    addErr = diff*(endPercent-percent2)/endPercent

            else:
                dissonant = False
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
            dissonant = False
            if(insideIdentified_2(identified, compare) == False):
                if(val0 > compare):
                    addNum = (0.1*(percent2)/endPercent)
                    addErr = 1*(percent2)/endPercent
                    numerator += addNum
                else:
                    addNum = (0.1 *(percent)/endPercent)
                    addErr = 1*(percent)/endPercent
                    numerator += addNum
            else:
                if(val0 > compare):
                    addNum = (0.05*(percent2)/endPercent)
                    addErr = 1*(percent2)/endPercent
                    numerator += addNum
                else:
                    addNum = (0.05*(percent)/endPercent)
                    addErr = 1*(percent2)/endPercent
                    numerator += addNum
        errorVal.append([addErr,currOtherSig[i],dissonant])
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
        Next step is to get the error guesses for each signal out of this function. But currently, this
        function just serves as an error probability function.
    '''


    #print(numerator)
    #print(size)

    #print(errorDistances)
    #print(round(numerator/size*100,2), "%")

    return [arr, round(numerator/size*100,2), errorVal]

def removeHighestError(arr, arr2, identified):
    '''
        This function after getting the 3 lists: All node data, all node errors, identified node data
        Searches for the highest / lowest error in the node_errors (that are values returned from error_calculation)
        does a deviation on both of them, whichever is higher, adds that signal into the identified data.
        If there are 0 values in identified it does things differently. Creates a list called arr2_sorted, then
        does deviation on the lowest and highest. Then adds whichever is higher to the identified list.

        If there are more than 0 values in the identified then, again find the lowest and highest, but it also excludes
        the error_values that are already inside the identified list. So then again does the same thing as previously mentioned.

        Then if the identified length and the arr length difference is 3, then we actually take out the values from arr.
        Then they are going to be passed to the function, for so it adds those 3 signals with the neccesary NodeIDs
        for later trilateriation.
    '''
    arrCopy = arr.copy()
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
        return arr, identified, arrCopy
    elif(len(arr) - len(identified) == 3):
        print("\n")
        print(arr)
        print("\n")
        print(identified)
        for ident in identified:
            arr.remove(ident[0])

        return arr, identified, arrCopy
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
        return arr, identified, arrCopy



def deviation(arr, value):
    '''
     Takes an array of values and a value inside of that array. Calculates how much does
     that value deviate from the other values. The function is used when choosing which error
     probability to take into the identified signals.
    '''
    total = 0
    for i in range(0, len(arr)):
        total += pow(arr[i]-value,2)
    result = math.sqrt(total/(len(arr)-1))
    return result


def insideIdentified(arr, index):
    '''
        This one gets an array (this function only is used by the identified list) and an index.
        Then it checks whether that index is inside the identified or not. If it is return True, otherwise False.
        This function is crucial in determining, whether the highest 'error' value is already inside the identified list.
    '''
    for i in range(0, len(arr)):
        if(arr[i][1] == index):
            return True
    return False

def insideIdentified_2(arr, value):
    '''
        This works in similar manner, but it check for the nodeDistSum value. Check whether it is inside the identified or not.
        There was an issue with the index approach, so needed to use this in a specific case.
    '''
    for i in range(0, len(arr)):
        if(arr[i][0][1] == value):
            return True
    return False


def findIndex(arr, value):
    '''
        This is used to find the lowest and/or highest error inside the identified list. The function only
        is used once,when the first value is taken out. So, there is no need to worry about duplication.
        (so taking out the wrong value etc.)
    '''
    for i in range(0, len(arr)):
        if(arr[i] == value):
            return i


def testLowest(arr):
    '''
        This basically is the function that can replace the error_calculation function, if we want to
        choose the lowest error producing signals, just by looking at the error values. So, bascially gettin all the signals
        that are the lowest producing (the 3 lowest).
    '''
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

def optimizeSignal(arr, arr2, arr3, arr4):
    '''
        This funciton is going to change the identified error signals' signal and then a new signal is going to be passed,
        to the data gathering, which will be a more accurate, optimized signal of the signals. Then, multilateration is
        going to be applied. Eventually it should increase the accuracy, hopefully this gets it below 50metres.

        arr structure:
            All the signal data in the current batch
        arr2 structure:
            Data on whether a signal is dissonant, if so what is the difference for each of them etc.
            [[sigDiff*probability, signalThatIsDissonantTo, DissonantOrNotBool],...]
        arr3 structure:
            Identified signals, that are going to be changed for better multilateration.
    '''
    print("%%% Optimize Signal %%%")
    #input()
    print(arr)
    print(arr2)
    print(arr3)
    currReplace = arr3[len(arr3)-1][0]
    print("\n")
    print("This is currReplace:")
    print(currReplace)
    #input()
    for signalI in range(0,len(arr)):
        if(arr[signalI] == currReplace):
            errIndex = signalI
    print("##############")
    analyseResult = arr2[errIndex]
    #Now we will try to optimize the signal, so we get
    #the lowest possible dissonances.
    currSig = currReplace[0]
    maxi = -1
    for i in analyseResult:
        if(maxi < abs(i[0])):
            maxi = i[0]
    maxi = int(round(maxi,0))
    if(maxi > 5):
        maxi = 5
    print(maxi)
    possibleChange = [0]
    for number in range(1,maxi+1):
        possibleChange.append(number)
        possibleChange.append(-number)
    print(possibleChange)
    evaluateChanges = []
    for change in possibleChange:
        changeSig = currSig + change
        good = 0
        bad = 0
        index = 0
        for otherSig in analyseResult:
            #Try to compare the results, how it is changed, how many good / bad you have
            print(otherSig)
            if(otherSig[2] == False):
                if(currSig > otherSig[1]):
                    if(changeSig >= otherSig[1]):
                        good += 1
                    else:
                        bad += 1
                elif(currSig < otherSig[1]):
                    if(changeSig <= otherSig[1]):
                        good += 1
                    else:
                        bad += 1
                else:
                    dist1 = currReplace[1]
                    dist2 = findDistance(arr, currReplace, index)
                    dists = [dist1,dist2]
                    signs = [changeSig,otherSig[1]]
                    if(signalFit(dists, signs) == True):
                        good += 1
                    else:
                        bad += 1
            else:
                if(currSig > otherSig[1]):
                    if(changeSig > otherSig[1]):
                        bad += 1
                    else:
                        good += 1
                elif(currSig < otherSig[1]):
                    if(changeSig < otherSig[1]):
                        bad += 1
                    else:
                        good += 1
                else:
                    dist1 = currReplace[1]
                    dist2 = findDistance(arr, currReplace, index)
                    dists = [dist1,dist2]
                    signs = [changeSig,otherSig[1]]
                    if(signalFit(dists, signs) == True):
                        good += 1
                    else:
                        bad += 1
            index += 1
        evaluateChanges.append([good-bad, changeSig])
        #print(good, bad, changeSig)
    print("Change Signals!")
    print(evaluateChanges)
    #input()
    '''
        Now, choose the one that has the best 'good' - then replace the signal with that.
        So, now we have an optimised signal. We will be able to use this for multilateration.
    '''
    maxi2 = -1000
    for ev in range(0,len(evaluateChanges)):
        if(maxi2 < evaluateChanges[ev][0]):
            maxi2 = evaluateChanges[ev][0]
    print("Maxi!!!")
    print(maxi2)
    #input()
    reducedChanges = []
    for ev2 in range(0,len(evaluateChanges)):
        if(evaluateChanges[ev2][0] == maxi2):
            reducedChanges.append(evaluateChanges[ev2])
    print("Reduced changes!!")
    print(reducedChanges)
    #input()
    if(len(reducedChanges) == 1):
        optimisedSig = reducedChanges[0][1]
        currReplace[0] = optimisedSig
        arr4.append(currReplace)
    else:
        print("#########REDUCED CHANGES#########")
        print(reducedChanges)
        optimisedSig = findOptimisedSignal(reducedChanges, currSig)
        currReplace[0] = optimisedSig
        arr4.append(currReplace)
        print("#########REDUCED CHANGES#########")
        #input()
    print(analyseResult)
    print(currReplace)
    #input()

    return arr4

def findOptimisedSignal(arr, ogSign):
    lower = 0
    higher = 0
    lowerData = []
    higherData = []
    for i in range(0,len(arr)):
        if(ogSign > arr[i][1]):
            lower += 1
            lowerData.append(arr[i])
        if(ogSign < arr[i][1]):
            higher += 1
            higherData.append(arr[i])
    if(len(higherData) == 0):
        total = 0
        for j in lowerData:
            total += j[1]
        mean = round(total/len(lowerData),2)
    elif(len(lowerData) == 0):
        total = 0
        for j in higherData:
            total += j[1]
        mean = round(total/len(higherData),2)
    else:
        if(higher > lower):
            total = 0
            for j in lowerData:
                total += j[1]
            mean = round(total/len(lowerData),2)
        else:
            total = 0
            for j in higherData:
                total += j[1]
            mean = round(total/len(higherData),2)
    if(mean < -101):
        mean = -101
    if(mean > -73):
        mean = -73
    return mean



def signalFit(arr, arr2):
    '''
        Checks whether the signals follow the rule or not.
        sig1 > sig2
        dist1 < dist 2
    '''
    sig1 = arr2[0]
    sig2 = arr2[1]
    dist1 = arr[0]
    dist2 = arr[1]
    if(sig1 > sig2):
        if(dist1 < dist2):
            return True
        else:
            return False
    else:
        if(dist1 > dist2):
            return True
        else:
            return False

def findDistance(arr, itemSkip ,count):
    '''
        Finds the corresponding distances when the optimisation encounters signals that are the same.
        So, we can apply a different optimisation.
    '''
    count2 = 0
    distance = None
    for i in range(0,len(arr)):
        if(arr[i] == itemSkip): continue
        if(count2 == count):
            distance = arr[i][1]
        count2 += 1
    return distance



def loadANNData_2(isTrilat=False, month="June"):
    '''
        Idea here would be gather the signal and then the calculated distance, then get what the error is
        then train the signal and the error on the model. Model would try to predict the error that is
        contributed to the signal~distance.

        Currently, it just gathers 3-lowest-error producing errors that were identifed by the errr_calculation and
        remove_highest_error functions.
    '''
    #Loads the grid, so we can exclude going over points that are already outside of the grid
    if(month == "June"):
        grid, sections, nodes = utils.loadSections_Old()
    else:
        grid, sections, nodes = utils.loadSections()

    #Loads the data that was gathered inside the distanceNNData.json
    #
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

    dronePos = None
    fileX = []
    fileY = []
    for index, i in enumerate(data["X"]):
        actual = data['y'][index]
        predicted = utils.calculateDist_2(i[0])
        err = actual - predicted
        err = np.float64(err)
        '''
            Intitialises the importanD, values that are going to be used in the function or other places
            i[0] - signalStr
            i[1] - nodeSum- for the current node that received i[0]
            i[2] - all signals
            actual - actual distance
            predicted - predicted distance by equation
            i[3] - id; so we can identify different batches etc.
            i[4] - nodeID
        '''
        importantD = [i[0], i[1], i[2], actual, predicted, err, i[3], i[4]]

        importantD[1] = sum(importantD[1])/len(importantD[1])

        #Gather dronePosition for current batch
        if(dronePos != data["Z"][index]):
            keptDronePos = dronePos
            dronePos = data["Z"][index]
        #Start gathering data into importantDs, distanceSum, actualDist, allSignals if the batchN(i[3]) is the same still
        if(i[3] == batchN):
            total = 0
            for number in i[1]:
                total += number
            distanceSum.append([total/len(i[1]), a])
            actualDist.append(actual)
            allSignals.append(i[0])
            importantDs.append(importantD)
            a += 1

        #If the batchN is not the same, that means we need to start a new batch and we can
        #calculate the error probabilits for the specific batch, because improtantDs has all the
        #signals in the batch.
        else:
            #First we need to add the distances from other nodes sums to the each of the importantDs,
            #which would then complete our data that the error_calculation will work on
            for e in range(0,len(importantDs)):
                addTotals = []
                for j in range(0,len(distanceSum)):
                    if(e == distanceSum[j][1]): continue
                    addTotals.append(distanceSum[j][0])
                importantDs[e].insert(3, addTotals)
            '''
                This is where we calculate the value between that will be between -1 to 1, that will tell us
                how big the error could be
                The next value then will tell us how likely the error is
                Those two values are going to be used by the machine along with the signal, to predict the error
                Then the error is going to be added to the calcualted distance, giving a better result for the RSSI ~ Distance
            '''

            #Get coordinates of the drone, if inside the grid then do the calcualtion otherwise skip for now
            utmCord = utm.from_latlon(keptDronePos[0],keptDronePos[1])
            inside = utils.pointToSection(utmCord[0], utmCord[1], sections)
            '''
                Idea for error calculation // signal re-calculation. We will re-calculate all signals except for 3.
                So, we basically will make use of the trilateration - rewrite all signals except for those that have
                the highest accuracy according to the function.
                Then we can do two things. Either do multilateration with the all data (including the 3 signals
                that were not change) or do multilateration with the signals excluding the 3 signals taht were not
                changed.
                This comes down to two ideas. If we rewrite a finite amount of signals, we can increase the accuracy,
                by what extent is yet to be discovered.
            '''

            if(inside != -1):
                print("\n")
                print("This is the data before selection: ")
                print(importantDs)
                print("\n")
                identified = []
                optIdentified = []
                #While loop that will run the error_calculations again and again, till
                #3 singals are identified by the removeHighestError and the error_calculation funcitons
                canCont = False
                while canCont == False:
                    errorCalcs = []
                    errorVals = []
                    newImportants = []
                    #Calculates all the errors for all the signals in the batch, then adds them to a list
                    #errorCalcs -> errorProbabilites for the signals
                    #newImprotants -> the data for the the specific errorProbabiltiy, this is identical
                    #to the importantDs, but it is easier to introduce another list, so we can replace this later on, without
                    #damaging the importantDs.
                    for important in importantDs:
                        newImportant = error_calculation(important, identified)
                        errorCalcs.append(newImportant[1])
                        newImportants.append(newImportant[0])
                        errorVals.append(newImportant[2])
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%")
                    print(errorVals)
                    print(errorCalcs)
                    print("%%%%%%%%%%%%%%%%%%%%%%%%%")
                    #input()
                    print("\n")
                    newImportants, identified, newImportantsCopy = removeHighestError(newImportants, errorCalcs, identified)
                    optIdentified = optimizeSignal(newImportantsCopy, errorVals, identified, optIdentified)
                    #Remove the commenting to from the testLowest, to test what would be the value,
                    #if we chose all the best values for trilateration, turns out the function is working, just
                    #trilateriation is not that powerful haha.
                    #newImportants = testLowest(newImportants)
                    importantDs = newImportants

                    #If it has the length of 3, so 3 signals are identified then we can stop.
                    if(len(newImportants) == 3):
                        canCont = True
                print("So this is going to be used for Trilateration: ")

                '''
                    Gathering the data for a trilateration json ahhahaha
                '''
                if(isTrilat == False):
                    for optSig in optIdentified:
                        importantDs.append(optSig)

                fileData = {}
                for fileE in importantDs:
                    fileData[fileE[len(fileE)-1]] = fileE[0]
                #print(fileData)
                outerFileData = {}
                #Not sure if that would affect the results, but I just did this why not
                #Was lazy to rewrite the code at this point
                outerFileData["time"] = random.randint(0,1000000)
                outerFileData["tag"] = random.randint(0,1000000)
                outerFileData["data"] = fileData

                fileX.append(outerFileData)
                fileY.append(keptDronePos)
                print(importantDs)
                print("\n")
                #input()
            #Reset the batch data, start with new batch and start adding the values.
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

    input()
    print("\n")
    print("\n")

    assert(len(fileX) == len(fileY))
    finalData = {}
    finalData['X']=fileX
    finalData['y']=fileY
    if(isTrilat == False):
        with open("../Data/"+month+"/multilatFunctionData.json","w+") as f:
                json.dump(finalData, f)
    else:
        #Save the file in a json, that will be used for modelling / training
        with open("../Data/"+month+"/trilatData.json","w+") as f:
            json.dump(finalData, f)

def LoadMLPData(month="June"):
    '''
        This function is just plainly bad. There was a paper, which proposed that if they have the signal values then an MLPRegressor can
        predict the position of the drone/signal etc. But it's just not true lol. I'm gonna keep it here, just for negative results.
    '''
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


if __name__=="__main__":
    #loadANNData_2()
    proximityDataManipulation(month="October")

'''
    Recursive signal rewriting - > took it out as I don't think it is very professional. This was deleted from multilat.py
'''
'''
    Combined all the prediciton functions, so we don't need to use separate for each of them.
'''
def predictions(rssiThreshold=-105.16,keepNodeIds=False, isTriLat = False, optMultilat=False, month="June"):
    """Combined march and june predictions."""
    if(month =="June"):
        # Load in the June JSON
        data = utils.loadData(month="June",isTrilat=isTrilat, optMultilat=optMultilat)
        nodes = utils.loadNodes(rewriteUTM=True)
    if(month =="March"):
        #Load March - new JSON
        data = utils.loadData(month="March",,isTrilat=isTrilat, optMultilat=optMultilat)
        nodes = utils.loadNodes(rewriteUTM=False)
    if(month == "October"):
        data = utils.loadData(month="October",isTrilat=isTrilat, optMultilat=optMultilat)
        nodes = utils.loadNodes_46()
    # assign the relevant variables
    X = data["X"]
    y = data["y"]
    # need to rewrite the node location using the utm module
    # because we need the utm module for the Drone's gps

    # for stats
    numberOfTests = 0
    averageError = 0
    results = {}
    currentMax = 0
    for idx in range(len(X)):
        # the two important lists, the distances to the nodes
        # and the node locations themselves (with updated utm)
        distToNodes = []
        nodeLocs = []
        nodesTargeted = []

        actualDist = []
        signals = []
        theids = []


        for dataEntry in X[idx]["data"].keys():
            nodeId = dataEntry
            if X[idx]["data"][nodeId] <=rssiThreshold or nodeId not in nodes.keys(): continue
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==3 and X[idx]["data"][nodeId]>currentMax:
                continue
            if(month == "October"):
                distToNodes.append(utils.calculateDist_2(X[idx]["data"][nodeId]))
            else:
                distToNodes.append(utils.calculateDist_2(X[idx]["data"][nodeId]))

            if dataEntry=="3288000000": nodeId="3288e6"
            nodeLocs.append([nodes[nodeId]["NodeUTMx"],nodes[nodeId]["NodeUTMy"]])

            '''
                TESTING AREA FOR NEW FILTER
                Works to an extent.
            '''
            '''
            nodeLocation = np.array([nodes[nodeId]["NodeUTMx"],nodes[nodeId]["NodeUTMy"]])
            if(month=="March"):
                utmGT = [y[idx][0], y[idx][1]]
            else:
                utmGT = utm.from_latlon(y[idx][0], y[idx][1])
            gt = np.array([np.float64(utmGT[0]),np.float64(utmGT[1])])
            aDist = np.linalg.norm(nodeLocation-gt)
            actualDist.append(aDist)
            signals.append(X[idx]["data"][nodeId])
            theids.append(nodeId)

            '''
            '''
                END OF TESTING AREA
            '''

            if keepNodeIds:
                nodesTargeted.append(nodeId)
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==4:
                # find the maximum value and remove the index from all lists
                ind = np.argmax(distToNodes)
                del distToNodes[ind]
                del nodeLocs[ind]
                if keepNodeIds:
                    del nodesTargeted[ind]
            currentMax = max(distToNodes)

        # need at least 3 values
        if len(distToNodes)<3: continue

        numberOfEntries = 0

        '''
            Data manipulation for other tpye of multilateration; seems worse.
        '''
        '''
        nls_points = []
        for n in range(0,len(distToNodes)):
            latlon = utm.to_latlon(nodeLocs[n][0],nodeLocs[n][1],18,'N')
            nls_points.append(((latlon[0],latlon[1]),distToNodes[n]/1000))
        res_2 = np.array(multilateration(nls_points))
        res_2 = utm.from_latlon(res_2[0],res_2[1])
        '''


        # Calculate the multilateration
        res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))

        if(month=="March"):
            utmGT = [y[idx][0], y[idx][1]]
        else:
            utmGT = utm.from_latlon(y[idx][0], y[idx][1])

        gt = np.array([np.float64(utmGT[0]),np.float64(utmGT[1])])
        dist = np.linalg.norm(res-gt)
        '''
            TESTING AREA START
            Works to an extent
        '''
        '''
        #print("Next")
        #print("Node loc: ", nodeLocs)
        #print("Calculated Dist: ",distToNodes)
        #print("Signals: ",signals)
        #print("Actual Dist: ",actualDist)
        #print("NodeIDs: ",theids)
        #print("Actual position: ", gt)
        #print("Predicted position: ", res)
        #print("Error: ", dist)
        #print("\n")



        errorArea = False
        lineValues = []
        #print("\n")
        for k in range(0,len(nodeLocs)):
            summing = 0
            for j in range(0,len(nodeLocs)):
                if(nodeLocs[k] == nodeLocs[j]): continue
                line = np.linalg.norm(np.array(nodeLocs[k])-np.array(nodeLocs[j]))
                summing += line
                #print(theids[k], "to ", theids[j],": ", line)
            lineValues.append([summing,signals[k],distToNodes[k]])
        #print(lineValues)
        check = None
        newLinesValues = lineValues.copy()
        #print("y")
        update = utils.rewrite(lineValues, newLinesValues)
        #print("This is DIST TO NODES: ", distToNodes)
        if(check != update):
            errorArea = True
            for item in range(0,len(distToNodes)):
                if(distToNodes[item] != update[item][2]):
                    distToNodes[item] = update[item][2]
                    signals[item] = update[item][1]
        #print("THIS IS DIST TO NODES:", distToNodes)
        #input()
        #print("\n")
        #print("THIS IS THE END: ", update)

        if(errorArea== True):
            #print("\n")
            #print("There was a change!")
            #print("\n")
            #print("Calculated Dist: ",distToNodes)
            #print("Node loc: ", nodeLocs)
            res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))
            dist = np.linalg.norm(res-gt)
            #print("Node loc: ", nodeLocs)

            #print("Signals: ",signals)
            #print("Actual Dist: ",actualDist)
            #print("NodeIDs: ",theids)
             #print("Actual position: ", gt)
            #print("Predicted position: ", res)
            #print("Error: ", dist)
        #input()
        '''
        '''
            TESTING AREA END
        '''


        testId = X[idx]["time"]+X[idx]["tag"]
        '''
            Delete the commenting to test it with the other type of multilateration algorithm; it seems to be worse
        '''
        #res= [res_2[0],res_2[1]]

        results[testId] = {"gt":gt,
                       "res":res,
                       "error":dist,
                       "nodeDists":distToNodes,
                       "nodeLocs":nodeLocs}
        if keepNodeIds:
            results[testId]["nodeIds"] = nodesTargeted
        averageError+=dist
        numberOfTests+=1
    print("Using a threshold of {} for the RSSI, with multilateration: {}, using trilateration: {}, the average error was {} m".format(rssiThreshold, not isTriLat, isTriLat, (averageError/numberOfTests)))
    #print(numberOfTests)
    return results

def junePredictions(rssiThreshold=-105.16,keepNodeIds=False, isTriLat = False):
    """ This should be combined with the March function in reality, which requires rewriting the util load data function
        but the premise is the exact same and works the same way."""
    # Load in the June JSON
    data = utils.loadData(month="June")
    # assign the relevant variables
    X = data["X"]
    y = data["y"]
    # need to rewrite the node location using the utm module
    # because we need the utm module for the Drone's gps
    nodes = utils.loadNodes(rewriteUTM=True)
    # for stats
    numberOfTests = 0
    averageError = 0
    results = {}
    currentMax = 0
    for idx in range(len(X)):
        # the two important lists, the distances to the nodes
        # and the node locations themselves (with updated utm)
        distToNodes = []
        nodeLocs = []
        nodesTargeted = []


        for dataEntry in X[idx]["data"].keys():
            nodeId = dataEntry
            if X[idx]["data"][nodeId] <=rssiThreshold or nodeId not in nodes.keys(): continue
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==3 and X[idx]["data"][nodeId]>currentMax:
                continue
            distToNodes.append(utils.calculateDist_2(X[idx]["data"][nodeId]))
            if dataEntry=="3288000000": nodeId="3288e6"
            nodeLocs.append([nodes[nodeId]["NodeUTMx"],nodes[nodeId]["NodeUTMy"]])
            if keepNodeIds:
                nodesTargeted.append(nodeId)
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==4:
                # find the maximum value and remove the index from all lists
                ind = np.argmax(distToNodes)
                del distToNodes[ind]
                del nodeLocs[ind]
                if keepNodeIds:
                    del nodesTargeted[ind]
            currentMax = max(distToNodes)

        # need at least 3 values
        if len(distToNodes)<3: continue

        numberOfEntries = 0
        # Calculate the multilateration
        res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))
        utmGT = utm.from_latlon(y[idx][0], y[idx][1])

        gt = np.array([np.float64(utmGT[0]),np.float64(utmGT[1])])
        dist = np.linalg.norm(res-gt)
        testId = X[idx]["time"]+X[idx]["tag"]
        results[testId] = {"gt":gt,
                       "res":res,
                       "error":dist,
                       "nodeDists":distToNodes,
                       "nodeLocs":nodeLocs}
        if keepNodeIds:
            results[testId]["nodeIds"] = nodesTargeted
        averageError+=dist
        numberOfTests+=1
    print("Using a threshold of {} for the RSSI, with multilateration: {}, using trilateration: {}, the average error was {} m".format(rssiThreshold, not isTriLat, isTriLat, (averageError/numberOfTests)))
    #print(numberOfTests)
    return results


def marchPredictions(rssiThreshold=-105.16, pruned=False, isTriLat = False):
    data = utils.loadData(pruned=pruned)
    nodes = utils.loadNodes()
    numberOfTests = 0
    averageError = 0
    results = {}
    """
        Each test is within a 2 minute time frame iirc, so
        I will take the average distance for each node in that time frame and then create
        the requisite input for the gpsSolve and then compare to the actual distance
    """
    for id in data.keys():
        freq = {}
        distToNodes = []
        nodeLocs = []
        numberOfEntries = 0
        for dataEntry in data[id]["Data"]:
            if dataEntry["NodeId"]=="3288000000": dataEntry["NodeId"]="3288e6"
            if dataEntry["TagRSSI"] <=rssiThreshold or dataEntry["NodeId"] not in nodes.keys(): continue
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==3 and X[idx]["data"][nodeId]>min(distToNodes):
                continue
            if dataEntry["NodeId"] not in freq:
                freq[dataEntry["NodeId"]] = [0,0]
            freq[dataEntry["NodeId"]][0]+=utils.calculateDist_2(dataEntry["TagRSSI"])
            freq[dataEntry["NodeId"]][1]+=1
        """
            Average the distances and populate two lists, one with distances to nodes
            and the other the location of the nodes
        """
        if freq == {} or len(freq.keys())<3: continue
        for node in freq.keys():
            freq[node][0]/=freq[node][1]
            distToNodes.append(freq[node][0])
            nodeLocs.append([nodes[node]["NodeUTMx"],nodes[node]["NodeUTMy"]])

        res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))
        gt = np.array([np.float64(data[id]["GroundTruth"]["TestUTMx"]),np.float64(data[id]["GroundTruth"]["TestUTMy"])])
        dist = np.linalg.norm(res-gt)

        results[id] = {"gt":gt,
                       "res":res,
                       "error":dist,
                       "nodeDists":distToNodes,
                       "nodeLocs":nodeLocs}

        averageError+=dist
        numberOfTests+=1

    print("Using a threshold of {} for the RSSI, the average error was {} m".format(rssiThreshold,(averageError/numberOfTests)))

    return results



'''
    This was deleted from utils.py
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

#This was dropped from Proxmimity.py
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

            batchSignals.append(newSignals)
            targetPos.append(dronePos)
    assert(len(targetPos) == len(batchSignals))

    return batchSignals, targetPos

#Dropped from the model.py
def MLPModel():
    '''
        Tried out the method proposed in a paper. But this is just plain bad. Can't even fith the data.
    '''
    X, y = LoadMLPData()
    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=0.8, random_state=101)
    clf = MLPRegressor(hidden_layer_sizes=(8,10,8,8,6), max_iter=1000,activation='relu',solver='adam', random_state=1)
    clf2= MLPRegressor(hidden_layer_sizes=(8,8,6), max_iter=300,activation='relu',solver='adam', random_state=1)
    #print(y_train[:,0])
    #print(y_train[:,1])
    #print(y_train)
    clf.fit(X_train, y_train)
    #yPred = clf.predict(X_remaining)

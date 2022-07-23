"""
    This file will contain multilateration methods, as we continue to work on reducing error.
    Currently, it has a reworked implementation from https://github.com/glucee/Multilateration/blob/master/Python/example.py
"""
import numpy as np
import scipy.optimize as opt
import utils, argparse
import utm
from scipy.optimize import minimize
import numpy as np

def gps_solve(distsToNodes, nodeLocations):
    """
        Implementation to solve the equations for the circle intersections comes from:
        https://github.com/glucee/Multilateration/blob/master/Python/example.py
    """
    def error(x, c, r):
        return sum([(np.linalg.norm(x - c[i]) - r[i]) ** 2 for i in range(len(c))])

    l = len(nodeLocations)
    S = sum(distsToNodes)
    # compute weight vector for initial guess
    W = [((l - 1) * S) / (S - w) for w in distsToNodes]
    # get initial guess of point location
    x0 = sum([W[i] * nodeLocations[i] for i in range(l)])
    # optimize distance from signal origin to border of spheres
    return minimize(error, x0, args=(nodeLocations, distsToNodes), method='Nelder-Mead').x


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
        for dataEntry in data[id]["Data"]:
            if dataEntry["NodeId"]=="3288000000": dataEntry["NodeId"]="3288e6"
            if dataEntry["TagRSSI"] <=rssiThreshold or dataEntry["NodeId"] not in nodes.keys(): continue
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTriLat and len(distToNodes)==3 and X[idx]["data"][nodeId]>min(distToNodes):
                continue
            if dataEntry["NodeId"] not in freq:
                freq[dataEntry["NodeId"]] = [0,0]
            freq[dataEntry["NodeId"]][0]+=utils.calculateDist(dataEntry["TagRSSI"])
            freq[dataEntry["NodeId"]][1]+=1
        """
            Average the distances and populate two lists, one with distances to nodes
            and the other the location of the nodes
        """
        if freq == {} or len(freq.keys())<3: continue
        distToNodes = []
        nodeLocs = []
        numberOfEntries = 0
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
            distToNodes.append(utils.calculateDist(X[idx]["data"][nodeId]))
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

def main(args=None):


    rssiThreshold = -105.16
    trilat = False
    if args.rssi!=None:
        try:
            rssiThreshold = int(args.rssi)
        except:
            print("You must enter a valid number for the rssi filter")
    if args.trilat!=None:
        trilat = True

    if args.month==None:
        print("Please enter a month. Currently, these are the supported types: 'June' and 'March'")
    elif args.month.lower()=="june":
        junePredictions(rssiThreshold, isTriLat=trilat)
    elif args.month.lower()=="march":
        marchPredictions(rssiThreshold, isTriLat=trilat)
    else:
        print("Sorry, please enter a valid month. Currently, these are the supported types: 'June' and 'March'")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Multilat variables')
    parser.add_argument('--month', dest='month', type=str, help='Month of the data')
    parser.add_argument('--rssi', dest='rssi', type=int, help='rssi filter for the data')
    parser.add_argument('--trilat', dest='trilat', type=bool, help='Whether we\'re using trilateration rather than multi')
    parser.add_argument('--pruned', dest='pruned', type=bool, help='Whether or not to use the pruned data, only applicable for March')

    args = parser.parse_args()

    main(args)

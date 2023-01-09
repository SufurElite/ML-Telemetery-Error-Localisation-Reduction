"""
    This file will contain multilateration methods, as we continue to work on reducing error.

    Implementations derived from:
    https://github.com/glucee/Multilateration/blob/master/Python/example.py
    https://github.com/koru1130/multilateration
"""

import numpy as np
import scipy.optimize as opt
import utils, argparse
import utm
from scipy.optimize import minimize, least_squares
import numpy as np
from geopy import distance

def gps_solve(distsToNodes, nodeLocations):
    """
        Solves for the location given a list of distances to respective locations
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


def residuals_fn(points, dist):
    """ Function that the other type of multlateration uses. """
    def fn(location):
        return np.array( [ (dist(p, location).km - r)*(r*r) for (p, r) in points ] )
    return fn


def multilateration(points, dist_type ='geodesic'):
    """ Applies least squares using residuals to solve for multialteration """
    if dist_type == 'geodesic' :
        dist = distance.distance
    elif dist_type == 'great_circle' :
        dist = distance.great_circle
    else:
        raise Exception("dist_type error")

    ps = [x[0] for x in points]
    x0 = np.mean(np.array(ps),axis=0)
    return least_squares(residuals_fn(points, dist), x0).x


def predictions(rssiThreshold=-105.16,keepNodeIds=False, isTrilat = False, optMultilat=False, month="June", otherMultilat=False):
    """ Predictions for March and June """
    if(month =="June"):
        # Load in the June JSON
        data = utils.loadData(month="June",isTrilat=isTrilat, optMultilat=optMultilat)
        nodes = utils.loadNodes(rewriteUTM=True)
    if(month =="March"):
        #Load March - new JSON
        data = utils.loadData(month="March",isTrilat=isTrilat, optMultilat=optMultilat)
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
            if isTrilat and len(distToNodes)==3 and X[idx]["data"][nodeId]>currentMax:
                continue
            if(month == "October"):
                distToNodes.append(utils.calculateDist_2(X[idx]["data"][nodeId]))
            else:
                distToNodes.append(utils.calculateDist_2(X[idx]["data"][nodeId]))

            if dataEntry=="3288000000": nodeId="3288e6"
            nodeLocs.append([nodes[nodeId]["NodeUTMx"],nodes[nodeId]["NodeUTMy"]])

            if keepNodeIds:
                nodesTargeted.append(nodeId)
            # If we're doing trilateration rather than multi, keep only the 3 lowest values
            if isTrilat and len(distToNodes)==4:
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
        if not otherMultilat:
            res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))
        else:
            points = [(nodeLocs[i],distToNodes[i]) for i in range(len(distToNodes))]
            res = multilateration(points)

        if(month=="March"):
            utmGT = [y[idx][0], y[idx][1]]
        else:
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
    print("Using a threshold of {} for the RSSI, with multilateration: {}, using trilateration: {}, the average error was {} m".format(rssiThreshold, not isTrilat, isTrilat, (averageError/numberOfTests)))
    return results


def main(args=None):

    rssiThreshold = -105.16
    trilat = False
    optMultilat = False
    otherMultilat = False

    if args.rssi!=None:
        try:
            rssiThreshold = int(args.rssi)
        except:
            print("You must enter a valid number for the rssi filter")
    if args.trilat:
        trilat = True
    if args.opt_multilat:
        optMultilat = True
    if args.other_multilat:
        otherMultilat = True
        
    if args.month==None:
        print("Please enter a month. Currently, these are the supported types: 'June' and 'March'")
    elif args.month.lower()=="june" or args.month.lower()=="march" or args.month.lower()=="october":
        predictions(rssiThreshold,isTrilat=trilat, optMultilat=optMultilat, otherMultilat=otherMultilat)
    else:
        print("Sorry, please enter a valid month. Currently, these are the supported types: 'June' and 'March'")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Multilat variables')
    parser.add_argument('--month', dest='month', type=str, help='Month of the data')
    parser.add_argument('--rssi', dest='rssi', type=int, help='rssi filter for the data')
    parser.add_argument('--trilat', dest='trilat', type=bool, action=argparse.BooleanOptionalAction, help='Whether we\'re using trilateration rather than multi, include this flag')
    parser.add_argument('--pruned', dest='pruned', type=bool, action=argparse.BooleanOptionalAction, help='Whether or not to use the pruned data, only applicable for March')
    parser.add_argument('--optMultilat', dest='opt_multilat', type=bool, action=argparse.BooleanOptionalAction, help='If in loading the data you want to use not to use the optimal mulitlateration, include this flag')
    parser.add_argument('--otherMultilat', dest='other_multilat', type=bool, action=argparse.BooleanOptionalAction, help='If you want to use the other multilateration in prediction, include this flag')

    args = parser.parse_args()

    main(args)
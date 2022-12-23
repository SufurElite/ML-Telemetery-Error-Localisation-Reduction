"""
    This file will contain multilateration methods, as we continue to work on reducing error.
    Currently, it has a reworked implementation from https://github.com/glucee/Multilateration/blob/master/Python/example.py
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
'''
    Another implementation to solve the multilateration.
    https://github.com/koru1130/multilateration
'''

'''
    Function that the other type of multlateration uses.
'''
def residuals_fn(points, dist):
    def fn(location):
        return np.array( [ (dist(p, location).km - r)*(r*r) for (p, r) in points ] )
    return fn

'''
    Other tpye of multilateration, can remove commenting to use this.
'''
def multilateration(points, dist_type ='geodesic'):
    if dist_type == 'geodesic' :
        dist = distance.distance
    elif dist_type == 'great_circle' :
        dist = distance.great_circle
    else:
        raise Exception("dist_type error")

    ps = [x[0] for x in points]
    x0 = np.mean(np.array(ps),axis=0)
    return least_squares(residuals_fn(points, dist), x0).x

'''
    Combined all the prediciton functions, so we don't need to use separate for each of them.
'''
def predictions(rssiThreshold=-105.16,keepNodeIds=False, isTrilat = False, optMultilat=False, month="June", otherMultilat=False):
    """Combined march and june predictions."""
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
        res = np.array(gps_solve(distToNodes, list(np.array(nodeLocs))))

        if(month=="March"):
            utmGT = [y[idx][0], y[idx][1]]
        else:
            utmGT = utm.from_latlon(y[idx][0], y[idx][1])

        gt = np.array([np.float64(utmGT[0]),np.float64(utmGT[1])])
        dist = np.linalg.norm(res-gt)

        testId = X[idx]["time"]+X[idx]["tag"]


        '''
            Other type of multialteration - seems worse than the previous one.
        '''
        if(otherMultilat==True):
            res= [res_2[0],res_2[1]]

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
    elif args.month.lower()=="june" or args.month.lower()=="march" or args.month.lower()=="october":
        predictions(rssiThreshold,isTrilat=isTrilat, optMultilat=optMultilat, otherMultilat=otherMultilat)
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

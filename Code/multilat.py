"""
    This file will contain different multilateration methods, as we continue to work on reducing error.
    Currently, it has a reworked implementation from https://github.com/glucee/Multilateration/blob/master/Python/example.py
"""
import numpy as np
import scipy.optimize as opt
import utils, argparse
from utils import loadData, loadNodes, calculateDist

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


def marchPredictions(rssiThreshold=-105.16, pruned=False):
    data = loadData(pruned=pruned)
    nodes = loadNodes()
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
            if dataEntry["TagRSSI"] <=rssiThreshold or dataEntry["NodeId"] not in nodes.keys(): continue
            if dataEntry["NodeId"] not in freq:
                freq[dataEntry["NodeId"]] = [0,0]
            freq[dataEntry["NodeId"]][0]+=calculateDist(dataEntry["TagRSSI"])
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



def main(args=None):
    marchPredictions(-86, True)
    """
    stations = list(np.array([[1,1], [0,1], [1,0], [0,0]]))
    distances_to_station = [0.1, 0.5, 0.5, 1.3]
    print(gps_solve(distances_to_station, stations))
    if args.type=="stack":
        stack()
    else:
        print("Sorry, please list a valid type. Currently, these are the supported types: 'stack'")
    """


if __name__=="__main__":
    #parser = argparse.ArgumentParser(description='Personal information')
    #parser.add_argument('--type', dest='type', type=str, help='Type of multilateration approach')

    #args = parser.parse_args()

    main()
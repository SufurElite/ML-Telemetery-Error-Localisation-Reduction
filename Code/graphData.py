from kmlInterface import HabitatMap, Habitat
from sklearn.model_selection import train_test_split
import numpy as np
import utils, utm

class dataLoader():
    def __init__(self, covariate:bool = False, isFlatten:bool = False, loadConcatData:bool = False, month:str = "June"):
        self.month = month
        self.covariate = covariate
        if self.covariate:
            self.covariateModel = utils.loadCovariateModel()
        self.habitatMap = HabitatMap()
        self.baseAdjacency, self.nodesToHabitat = self.createBaseAdjacencyMatrix()
        self.isFlatten = isFlatten
        if loadConcatData and self.covariate:
            self.data = self.getConcatData()
        else:
            self.data = utils.loadRSSModelData(rowToAdjacencyMatrix=self.rowToAdjacencyMatrix)
        

    def createBaseAdjacencyMatrix(self, justNodes=False):
        nodes = utils.loadNodes(rewriteUTM=True)
        nodeKeys = list(nodes.keys())
        nodesToHabitat = []
        if justNodes:
            A = np.zeros(shape=(len(nodes),len(nodes)))
        else:
            A = np.zeros(shape=(len(nodes)+1,len(nodes)+1))
        for i in range(len(nodeKeys)):
            nodesToHabitat.append(self.habitatMap.whichHabitat(nodes[nodeKeys[i]]["NodeUTMx"],nodes[nodeKeys[i]]["NodeUTMy"])[0])
            for j in range(i+1, len(nodeKeys)):
                A[i][j]=utils.distBetweenNodes(nodeKeys[i], nodeKeys[j], nodes)
                A[j][i]= A[i][j]
        
        return A, nodesToHabitat

    def rowToAdjacencyMatrix(self, row):
        """ Turns a row with distances to each node the adjacency matrix will be of the shape n x n, where"""
        A = np.copy(self.baseAdjacency)
        for i in range(len(row)):
            A[len(A)-1][i] = row[i]
            A[i][len(A)-1] = row[i]
        if self.isFlatten:
            A = A.flatten()
        return A
    
    def getConcatData(self):
        
        # load in the data
        data = utils.loadData("June")
        X_vals = data["X"]
        y_vals = data["y"]
        
        assert(len(X_vals)==len(y_vals))
        print("Initial amount of data {}".format(len(X_vals)))
        # load in the nodes
        nodes = utils.loadNodes(rewriteUTM=True)
        nodeKeys = list(nodes.keys())

        # Generate the node locations in a list
        nodeLocs = []
        for key in nodeKeys:
            nodeLocs.append([nodes[key]["NodeUTMx"],nodes[key]["NodeUTMy"]])

        X = []
        y = []
        
        adjMat = self.createBaseAdjacencyMatrix(justNodes=True)[0].flatten()
        for i in range(len(X_vals)):
            tmp_x = [0 for i in range(len(nodeKeys)*2)]
            
            # signal_X is used to predict the habitat from signals
            signal_X = [0 for i in range(len(nodeKeys))]
            predictedTagHab = None
            
            tagGt = utm.from_latlon(y_vals[i][0], y_vals[i][1])
            tagLoc = np.array([np.float64(tagGt[0]),np.float64(tagGt[1])])
            tagNodeDists = [0 for i in range(len(nodeKeys))]
            for j in range(len(nodeKeys)):
                nodeKey = nodeKeys[j]
                tmp_x[j*2+1] = self.nodesToHabitat[j]
                nodeLoc = np.array([np.float64(nodes[nodeKey]["NodeUTMx"]), np.float64(nodes[nodeKey]["NodeUTMy"])])
                tagNodeDists[j] = np.linalg.norm(nodeLoc-tagLoc)
                if nodeKey not in X_vals[i]["data"] or X_vals[i]["data"][nodeKey]<-102: continue
                # start for each value
                
                signal_X[j] = X_vals[i]["data"][nodeKey]
                tmp_x[j*2] = utils.calculateDist_2(X_vals[i]["data"][nodeKey])
            
            predictedTagHab = self.covariateModel.predict(np.array([signal_X]))[0]
            for j in range(len(nodeKeys)):
                nodeKey = nodeKeys[j]
                # this is for the y-values 
                if nodeKey not in X_vals[i]["data"] or X_vals[i]["data"][nodeKey]<-102: continue
                fullX = np.append(adjMat,tmp_x)
                fullX = np.append(fullX,predictedTagHab)
                fullX = np.append(fullX,j)
                tmp_y = tagNodeDists[j]
                X.append(fullX)
                y.append(tmp_y)
        X = np.array(X)
        y = np.array(y)
        
        print(len(X), len(y))
        assert(len(X)==len(y))
        return X,y

    def getTotalData(self):
        return self.data[0], self.data[1]
        
    def getSplitData(self, trainSize:float = .8, validationSplit: float = .5, includeValid:bool = False):
        # Split the data into train, test, and validation sets
        X_train, X_remaining, y_train, y_remaining = train_test_split(self.data[0], self.data[1], train_size=trainSize, random_state=101)
        if not includeValid:
            return X_train, X_remaining, y_train, y_remaining
        X_valid, X_test, y_valid, y_test = train_test_split(X_remaining,y_remaining, test_size=validationSplit)
        return X_train, X_test, X_valid, y_train, y_test, y_valid
        

    def prettyPrintMatrix(self, A):
        for i in range(len(A)):
            row = ""
            for j in range(len(A[i])):
                row+=str(int(A[i][j])) + " "
            print(row)

if __name__=="__main__":
    dl = dataLoader(covariate=True, loadConcatData=True)
    X,y = dl.getTotalData()
    print(X[0])
    print(y[0])
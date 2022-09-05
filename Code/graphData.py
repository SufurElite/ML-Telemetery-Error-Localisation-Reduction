from kmlInterface import HabitatMap, Habitat
from sklearn.model_selection import train_test_split
import numpy as np
import utils

class dataLoader():
    def __init__(self, covariate:bool = False, isFlatten:bool = False, month:str = "June"):
        self.month = month
        self.covariate = covariate
        if self.covariate:
            self.habitatMap = HabitatMap()
        self.baseAdjacency = self.createBaseAdjacencyMatrix()
        self.isFlatten = isFlatten
        self.data = utils.loadRSSModelData(rowToAdjacencyMatrix=self.rowToAdjacencyMatrix)

    def createBaseAdjacencyMatrix(self):
        nodes = utils.loadNodes(rewriteUTM=True)
        nodeKeys = list(nodes.keys())
        A = np.zeros(shape=(len(nodes)+1,len(nodes)+1))
        for i in range(len(nodeKeys)):
            for j in range(i+1, len(nodeKeys)):
                A[i][j]=utils.distBetweenNodes(nodeKeys[i], nodeKeys[j], nodes)
                A[j][i]= A[i][j]
        return A

    def rowToAdjacencyMatrix(self, row):
        """ Turns a row with distances to each node the adjacency matrix will be of the shape n x n, where"""
        A = np.copy(self.baseAdjacency)
        for i in range(len(row)):
            A[len(A)-1][i] = row[i]
            A[i][len(A)-1] = row[i]
        if self.isFlatten:
            A = A.flatten()
        return A
    
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
    dl = dataLoader()
    X_train, X_test, X_valid, y_train, y_test, y_valid = dl.getSplitData(includeValid=True)
    dl.prettyPrintMatrix(X_valid[0])
    print(y_valid[0])
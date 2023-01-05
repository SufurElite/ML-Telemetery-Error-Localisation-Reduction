
from multilat import predictions
from model import rmseModel
from utils import loadModelData, loadRSSModelData, loadCovariateData, loadSections, pointToSection, loadNodes
import proximity
import csv


"""
        This file is going to serve as the master code to run the whole data prediction with different models
"""

def SM(month="June", rssiThreshold=-102):
    '''
        1. SM - simple multilateration
    '''
    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = False, optMultilat=False, month=month, otherMultilat=False)

    return results
def SM_MLD(month="October", rssiThreshold=-102):
    '''
        2. SM_MLD - simple multilateration and machine learning based on distance
    '''
    results = rmseModel(month=month,threshold=rssiThreshold, useCovariate=True,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)
    return results
def SM_MLS(month="June", rssiThreshold=-102):
    '''
        3. SM_MLS - simple multilateration and machine learning based on signal
    '''
    results = rmseModel(month=month,threshold=rssiThreshold, useCovariate=True,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)

    return results
def SM_P(month="June", rssiThreshold=-102):
    '''
        4. SM_P - simple multilateration with signal rewriting based on the probabilistic decision tree approach
    '''
    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = False, optMultilat=True, month=month, otherMultilat=False)

    return results
def ST_P(month="June", rssiThreshold=-102):
    '''
        5. ST_P - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
    '''
    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = True, optMultilat=False, month=month, otherMultilat=False)

    return results
def SM_P_MLD(month="June", rssiThreshold=-102):
    '''
        6. SM_P_MLD - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on distance
    '''
    results = rmseModel(month=month, threshold=rssiThreshold, useCovariate=True,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=False, optMultilat=True, otherMultilat=False)

    return results
def SM_P_MLS(month="June", rssiThreshold=-102):
    '''
        7. SM_P_MLS - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on signal
    '''
    results = rmseModel(month=month, threshold=rssiThreshold, useCovariate=True,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=False, optMultilat=True, otherMultilat=False)

    return results
def ST_P_MLD(month="June", rssiThreshold=-102):
    '''
        8. ST_P_MLD - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on distance
    '''
    results = rmseModel(month=month, threshold=rssiThreshold, useCovariate=True,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=True, optMultilat=False, otherMultilat=False)

    return results
def ST_P_MLS(month="June", rssiThreshold=-102):
    '''
        9. ST_P_MLS - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on signal
    '''
    results = rmseModel(month=month, threshold=rssiThreshold, useCovariate=True,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=True, optMultilat=False, otherMultilat=False)

    return results
def main():
    function_mappings = {
            "SM" : SM,
            "SM_MLD" : SM_MLD,
            "SM_MLS" : SM_MLS,
            "SM_P" : SM_P,
            "ST_P" : ST_P,
            "SM_P_MLD" : SM_P_MLD,
            "SM_P_MLS" : SM_P_MLS,
            "ST_P_MLD" : ST_P_MLD,
            "ST_P_MLS" : ST_P_MLS
    }
    results = function_mappings["SM"]()

    with open('../Code/Results/result_1.csv', "w+", newline="" ,encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Actual_UTMx", "Actual_UTMy", "Predicted_UTMx", "Predicted_UTMy", "Error"])
        for id in results:
            writer.writerow([results[id]["gt"][0], results[id]["gt"][1], results[id]["res"][0], results[id]["res"][1], results[id]["error"]])




if __name__=="__main__":
    main()

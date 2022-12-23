
from multilat import predictions
from model import rmseModel
from utils import loadModelData, loadRSSModelData, loadCovariateData, loadSections, pointToSection, loadNodes
import proximity


"""
        This file is going to serve as the master code to run the whole data prediction with different models
"""
#rmseModel(useCovariate=True,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)

def SM(month="June", rssiThreshold=-102):
    '''
        1. SM - simple multilateration
    '''
    results = predictions(rssiThreshold, month)

    return results
def SM_MLD(month="October", rssiThreshold=-102):
    '''
        2. SM_MLD - simple multilateration and machine learning based on distance
    '''
    #isTrilat=isTrilat, optMultilat=optMultilat, otherMultilat=otherMultilat
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
    pass
def ST_P(month="June", rssiThreshold=-102):
    '''
        5. ST_P - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
    '''
    pass
def SM_P_MLD(month="June", rssiThreshold=-102):
    '''
        6. SM_P_MLD - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on distance
    '''
    pass
def SM_P_MLS(month="June", rssiThreshold=-102):
    '''
        7. SM_P_MLS - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on signal
    '''
    pass
def ST_P_MLD(month="June", rssiThreshold=-102):
    '''
        8. ST_P_MLD - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on distance
    '''
    pass
def ST_P_MLS(month="June", rssiThreshold=-102):
    '''
        9. ST_P_MLS - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on signal
    '''
    pass
def SM_R(month="June", rssiThreshold=-102):
    '''
        11. SM_R - simple multialteration with rewriting the signals using recursion approach
        (the idea is the same as the probabilistic decision tree on)
    '''
    pass
def SM_R_MLD(month="June", rssiThreshold=-102):
    '''
        11. SM_R_MLD - simple multialteration with rewriting the signals using recursion approach
        (the idea is the same as the probabilistic decision tree on) then applying machine learning based on distance
    '''
    pass
def SM_R_MLS(month="June", rssiThreshold=-102):
    '''
        12. SM_R_MLS - simple multialteration with rewriting the signals using recursion approach
        (the idea is the same as the probabilistic decision tree on) then applying machine learning based on signal
    '''
    pass

def main():
    print(SM_MLD())

if __name__=="__main__":
    main()

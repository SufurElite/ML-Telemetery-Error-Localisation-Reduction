
from multilat import predictions
from utils import getPlotValues
from model import rmseModel
import csv, os
import argparse
from plot import plotGridWithPoints

"""
        This file is going to serve as the master code to run the whole data prediction with different models
"""

def SM(month="June", rssiThreshold=-102, useCovariate=False):
    """
        1. SM - simple multilateration
    """

    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = False, optMultilat=False, month=month, otherMultilat=False)

    #Plotting
    allErrors, errorLocs, errorDirections, gridS = getPlotValues(results, month)
    plt = plotGridWithPoints(errorLocs,gridSetup=gridS,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=False, colorScale = True, errors=allErrors, sameNodeColor=False)

    return results, plt
def SM_MLD(month="June", rssiThreshold=-102, useCovariate=False):
    """
        2. SM_MLD - simple multilateration and machine learning based on distance
    """
    results, plt = rmseModel(month=month,threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)

    return results, plt
def SM_MLS(month="June", rssiThreshold=-102, useCovariate=False):
    """
        3. SM_MLS - simple multilateration and machine learning based on signal
    """
    results, plt = rmseModel(month=month,threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True)

    return results, plt
def SM_P(month="June", rssiThreshold=-102, useCovariate=False):
    """
        4. SM_P - simple multilateration with signal rewriting based on the probabilistic decision tree approach
    """
    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = False, optMultilat=True, month=month, otherMultilat=False)

    #Plotting
    allErrors, errorLocs, errorDirections, gridS = getPlotValues(results, month)
    plt = plotGridWithPoints(errorLocs,gridSetup=gridS,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=False, colorScale = True, errors=allErrors, sameNodeColor=False)

    return results, plt
def ST_P(month="June", rssiThreshold=-102, useCovariate=False):
    """
        5. ST_P - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
    """
    results = predictions(rssiThreshold=rssiThreshold,keepNodeIds=False, isTrilat = True, optMultilat=False, month=month, otherMultilat=False)

    #Plotting
    allErrors, errorLocs, errorDirections, gridS = getPlotValues(results, month)
    plt = plotGridWithPoints(errorLocs,gridSetup=gridS,isSections=True,plotHabitats=True,imposeLimits=True, useErrorBars=False, colorScale = True, errors=allErrors, sameNodeColor=False)

    return results, plt
def SM_P_MLD(month="June", rssiThreshold=-102, useCovariate=False):
    """
        6. SM_P_MLD - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on distance
    """
    results, plt = rmseModel(month=month, threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=False, optMultilat=True, otherMultilat=False)

    return results, plt
def SM_P_MLS(month="June", rssiThreshold=-102, useCovariate=False):
    """
        7. SM_P_MLS - simple multilateration with signal rewriting based on the probabilistic decision tree approach
           then applying machinelearning based on signal
    """
    results, plt = rmseModel(month=month, threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=False, optMultilat=True, otherMultilat=False)

    return results, plt
def ST_P_MLD(month="June", rssiThreshold=-102, useCovariate=False):
    """
        8. ST_P_MLD - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on distance
    """
    results, plt = rmseModel(month=month, threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=True,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=True, optMultilat=False, otherMultilat=False)

    return results, plt
def ST_P_MLS(month="June", rssiThreshold=-102, useCovariate=False):
    """
        9. ST_P_MLS - simple trilateration with choosing the best 3 signals based on the probabilistic decision tree approach
           then applying machine learning based on signal
    """
    results, plt = rmseModel(month=month, threshold=rssiThreshold, useCovariate=useCovariate,isErrorData=False,plotError=True, useColorScale=True, useErrorBars = False, sameNodeColor=True, isTrilat=True, optMultilat=False, otherMultilat=False)

    return results, plt
def all_funcs(function_mappings, month, rssiThreshold, useCovariate):
    """
        Runs through all functions given particular parameters and saves the results
    """
    func_names = list(function_mappings.keys())
    for func_name in func_names:
        print("Currently on {}".format(func_name))
        specific_result(function_mappings, func_name, month, rssiThreshold, useCovariate)


def specific_result(function_mappings, func_name, month, rssiThreshold, useCovariate):
    """
        Takes in a specific set of parameters to test and returns the result
    """
    results, plt = function_mappings[func_name](month, rssiThreshold, useCovariate)

    
    # Csv file save
    fname = "results_"+func_name+"_"+month+"_"+str(rssiThreshold)+"_Habitat-"+str(useCovariate)+".csv"
    fpath="../Code/Results/Total/"
    # check if this directory exists, if not then create it
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    fpath+=fname

    with open(fpath, "w+", newline="" ,encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Actual_UTMx", "Actual_UTMy", "Predicted_UTMx", "Predicted_UTMy", "Error"])
        for id in results:
            writer.writerow([results[id]["gt"][0], results[id]["gt"][1], results[id]["res"][0], results[id]["res"][1], results[id]["error"]])
    #Image save
    imageName = "results_"+func_name+"_"+month+"_"+str(rssiThreshold)+"_Habitat-"+str(useCovariate)+".png"
    imagePath ="../Code/Results/Total/"+imageName
    plt.savefig(imagePath)


if __name__=="__main__":
    # Parameters
    month="June"
    rssiThreshold=-102
    useCovariate=False

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

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--func", help="The specific function to test. Possibilities are {}".format(list(function_mappings.keys())), type = str)
    parser.add_argument("-m", "--month", help="month (either \"March\", \"June\", etc.)", type = str)
    parser.add_argument("-t", "--threshold", help="Radio Signal Threshold", type=int)
    parser.add_argument("-c", "--covariate", help="If you want to include the habitat prediction in the model, include this flag", type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("-a", "--all_variants", help="If you want to run all the functions, include this flag", type=bool, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.month:
        month = args.month
    if args.threshold:
        rssiThreshold=args.threshold
    if args.covariate!=None:
        useCovariate = args.covariate

    if args.all_variants:
        print("Run all the variants with parameters: \n\tMonth: {}, \n\tThreshold: {} \n\tUse Habitat Predictions: {}".format(month,rssiThreshold,useCovariate))
        all_funcs(function_mappings, month, rssiThreshold, useCovariate)
    elif args.func in function_mappings.keys():
        print("Running {} with parameters: \n\tMonth: {}, \n\tThreshold: {} \n\tUse Habitat Predictions: {}".format(args.func,month,rssiThreshold,useCovariate))
        specific_result(function_mappings, args.func, month, rssiThreshold, useCovariate)
    else:
        print("""Must either select a specific function or run all the variants
                \nE.g. a command would be written as python master.py -a True or python master.py -f SM
                \nrun python master.py --help for more information""")

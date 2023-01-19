"""
    Once the master.py file has been run and all the different test csvs are ready to be analysed,
    create a master table that distils the information to be viewed
"""
import os
import pandas as pd
import argparse

# these are the RSSI values to filter the totality of the data down
default_RSSIs = [-87, -90, -93, -96, -99, -101, -102]
# the directory with the data
dir = os.getcwd()+"/Results/Total/"

def get_results(df:pd.DataFrame):
    """
        Given the particular csv dataframe
        get the number of rows, its mean, and its standard deviation
    """
    num_rows = len(df)
    mean = df.loc[:, "Error"].mean()
    std = df.loc[:, "Error"].std()
    return num_rows, mean, std

def disect_name(fname: str):
    """
        Given a csv file's name, we want to be able to extract
        its function, month, threshold value, and habitat
    """

    starting_text = "results_"
    months = ["March","June", "October", "November"]

    method = ""
    month = ""
    threshold = 0,
    habitat = False
    pruned = False

    # the start of fname is simply results_, so let's start after that
    fname=fname[len(starting_text):]
    threshold_idx = -1

    # find the month
    for m in months:
        month_idx = fname.find(m)
        if month_idx>-1:
            # once the month is found, both the method, which is everything before the month
            # the month and then the start to the threshold, which is right after the month
            # can be found
            method = fname[:month_idx-1]
            threshold_idx = fname.find("-")
            month = fname[month_idx:threshold_idx-1]
            break

    if month == "": raise Exception("No month found")

    # the threshold ends with the start of the Habitat
    # we try to cast the threshold as an int
    habitat_idx = fname.find("_Habitat")
    try:
        threshold = int(fname[threshold_idx:habitat_idx])
    except:
        raise Exception("Incorrectly formatted threshold found. Found: {}".format(fname[threshold_idx:habitat_idx]))
    #Find whether the data is pruned or not
    pruned_idx = fname.find("_Pruned")
    if fname[habitat_idx+1+len("Habitat-"):pruned_idx] == "True":
        habitat = True
    # the habitat value is then either True or False and starts
    # after _Habitat- up to the start of the file extension
    pruned_val = fname[pruned_idx+len("Habitat-"):len(fname)-4]
    if pruned_val=="True":
        pruned = True
    return method, month, threshold, habitat, pruned

def main(certain_threshold:bool = False):
    """
        Walk through the directory of the all the result test data
        and parse the different variations from the title and deduce
        the test error and standard deviation
    """
    cols = ["Function","Month","Threshold","Habitat Covariate","Pruned Data","Number of Tests","Average Test Error","Standard Deviation of Test Error"]
    result_df = pd.DataFrame(columns=cols)

    for file in os.listdir(dir):
        fname = os.fsdecode(file)
        if not fname.endswith(".csv"):
            continue

        method, month, threshold, habitat, pruned = disect_name(fname)
        if certain_threshold and threshold not in default_RSSIs: continue
        data = pd.read_csv(dir+fname)
        numVals, mean, std = get_results(data)
        s = pd.DataFrame([[method, month, threshold, habitat, pruned, numVals,
                               mean, std]], columns = cols)


        result_df = pd.concat([result_df, s], ignore_index = True, axis =0)

    result_df = result_df.sort_values(by=["Average Test Error"]).reset_index(drop=True)
    print("Number of rows in the dataframe: {}".format(len(result_df)))
    print(result_df.head())
    results_path = os.getcwd()+"/Results/"
    if certain_threshold:
        results_path+="selected_rss_master_test_table.csv"
    else:
        results_path+="total_rss_master_test_table.csv"
    result_df.to_csv(results_path, index=False)
    print("Saved data to {}".format(results_path))


if __name__=="__main__":

    if not os.path.isdir(dir):
        raise Exception("Could not find Results/Total/. You first need to run master.py")

    include_threshold = False

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--all_threshold", help="If you want to find the values for all the thresholds, include this flag", type=bool, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.all_threshold!=None:
        include_threshold = True

    main(include_threshold)

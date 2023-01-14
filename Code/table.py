import os
import pandas as pd
import argparse 

default_RSSIs = [-87, -90, -93, -96, -99, -101, -102]
dir = os.getcwd()+"/Results/Total/"

def get_results(df:pd.DataFrame):
    num_rows = len(df)
    mean = df.loc[:, "Error"].mean()
    std = df.loc[:, "Error"].std()
    return num_rows, mean, std

def disect_name(fname: str):
    starting_text = "results_"
    months = ["March","June", "October", "November"]

    method = ""
    month = ""
    threshold = 0,
    habitat = False

    

    fname=fname[len(starting_text):]
    threshold_idx = -1
    for m in months:
        idx = fname.find(m)
        if idx>-1:
            method = fname[:idx-1]
            threshold_idx = fname.find("-")
            month = fname[idx:threshold_idx-1]
            break
    
    if month == "": raise Exception("No month found")
    
    habitat_idx = fname.find("_Habitat")
    try:
        threshold = int(fname[threshold_idx:habitat_idx])
    except:
        raise Exception("Incorrectly formatted threshold found. Found: {}".format(fname[threshold_idx:habitat_idx]))

    hab_val = fname[habitat_idx+1+len("Habitat-"):len(fname)-4]
    if hab_val=="True":
        habitat = True

    return method, month, threshold, habitat

def main(certain_threshold:bool = False):
    cols = ["Function","Month","Threshold","Habitat Covariate","Number of Tests","Average Test Error","Standard Deviation of Test Error"]
    result_df = pd.DataFrame(columns=cols)

    for file in os.listdir(dir):
        fname = os.fsdecode(file)
        if not fname.endswith(".csv"):
            continue
        
        method, month, threshold, habitat = disect_name(fname)
        if certain_threshold and threshold not in default_RSSIs: continue
        data = pd.read_csv(dir+fname)
        numVals, mean, std = get_results(data)
        s = pd.DataFrame([[method, month, threshold, habitat, numVals, 
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
    include_threshold = False
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all_threshold", help="If you want to find the values for all the thresholds, include this flag", type=bool, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.all_threshold!=None:
        include_threshold = True
    main(include_threshold)
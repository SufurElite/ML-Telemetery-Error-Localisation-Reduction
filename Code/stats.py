"""
"""

FTABLE = [  ]
import pandas as pd
import math
from scipy.stats import f, t

def t_test(mean1, mean2, std1, std2, n1, n2, alpha = 0.05):
    # calculate the pooled standard deviation
    sp = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2)
    sp = sp**0.5
    
    # calculate the t-statistic
    t_stat = (mean1 - mean2) / (sp * (1/n1 + 1/n2)**0.5)
    
    # calculate the degrees of freedom
    dof = n1 + n2 - 2
    
    # calculate the two-sided p-value
    p_value = 2 * t.cdf(-abs(t_stat), dof)
    if p_value > alpha:
        return False, p_value
    return True, p_value

def f_test(std1, std2, df1,df2,alpha=0.05,print_p:bool = False):
    F = (std1**2 / std2**2) if std1 > std2 else (std2**2 / std1**2)
    
    p_value = 1-(f.cdf(F, df1, df2) if std1 >std2 else f.cdf(F, df2, df1) )
    if print_p:
        print(p_value)
    if p_value > alpha:
        return False, p_value
    return True, p_value


def getInitialStats(month="November"):
    """ Used to decide the top 4 model selection that we want to test more thoroughly"""
    with open("Results/selected_rss_master_test_table.csv", "r") as f:
        df = pd.read_csv(f)
    df.sort_values(by = ['Function','Number of Tests'], axis=0, ascending=[True,False], inplace=True, ignore_index=True, key=None)
    
    df = df[df['Month']==month]
    # covariate is negligible
    df = df[df['Habitat Covariate']==True]
    # excluding the outside values
    df = df[df['Pruned Data']==True]

    df = df[df['Threshold'].isin([-102,-96])]
    
    f_test_dicts = []
    t_test_dicts = []
    alphas = [0.05, 0.01, 0.005]
    models = set(df["Function"])
    print(df)
    print(len(df))
    print(models)
    # highly ineffecient, should be reworked
    # but fine with small numbers of tests
    for alpha in alphas:
        for index, row in df.iterrows():
            modelOne = row["Function"]
            threshold = row["Threshold"]
            degFreeOne = row["Number of Tests"]-1
            stdOne=row["Standard Deviation of Test Error"]
            tmp_df = df[df["Function"]!=modelOne]
            tmp_df  = tmp_df[tmp_df["Threshold"]==threshold]
            modelOneError =row["Average Test Error"]
            for idx, second_row in tmp_df.iterrows():
                modelTwo = second_row["Function"]
                degFreeTwo = second_row["Number of Tests"]-1
                stdTwo=second_row["Standard Deviation of Test Error"]
                modelTwoError =second_row["Average Test Error"]
                f_p_value,f_accept=f_test(stdOne,stdTwo, degFreeOne,degFreeTwo, alpha)
                t_p_value,t_accept=t_test(modelOneError, modelTwoError, stdOne,stdTwo, degFreeOne+1,degFreeTwo+1, alpha = 0.05)
                smallerModel = modelOne
                if stdOne>stdTwo:
                    smallerModel = modelTwo
                f_test_dicts.append({"model1":modelOne,"model2":modelTwo,"alpha":alpha,"p_value":f_p_value,"rejected":f_accept,"threshold":threshold,"MorePrecise":smallerModel, "ModelOneError":round(modelOneError,2), "modelTwoError":round(modelTwoError,2)})

                t_test_dicts.append({"model1":modelOne,"model2":modelTwo,"alpha":alpha,"p_value":t_p_value,"rejected":t_accept,"threshold":threshold,"MorePrecise":smallerModel, "ModelOneError":round(modelOneError,2), "modelTwoError":round(modelTwoError,2)})
    f_test_results = pd.DataFrame(f_test_dicts)
    t_test_results = pd.DataFrame(t_test_dicts)
    f_test_results.sort_values(by = ['model1','model2'], axis=0, ascending=[True, True], inplace=True, ignore_index=True, key=None)
    t_test_results.sort_values(by = ['model1','model2'], axis=0, ascending=[True, True], inplace=True, ignore_index=True, key=None)
    f_test_results.to_csv("Results/f_test_selected.csv",index=False)
    t_test_results.to_csv("Results/t_test_selected.csv",index=False)
def runModels():
    models = []

    pass

if __name__=="__main__":
    getInitialStats()
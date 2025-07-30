# predictive class for model import
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import copy
import sys
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

class PredictiveModel:
    # wrapper for moving average, to fit sklearn format
    class MovingAverage:
        def __init__(self):
            pass
        def fit(self,input,output):
            pass
        def predict(self,input):
            return_value = np.zeros((len(input),1))
            for i in range(len(input)):
                interval = round(len(input[i])/5)
                for j in range(0,len(input[i]),interval):
                    return_value[i][0] += input[i][j]/5
            return return_value
    def __init__(self,_model_path,_metadata_path):
        self.model = joblib.load(_model_path)
        self.columns = []
        with open(_metadata_path,"r") as f:
            metadata_arr = f.readline().split()
            for i in range(1,len(metadata_arr)):
                self.columns.append(int(metadata_arr[i]))

    # takes a DataFrame with 5 rows of required data and returns prediction result
    def predict(self,data):
        input_data = []
        cnt = 0
        for i in range(5):
            for j in self.columns:
                input_data.append(data.iloc[i,j])
        input_data = np.array(input_data).reshape(1,len(input_data))
        return self.model.predict(input_data)

def find_residuals(_model_path,_metadata_path,_file_path,_test_arr,_name,_save_path):
    raw_data = pd.read_csv(_file_path)
    residuals = []
    model = PredictiveModel(_model_path,_metadata_path)
    for test_value in _test_arr:
        test_case = raw_data.iloc[test_value:test_value+5]
        test_answer = raw_data.iloc[test_value+5,1]
        predict_answer = model.predict(test_case)[0]
        residuals.append(predict_answer-test_answer)
    residuals = np.array(residuals)
    
    # basic plot 
    plt.figure(figsize=(14,8))
    sns.histplot(residuals,stat="density")
    sns.despine()
    plt.xlabel("Frequency",fontsize=18)
    plt.ylabel("Residual",fontsize=18)
    plt.title(f"Residuals for Best Predictor, {_name} Data, Histogram",fontsize=24)
    # add normal pdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, color='red', linewidth=2, label='Normal PDF')
    # output
    plt.savefig(_save_path+"_histplot.png")

    # basic plot 
    plt.figure(figsize=(14,8))
    sns.ecdfplot(residuals)
    sns.despine()
    plt.xlabel("Frequency",fontsize=18)
    plt.ylabel("Residual",fontsize=18)
    plt.title(f"Residuals for Best Predictor, {_name} Data, ECDF",fontsize=24)
    # add normal pdf
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.cdf(x, np.mean(residuals), np.std(residuals))
    plt.plot(x, p, color='red', linewidth=2, label='Normal PDF')
    # output
    plt.savefig(_save_path+"_ecdf.png")

    print(f"Fitted normal distribution to {_name} predictor: mean {np.mean(residuals)}, stddev {np.std(residuals)}")

def main():
    find_residuals("../../output/data_appendix_output/delay_model.txt",
                "../../output/data_appendix_output/delay_model_data.txt",
                "../../data/analysis_data/delay_inputs.csv",
                range(1499,1666),
                "Delay",
                "../../output/results/delay_predictor_residuals")
    find_residuals("../../output/data_appendix_output/gse_model.txt",
                "../../output/data_appendix_output/gse_model_data.txt",
                "../../data/analysis_data/gse_inputs.csv",
                range(3775,4194),
                "GSE",
                "../../output/results/gse_predictor_residuals")

main()
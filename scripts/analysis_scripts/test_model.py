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
import shap

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
                interval = round(len(input[i])/int(sys.argv[1]))
                for j in range(0,len(input[i]),interval):
                    return_value[i][0] += input[i][j]/int(sys.argv[1])
            return return_value
    def __init__(self,_model_path,_metadata_path):
        self.model = joblib.load(_model_path)
        self.columns = []
        with open(_metadata_path,"r") as f:
            metadata_arr = f.readline().split()
            for i in range(1,len(metadata_arr)):
                self.columns.append(int(metadata_arr[i]))

    # returns prediction result
    def predict(self,data):
        return self.model.predict(np.array(data).reshape(1,len(data)))

def display_violin_plot(model, data, raw_data,save_path):
    explainer = shap.Explainer(model.model)
    shap_values = explainer.shap_values(data)
    data_names = []
    for i in range(1,6):
        for x in model.columns:
            data_names.append(raw_data.columns[x]+f"\n{6-i}d before")
    plt.figure(figsize=(14,8))
    shap.plots.violin(shap_values,feature_names=data_names,max_display=10,show=False,plot_size=[28,16])
    plt.savefig(save_path+"_importance.png")

def find_residuals(_model_path,_metadata_path,_file_path,_test_arr,_name,_save_path):
    raw_data = pd.read_csv(_file_path)
    residuals = []
    model = PredictiveModel(_model_path,_metadata_path)
    full_inputs = []
    for test_value in _test_arr:
        test_case = raw_data.iloc[test_value:test_value+int(sys.argv[1])]
        test_answer = raw_data.iloc[test_value+int(sys.argv[1]),1]
        input_data = []
        cnt = 0
        for i in range(int(sys.argv[1])):
            for j in model.columns:
                input_data.append(test_case.iloc[i,j])
        full_inputs.append(input_data)
        predict_answer = model.predict(input_data)[0]
        residuals.append(predict_answer-test_answer)
    full_inputs = np.array(full_inputs)
    display_violin_plot(model,full_inputs,raw_data,_save_path)
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
    plt.savefig(_save_path+"_residuals_histplot.png")

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
    plt.savefig(_save_path+"_residuals_ecdf.png")

    print(f"Fitted normal distribution to {_name} predictor: mean {np.mean(residuals)}, stddev {np.std(residuals)}")

def compute_range(_dataset_size,_row_count,_split_prop):
    data_points = _dataset_size-_row_count
    train_size = round(_split_prop*data_points)
    return range(train_size+1,data_points)

def main():
    find_residuals("../../output/data_appendix_output/delay_model.txt",
                "../../output/data_appendix_output/delay_model_data.txt",
                "../../data/analysis_data/delay_inputs.csv",
                compute_range(1671,int(sys.argv[1]),float(sys.argv[2])),
                "Delay",
                "../../output/results/delay_predictor")
    find_residuals("../../output/data_appendix_output/gse_model.txt",
                "../../output/data_appendix_output/gse_model_data.txt",
                "../../data/analysis_data/gse_inputs.csv",
                compute_range(4199,int(sys.argv[1]),float(sys.argv[2])),
                "GSE",
                "../../output/results/gse_predictor")

if __name__ == "__main__":
    main()
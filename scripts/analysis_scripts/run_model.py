# predictive class for model import
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

# for loading model
# model_path = "../../output/data_appendix_output/delay_model.txt"
model_path = "../../output/data_appendix_output/gse_model.txt"

# just for debug loading data
file_path = "../../data/analysis_data/delay_inputs.csv"
# file_path = "../../data/analysis_data/GSE_inputs.csv"

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
    def __init__(self,filepath):
        self.model = joblib.load(filepath)
    def predict(self,data):
        return 0

a = PredictiveModel(model_path)
print(a.predict(1))
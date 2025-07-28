# Tests 11 models on performance in predicting delay data and gated station entry data.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from keras.models import Sequential
from keras.layers import LSTM as KerasLSTM, Dense
import sys
import copy
import os
from contextlib import redirect_stdout

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

# wrapper for Keras LSTM, to fit sklearn format
class LSTM:
    def __init__(self):
        self.model = None

    def fit(self, input, output):
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                _input = input.reshape((len(input),int(len(input[0])/5),5))
                self.model = Sequential()
                self.model.add(KerasLSTM(64, input_shape=(int(len(input[0])/5),5)))
                self.model.add(Dense(1))
                self.model.compile(optimizer='adam', loss='mse')
                self.model.fit(_input, output, epochs=200, batch_size=32, verbose=0)
        return self

    def predict(self, input):
        return_value = None
        with open(os.devnull, 'w') as fnull:
            with redirect_stdout(fnull):
                _input = input.reshape((len(input),int(len(input[0])/5),5))
                return_value = self.model.predict(_input)
        return return_value

# list of models; note LSTM was eventually discarded
model_collection = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Linear": LinearRegression(),
    "Ridge": Ridge(max_iter=2500),
    "Lasso": Lasso(),
    "GradientBoost": GradientBoostingRegressor(random_state=0),
    "SupportVector": SVR(),
    "MultilayerPerceptron": MLPRegressor(max_iter=500, random_state=0),
    "kNearestNeighbor": KNeighborsRegressor(),
    "MovingAverage": MovingAverage(),
    "LSTM": LSTM(),
    "Poisson": PoissonRegressor(max_iter=500)
}

# read analysis data and convert into format required for ML training
def process_data(source,input_rows,input_len,output_len,input_cols,output_cols):
    df = pd.read_csv(source)
    data_len = len(df)-input_rows
    input = np.ones((data_len,input_len))
    output = np.ones((data_len,output_len))
    for i in range(data_len):
        input_curr = np.ones(input_rows*len(input_cols))
        cnt = 0
        for j in range(input_rows):
            for l in input_cols:
                input_curr[cnt] = df.iloc[i+j,l]
                cnt += 1
        output_curr = np.ones(output_len)
        cnt = 0
        for j in output_cols:
            output_curr[cnt] = df.iloc[i+input_rows,j]
            cnt += 1
        input[i] = input_curr
        output[i] = output_curr
    return input,output

# split data into train and test sets (prop is set to 0.9, or 90% train 10% test, when used); test set is the later segment of the data
def split_data(input,output,prop):
    input_split = round(prop*len(input))
    _input = np.split(input,[input_split])
    output_split = round(prop*len(output))
    _output = np.split(output,[output_split])
    return _input[0],_input[1],_output[0],_output[1]

# train model
def train_model(input,output,base_model):
    output = output.ravel()
    base_model.fit(input,output)
    return base_model

# test model, return RMSE
def test_model(input,output,model):
    output = output.ravel()
    pred = model.predict(input)
    return root_mean_squared_error(output,pred)

# try running model with some set of specs
def try_running(_source,
                _input_rows,
                _input_cols,
                _output_cols,
                _input_len,
                _output_len,
                _split_prop,
                _base_model,
                _model_name):
    raw_input,raw_output = process_data(_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols)
    train_input,test_input,train_output,test_output = split_data(raw_input,raw_output,_split_prop)
    model_1 = train_model(train_input,train_output,_base_model)
    rmse_1 = test_model(test_input,test_output,model_1)
    print(f"{_model_name}: RMSE is {float(rmse_1)*100:.2f}.")

# run given an array of strings corresponding to inputs
def run_by_name(source,inputs):
    corresponding_cols = {
        "target metric": [1],
        "day of week": [2],
        "season": [3],
        "weather": [4,5,6,7],
        "day of week OHE": [8,9,10,11,12,13,14],
        "season OHE": [15,16,17,18],
        "scaled day of week": [19],
        "scaled season": [20],
        "scaled weather": [21,22,23,24],
        "scaled day of week OHE": [25,26,27,28,29,30,31],
        "scaled season OHE": [32,33,34,35]
    }
    for input_name in inputs:
        if input_name not in corresponding_cols:
            raise ValueError("Data name does not exist.")
    model_name = f"Using model ({sys.argv[1]}), inputs of previous 5-day ("
    for i in range(len(inputs)):
        if i > 0:
            model_name += ', '
        model_name += inputs[i]
    model_name += f") to predict target metric"
    input_cols = []
    for input_name in inputs:
        input_cols += corresponding_cols[input_name]
    try_running(
    source,
    _input_rows = 5,
    _input_cols = input_cols,
    _output_cols = [1],
    _input_len = len(input_cols)*5,
    _output_len = 1,
    _split_prop = 0.9,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = model_name
    )

# run a series of tests with differing specs
def run_tests(source):
    # test with no additional data
    run_by_name(source,["target metric"])

    # test with all additional data
    run_by_name(source,["target metric","day of week","season","weather"])
    run_by_name(source,["target metric","day of week OHE","season OHE","weather"])
    run_by_name(source,["target metric","scaled day of week","scaled season","scaled weather"])
    run_by_name(source,["target metric","scaled day of week OHE","scaled season OHE","scaled weather"])

    # test with additional day of week data
    run_by_name(source,["target metric","day of week"])
    run_by_name(source,["target metric","day of week OHE"])
    run_by_name(source,["target metric","scaled day of week"])
    run_by_name(source,["target metric","scaled day of week OHE"])

    # test with additional season data
    run_by_name(source,["target metric","season"])
    run_by_name(source,["target metric","season OHE"])
    run_by_name(source,["target metric","scaled season"])
    run_by_name(source,["target metric","scaled season OHE"])

    # test with additional weather data
    run_by_name(source,["target metric","weather"])
    run_by_name(source,["target metric","scaled weather"])

    # test with additional day of week, season data
    run_by_name(source,["target metric","day of week","season"])
    run_by_name(source,["target metric","day of week OHE","season OHE"])
    run_by_name(source,["target metric","scaled day of week","scaled season"])
    run_by_name(source,["target metric","scaled day of week OHE","scaled season OHE"])

    # test with additional day of week, weather data
    run_by_name(source,["target metric","day of week","weather"])
    run_by_name(source,["target metric","day of week OHE","weather"])
    run_by_name(source,["target metric","scaled day of week","scaled weather"])
    run_by_name(source,["target metric","scaled day of week OHE","scaled weather"])
    
    # test with additional season, weather data
    run_by_name(source,["target metric","season","weather"])
    run_by_name(source,["target metric","season OHE","weather"])
    run_by_name(source,["target metric","scaled season","scaled weather"])
    run_by_name(source,["target metric","scaled season OHE","scaled weather"])

# run tests on both gated station entry and delay data
def main():
    print("GSE data:")
    run_tests("../../data/analysis_data/GSE_inputs.csv")
    print("Delay data:")
    run_tests("../../data/analysis_data/delay_inputs.csv")

main()
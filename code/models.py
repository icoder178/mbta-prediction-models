# identical tests, sklearn models
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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


model_collection = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    "Linear": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "GradientBoost": GradientBoostingRegressor(random_state=0),
    "SupportVector": SVR(),
    "MultilayerPerceptron": MLPRegressor(max_iter=1000, random_state=0),
    "kNearestNeighbor": KNeighborsRegressor(),
    "MovingAverage": MovingAverage(),
    "LSTM": LSTM()
}

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

def split_data(input,output,prop):
    input_split_pos = round(prop*len(input))
    _input = np.split(input,[input_split_pos,len(input)-input_split_pos])
    output_split_pos = round(prop*len(output))
    _output = np.split(output,[output_split_pos,len(output)-output_split_pos])
    return _input[1],_input[0],_output[1],_output[0]

def train_model_1(input,output,base_model):
    output = output.ravel()
    base_model.fit(input,output)
    return base_model

def test_model(input,output,model):
    output = output.ravel()
    pred = model.predict(input)
    return root_mean_squared_error(output,pred)

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
    model_1 = train_model_1(train_input,train_output,_base_model)
    rmse_1 = test_model(test_input,test_output,model_1)
    print(f"{_model_name}: RMSE is {float(rmse_1)*100:.2f}.")

def run_tests(source):
    try_running(
    source,
    _input_rows = 5,
    _input_cols = [1],
    _output_cols = [1],
    _input_len = 5,
    _output_len = 1,
    _split_prop = 0.1,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = f"{sys.argv[1]}, no additional data"
    )
    try_running(  
    source,  
    _input_rows = 5,
    _input_cols = [1,2,3,4,5,6,7],
    _output_cols = [1],
    _input_len = 35,
    _output_len = 1,
    _split_prop = 0.1,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = f"{sys.argv[1]}, additional weather, day of week, season"
    )
    try_running(    
    source,
    _input_rows = 5,
    _input_cols = [1,19,20,21,22,23,24],
    _output_cols = [1],
    _input_len = 35,
    _output_len = 1,
    _split_prop = 0.1,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = f"{sys.argv[1]}, additional normalized weather, day of week, season"
    )
    try_running(
    source,    
    _input_rows = 5,
    _input_cols = [1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
    _output_cols = [1],
    _input_len = 80,
    _output_len = 1,
    _split_prop = 0.1,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = f"{sys.argv[1]}, additional weather, day of week one-hot encoding, season one-hot encoding"
    )
    try_running(
    source,    
    _input_rows = 5,
    _input_cols = [1,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
    _output_cols = [1],
    _input_len = 80,
    _output_len = 1,
    _split_prop = 0.1,
    _base_model = copy.deepcopy(model_collection[sys.argv[1]]),
    _model_name = f"{sys.argv[1]}, additional normalized weather, day of week one-hot encoding, season one-hot encoding"
    )

print("GSE data:")
run_tests("../data/processed/GSE_inputs.csv")
print("Delay data:")
run_tests("../data/processed/delay_inputs.csv")
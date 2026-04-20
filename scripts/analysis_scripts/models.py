# Tests 10 models on performance in predicting delay data and gated station entry data.
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import copy
import sys
import joblib
import random
import ast
import os

# wrapper for moving average, to fit sklearn format
class MovingAverage:
    def __init__(self):
        pass
    def fit(self,input,output):
        pass
    def predict(self,input):
        return_value = np.zeros((len(input),1))
        for i in range(len(input)):
            interval = round((len(input[i]))/int(sys.argv[2]))
            for j in range(0,len(input[i]),interval):
                return_value[i][0] += input[i][j]/int(sys.argv[2])
        return return_value

# list of models
model_dict = {
    "RandomForest": (
        "../../data/intermediate_data/RandomForest",
        RandomForestRegressor(random_state=0)
    ),
    "Linear": (
        "../../data/intermediate_data/Linear",
        LinearRegression()
    ),
    "Ridge": (
        "../../data/intermediate_data/Ridge",
        Ridge()
    ),
    "Lasso": (
        "../../data/intermediate_data/Lasso",
        Lasso(random_state=0)
    ),
    "GradientBoost": (
        "../../data/intermediate_data/GradientBoost",
        GradientBoostingRegressor(random_state=0)
    ),
    "SupportVector": (
        "../../data/intermediate_data/SupportVector",
        LinearSVR(random_state=0)
    ),
    "MultilayerPerceptron": (
        "../../data/intermediate_data/MultilayerPerceptron",
        MLPRegressor(random_state=0)
    ),
    "kNearestNeighbor": (
        "../../data/intermediate_data/kNearestNeighbor",
        KNeighborsRegressor()
    ),
    "MovingAverage": (
        "../../data/intermediate_data/MovingAverage",
        MovingAverage()
    ),
    "Poisson": (
        "../../data/intermediate_data/Poisson",
        PoissonRegressor()
    )
}

# columns linking input strings to results
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

# input sets tested for each model, in output order
input_sets = [
    ["target metric"],
    ["target metric","day of week","season","weather"],
    ["target metric","day of week OHE","season OHE","weather"],
    ["target metric","scaled day of week","scaled season","scaled weather"],
    ["target metric","scaled day of week OHE","scaled season OHE","scaled weather"],
    ["target metric","day of week"],
    ["target metric","day of week OHE"],
    ["target metric","scaled day of week"],
    ["target metric","scaled day of week OHE"],
    ["target metric","season"],
    ["target metric","season OHE"],
    ["target metric","scaled season"],
    ["target metric","scaled season OHE"],
    ["target metric","weather"],
    ["target metric","scaled weather"],
    ["target metric","day of week","season"],
    ["target metric","day of week OHE","season OHE"],
    ["target metric","scaled day of week","scaled season"],
    ["target metric","scaled day of week OHE","scaled season OHE"],
    ["target metric","day of week","weather"],
    ["target metric","day of week OHE","weather"],
    ["target metric","scaled day of week","scaled weather"],
    ["target metric","scaled day of week OHE","scaled weather"],
    ["target metric","season","weather"],
    ["target metric","season OHE","weather"],
    ["target metric","scaled season","scaled weather"],
    ["target metric","scaled season OHE","scaled weather"]
]

selected_hyperparameters = {}

display_names = {
    "SupportVector": "Linear Support Vector Regression"
}

# returns user-facing model name
def display_model_name(_model_name):
    return display_names.get(_model_name,_model_name)

# returns stable name for a feature set
def feature_set_name(_inputs):
    return " + ".join(_inputs)

# loads selected hyperparameters for final training
def load_selected_hyperparameters(_model_name):
    global selected_hyperparameters
    if _model_name in selected_hyperparameters:
        return selected_hyperparameters[_model_name]
    file_path = "../../data/intermediate_data/selected_hyperparameters.csv"
    if not os.path.exists(file_path):
        file_path = f"../../data/intermediate_data/selected_hyperparameters_{_model_name}.csv"
    if not os.path.exists(file_path):
        selected_hyperparameters[_model_name] = {}
        return selected_hyperparameters[_model_name]
    data = pd.read_csv(file_path)
    data = data[data['model'] == _model_name]
    return_value = {}
    for i in range(len(data)):
        params_text = data.iloc[i]['params']
        if pd.isna(params_text):
            params = {}
        else:
            params = ast.literal_eval(params_text)
        return_value[(data.iloc[i]['task'],data.iloc[i]['feature_set'])] = params
    selected_hyperparameters[_model_name] = return_value
    return return_value

# returns a model configured with selected hyperparameters
def configured_model(_model_name,_base_model,_task_name,_inputs):
    return_value = copy.deepcopy(_base_model)
    params_map = load_selected_hyperparameters(_model_name)
    params_key = (_task_name,feature_set_name(_inputs))
    if len(params_map) > 0 and params_key not in params_map:
        raise ValueError(f"No selected hyperparameters for {_model_name}, {_task_name}, {feature_set_name(_inputs)}.")
    params = params_map.get(params_key,{})
    if len(params) > 0:
        return_value.set_params(**params)
    return return_value

# read 1 analysis data file and convert into format required for ML training
def process_single_data(_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols):
    df = pd.read_csv(_source).to_numpy()
    data_len = len(df)-_input_rows
    input = np.ones((data_len,_input_len))
    output = np.ones((data_len,_output_len))
    for i in range(data_len):
        input_curr = np.ones(_input_rows*len(_input_cols))
        cnt = 0
        for j in range(_input_rows):
            for l in _input_cols:
                input_curr[cnt] = df[i+j][l]
                cnt += 1
        output_curr = np.ones(_output_len)
        cnt = 0
        for j in _output_cols:
            output_curr[cnt] = df[i+_input_rows][j]
            cnt += 1
        input[i] = input_curr
        output[i] = output_curr
    return input,output

# read train and test analysis data and convert into format required for ML training
def process_data(_train_source,_test_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols):
    train_input,train_output = process_single_data(_train_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols)
    test_input,test_output = process_single_data(_test_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols)
    if sys.argv[3] != "NO_BOOTSTRAP":
        random.seed(sys.argv[3]+" train")
        selections = np.array([random.randint(0,len(train_input)-1) for _ in range(len(train_input))])
        selections.sort()
        train_input = train_input[selections]
        train_output = train_output[selections]
        random.seed(sys.argv[3]+" test")
        selections = np.array([random.randint(0,len(test_input)-1) for _ in range(len(test_input))])
        selections.sort()
        test_input = test_input[selections]
        test_output = test_output[selections]
    return train_input,test_input,train_output,test_output

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
# return rmse, model
def try_running(_train_source,
                _test_source,
                _input_rows,
                _input_cols,
                _output_cols,
                _input_len,
                _output_len,
                _base_model):
    train_input,test_input,train_output,test_output = process_data(_train_source,_test_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols)
    current_model = train_model(train_input,train_output,_base_model)
    rmse = test_model(test_input,test_output,current_model)
    return rmse,current_model

# run given an array of strings corresponding to inputs
# return rmse, model, output string
def run_by_name(_train_source,_test_source,_inputs,_model_name,_base_model,_task_name):
    for input_name in _inputs:
        if input_name not in corresponding_cols:
            raise ValueError("Data name does not exist.")
    description = f"Using model ({display_model_name(_model_name)}), inputs of previous {int(sys.argv[2])}-day ("
    for i in range(len(_inputs)):
        if i > 0:
            description += ', '
        description += _inputs[i]
    description += f") to predict target metric"
    input_cols = []
    for input_name in _inputs:
        input_cols += corresponding_cols[input_name]
    current_model = configured_model(_model_name,_base_model,_task_name,_inputs)
    rmse, result_model = try_running(
    _train_source = _train_source,
    _test_source = _test_source,
    _input_rows = int(sys.argv[2]),
    _input_cols = input_cols,
    _output_cols = [1],
    _input_len = len(input_cols)*int(sys.argv[2]),
    _output_len = 1,
    _base_model = current_model
    )
    rmse = float(rmse)
    return [rmse,result_model,f"{description}: RMSE is {rmse}."]


# run a series of tests with differing specs on the same model
# return .out file output, data on best model, best model
def run_tests(_train_source,_test_source,_model_name,_model,_task_name):
    results = []
    for current_inputs in input_sets:
        results.append(run_by_name(_train_source,_test_source,current_inputs,_model_name,_model,_task_name)
                       + current_inputs)

    descriptions = ""
    rmse_array = []
    for x in results:
        descriptions += x[2]
        descriptions += "\n"
        rmse_array.append(x[0])
    min_rmse = results[0][0]
    min_rmse_data = results[0]
    for i in range(1,len(results)):
        if results[i][0] < min_rmse:
            min_rmse = results[i][0]
            min_rmse_data = results[i]
    best_model = min_rmse_data[1]
    
    best_model_data = [min_rmse]
    for i in range(3,len(min_rmse_data)):
        best_model_data += corresponding_cols[min_rmse_data[i]]

    return descriptions, best_model_data, best_model, rmse_array

# run tests on both gated station entry and delay data
def main():
    target_name = sys.argv[1]
    target_path = model_dict[sys.argv[1]][0]
    target_model = model_dict[sys.argv[1]][1]
    descriptions = "GSE data:\n"
    current_desc,gse_model_data,gse_model,gse_rmse = run_tests("../../data/analysis_data/GSE_train_inputs.csv","../../data/analysis_data/GSE_test_inputs.csv",target_name,target_model,"gse")
    descriptions += current_desc+"Delay data:\n"
    current_desc,delay_model_data,delay_model,delay_rmse = run_tests("../../data/analysis_data/delay_train_inputs.csv","../../data/analysis_data/delay_test_inputs.csv",target_name,target_model,"delay")
    descriptions += current_desc

    if sys.argv[3] == 'NO_BOOTSTRAP':
        #output readable results
        with open(target_path+"_readable.txt","w") as f:
            print(descriptions,end='',file=f)
        
        #output gse model metadata
        with open(target_path+"_gse_model_data.txt","w") as f:
            for x in gse_model_data:
                print(x,end=' ',file=f)
        
        #output delay model metadata
        with open(target_path+"_delay_model_data.txt","w") as f:
            for x in delay_model_data:
                print(x,end=' ',file=f)
        
        #output gse model
        joblib.dump(gse_model,target_path+"_gse_model.txt")

        #output delay model
        joblib.dump(delay_model,target_path+"_delay_model.txt")

        print(f"{display_model_name(target_name)} done!")
    
    else:
        gse_rmse_no_ad = gse_rmse[0] 
        gse_rmse_ad = min(gse_rmse)
        gse_rmse_day_of_week = min(gse_rmse[5:9])
        gse_rmse_season = min(gse_rmse[9:13])
        gse_rmse_weather = min(gse_rmse[13:15])
        gse_improvement_ad = 100-100*gse_rmse_ad/gse_rmse_no_ad
        gse_improvement_day_of_week = 100-100*gse_rmse_day_of_week/gse_rmse_no_ad
        gse_improvement_season = 100-100*gse_rmse_season/gse_rmse_no_ad
        gse_improvement_weather = 100-100*gse_rmse_weather/gse_rmse_no_ad

        delay_rmse_no_ad = delay_rmse[0]
        delay_rmse_ad = min(delay_rmse)
        delay_rmse_day_of_week = min(delay_rmse[5:9])
        delay_rmse_season = min(delay_rmse[9:13])
        delay_rmse_weather = min(delay_rmse[13:15])
        delay_improvement_ad = 100-100*delay_rmse_ad/delay_rmse_no_ad
        delay_improvement_day_of_week = 100-100*delay_rmse_day_of_week/delay_rmse_no_ad
        delay_improvement_season = 100-100*delay_rmse_season/delay_rmse_no_ad
        delay_improvement_weather = 100-100*delay_rmse_weather/delay_rmse_no_ad

        print(gse_rmse_no_ad,gse_rmse_ad,gse_improvement_ad,gse_improvement_day_of_week,gse_improvement_season,gse_improvement_weather,
              delay_rmse_no_ad,delay_rmse_ad,delay_improvement_ad,delay_improvement_day_of_week,delay_improvement_season,delay_improvement_weather)

if __name__ == "__main__":
    main()

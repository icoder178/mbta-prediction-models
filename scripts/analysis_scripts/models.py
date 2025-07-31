# Tests 10 models on performance in predicting delay data and gated station entry data.
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
import random

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
        Lasso()
    ),
    "GradientBoost": (
        "../../data/intermediate_data/GradientBoost",
        GradientBoostingRegressor(random_state=0)
    ),
    "SupportVector": (
        "../../data/intermediate_data/SupportVector",
        SVR()
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

# read analysis data and convert into format required for ML training
def process_data(_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols):
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
    if sys.argv[4] != "NO_BOOTSTRAP":
        random.seed(sys.argv[4])
        selections = np.array([random.randint(0,len(input)-1) for _ in range(len(input))])
        selections.sort()
        input = input[selections]
        output = output[selections]
    return input,output

# split data into train and test sets
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
# return rmse, model
def try_running(_source,
                _input_rows,
                _input_cols,
                _output_cols,
                _input_len,
                _output_len,
                _split_prop,
                _base_model):
    raw_input,raw_output = process_data(_source,_input_rows,_input_len,_output_len,_input_cols,_output_cols)
    train_input,test_input,train_output,test_output = split_data(raw_input,raw_output,_split_prop)
    current_model = train_model(train_input,train_output,_base_model)
    rmse = test_model(test_input,test_output,current_model)
    return rmse,current_model

# run given an array of strings corresponding to inputs
# return rmse, model, output string
def run_by_name(_source,_inputs,_model_name,_base_model):
    current_source = _source
    current_model = _base_model
    for input_name in _inputs:
        if input_name not in corresponding_cols:
            raise ValueError("Data name does not exist.")
    description = f"Using model ({_model_name}), inputs of previous {int(sys.argv[2])}-day ("
    for i in range(len(_inputs)):
        if i > 0:
            description += ', '
        description += _inputs[i]
    description += f") to predict target metric"
    input_cols = []
    for input_name in _inputs:
        input_cols += corresponding_cols[input_name]
    rmse, result_model = try_running(
    _source = current_source,
    _input_rows = int(sys.argv[2]),
    _input_cols = input_cols,
    _output_cols = [1],
    _input_len = len(input_cols)*int(sys.argv[2]),
    _output_len = 1,
    _split_prop = float(sys.argv[3]),
    _base_model = current_model
    )
    rmse = float(rmse)
    return [rmse,result_model,f"{description}: RMSE is {rmse}."]


# run a series of tests with differing specs on the same model
# return .out file output, data on best model, best model
def run_tests(_source,_model_name,_model):
    results = []
    # test with no additional data
    results.append(run_by_name(_source,["target metric"],_model_name,copy.deepcopy(_model))
                   +["target metric"])

    # test with all additional data
    results.append(run_by_name(_source,["target metric","day of week","season","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week","season","weather"])
    results.append(run_by_name(_source,["target metric","day of week OHE","season OHE","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week OHE","season OHE","weather"])
    results.append(run_by_name(_source,["target metric","scaled day of week","scaled season","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week","scaled season","scaled weather"])
    results.append(run_by_name(_source,["target metric","scaled day of week OHE","scaled season OHE","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week OHE","scaled season OHE","scaled weather"])

    # test with additional day of week data
    results.append(run_by_name(_source,["target metric","day of week"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week"])
    results.append(run_by_name(_source,["target metric","day of week OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week OHE"])
    results.append(run_by_name(_source,["target metric","scaled day of week"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week"])
    results.append(run_by_name(_source,["target metric","scaled day of week OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week OHE"])

    # test with additional season data
    results.append(run_by_name(_source,["target metric","season"],_model_name,copy.deepcopy(_model))
                   +["target metric","season"])
    results.append(run_by_name(_source,["target metric","season OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","season OHE"])
    results.append(run_by_name(_source,["target metric","scaled season"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled season"])
    results.append(run_by_name(_source,["target metric","scaled season OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled season OHE"])

    # test with additional weather data
    results.append(run_by_name(_source,["target metric","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","weather"])
    results.append(run_by_name(_source,["target metric","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled weather"])

    # test with additional day of week, season data
    results.append(run_by_name(_source,["target metric","day of week","season"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week","season"])
    results.append(run_by_name(_source,["target metric","day of week OHE","season OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week OHE","season OHE"])
    results.append(run_by_name(_source,["target metric","scaled day of week","scaled season"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week","scaled season"])
    results.append(run_by_name(_source,["target metric","scaled day of week OHE","scaled season OHE"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week OHE","scaled season OHE"])

    # test with additional day of week, weather data
    results.append(run_by_name(_source,["target metric","day of week","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week","weather"])
    results.append(run_by_name(_source,["target metric","day of week OHE","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","day of week OHE","weather"])
    results.append(run_by_name(_source,["target metric","scaled day of week","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week","scaled weather"])
    results.append(run_by_name(_source,["target metric","scaled day of week OHE","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled day of week OHE","scaled weather"])
    
    # test with additional season, weather data
    results.append(run_by_name(_source,["target metric","season","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","season","weather"])
    results.append(run_by_name(_source,["target metric","season OHE","weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","season OHE","weather"])
    results.append(run_by_name(_source,["target metric","scaled season","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled season","scaled weather"])
    results.append(run_by_name(_source,["target metric","scaled season OHE","scaled weather"],_model_name,copy.deepcopy(_model))
                   +["target metric","scaled season OHE","scaled weather"])

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
    current_desc,gse_model_data,gse_model,gse_rmse = run_tests("../../data/analysis_data/GSE_inputs.csv",target_name,target_model)
    descriptions += current_desc+"Delay data:\n"
    current_desc,delay_model_data,delay_model,delay_rmse = run_tests("../../data/analysis_data/delay_inputs.csv",target_name,target_model)
    descriptions += current_desc

    if sys.argv[4] == 'NO_BOOTSTRAP':
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

        print(f"{target_name} done!")
    
    else:
        gse_rmse_no_ad = gse_rmse[0] 
        gse_rmse_ad = min(gse_rmse)
        gse_rmse_day_of_week = min(gse_rmse[1:5])
        gse_rmse_season = min(gse_rmse[5:9])
        gse_rmse_weather = min(gse_rmse[9:10])
        gse_improvement_ad = 100-100*gse_rmse_ad/gse_rmse_no_ad
        gse_improvement_day_of_week = 100-100*gse_rmse_day_of_week/gse_rmse_no_ad
        gse_improvement_season = 100-100*gse_rmse_season/gse_rmse_no_ad
        gse_improvement_weather = 100-100*gse_rmse_weather/gse_rmse_no_ad

        delay_rmse_no_ad = delay_rmse[0]
        delay_rmse_ad = min(delay_rmse)
        delay_rmse_day_of_week = min(gse_rmse[1:5])
        delay_rmse_season = min(gse_rmse[5:9])
        delay_rmse_weather = min(gse_rmse[9:10])
        delay_improvement_ad = 100-100*delay_rmse_ad/delay_rmse_no_ad
        delay_improvement_day_of_week = 100-100*delay_rmse_day_of_week/delay_rmse_no_ad
        delay_improvement_season = 100-100*delay_rmse_season/delay_rmse_no_ad
        delay_improvement_weather = 100-100*delay_rmse_weather/delay_rmse_no_ad

        print(gse_rmse_no_ad,gse_rmse_ad,gse_improvement_ad,gse_improvement_day_of_week,gse_improvement_season,gse_improvement_weather,
              delay_rmse_no_ad,delay_rmse_ad,delay_improvement_ad,delay_improvement_day_of_week,delay_improvement_season,delay_improvement_weather)

if __name__ == "__main__":
    main()
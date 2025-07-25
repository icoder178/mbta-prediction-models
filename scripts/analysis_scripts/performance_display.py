import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# converts a human-readable line of the input data into the RMSE value it contains
def line_to_num(line):
    line = line[:-2]
    position = len(line)-1
    while line[position] != ' ':
        position -= 1
    line = line[position+1:]
    return float(line)

# reads 1 set of human-readable values and returns tuple of RMSE given no additional data and minimum RMSE given additional data
def read_set(source):
    source.readline()
    nums = [line_to_num(source.readline()) for _ in range(5)]
    return [nums[0],min(nums[1],nums[2],nums[3],nums[4])]

# reads data for full file
def read_source(filepath):
    with open(filepath) as source:
        return read_set(source),read_set(source)

# returns data, sorted ascdending by total of RMSE on two metrics
def find_data():
    file_list = [
    ["../../output/data_appendix_output/RandomForest.out","RandomForest"],
    ["../../output/data_appendix_output/Linear.out","Linear"],
    ["../../output/data_appendix_output/Ridge.out","Ridge"],
    ["../../output/data_appendix_output/Lasso.out","Lasso"],
    ["../../output/data_appendix_output/GradientBoost.out","GradientBoost"],
    ["../../output/data_appendix_output/SupportVector.out","SupportVector"],
    ["../../output/data_appendix_output/MultilayerPerceptron.out","MultilayerPerceptron"],
    ["../../output/data_appendix_output/kNearestNeighbor.out","kNearestNeighbor"],
    ["../../output/data_appendix_output/MovingAverage.out","MovingAverage"],
    ["../../output/data_appendix_output/LSTM.out","LSTM"],
    ["../../output/data_appendix_output/Poisson.out","Poisson"]]
    gse_array = []
    delay_array = []
    for file in file_list:
        filepath = file[0]
        filename = file[1]
        result = read_source(filepath)
        gse_array.append(((result[0][0]+result[0][1]),(filename,result[0])))
        delay_array.append(((result[1][0]+result[1][1]),(filename,result[1])))
    gse_array.sort()
    delay_array.sort()
    gse_data = pd.DataFrame()
    gse_data['Data'] = ['No Additional Data','Additional Data']
    gse_data = gse_data.set_index('Data')
    for x in gse_array:
        gse_data[x[1][0]] = x[1][1]
    delay_data = pd.DataFrame()
    delay_data['Data'] = ['No Additional Data','Additional Data']
    delay_data = delay_data.set_index('Data')
    for x in delay_array:
        delay_data[x[1][0]] = x[1][1]
    gse_data = gse_data.transpose()
    delay_data = delay_data.transpose()
    return gse_data,delay_data

# saves data recieved to bar plot
def display_data(data,title,xlabel,ylabel,top_n,name):
    _data = data.reset_index()
    _data.columns = ['Data','No Additional Data','Additional Data']
    _data = _data[:top_n]
    _data = _data.melt(id_vars='Data', var_name='Condition', value_name='Value')
    plt.figure(figsize=(14,8))
    sns.barplot(_data,x='Data',y='Value',hue='Condition',errorbar=None)
    sns.despine()
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(f"../../output/results/{name}.png")

# run on both gated station entry data and delay data
def main():
    gse_data,delay_data = find_data()
    print("GSE data:")
    print(gse_data)
    print("\nDelay data:")
    print(delay_data)
    display_data(gse_data,"Top 8 Model Performance on Gated Station Entry Data","Model Name","Model RMSE",8,"Delay_data")
    display_data(delay_data,"Top 8 Model Performance on MBTA Delay Data","Model Name","Model RMSE",8,"GSE_data")

main()
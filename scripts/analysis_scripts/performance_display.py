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

# reads 1 set of human-readable values and returns tuple (in order) of:
# RMSE given no additional data
# minimum RMSE given any type of data
# minimum RMSE given day of week data
# minimum RMSE given season data
# minimum RMSE given weather data
def read_set(source):
    source.readline()
    nums = [line_to_num(source.readline()) for _ in range(27)]
    return min(nums[0:27]),nums[0],min(nums[5:9]),min(nums[9:13]),min(nums[13:15])

# reads data for full file
def read_source(filepath):
    with open(filepath) as source:
        return read_set(source),read_set(source)

# returns data, sorted ascending by total of RMSE on two metrics
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
    ["../../output/data_appendix_output/Poisson.out","Poisson"]]
    gse_array = []
    delay_array = []
    for file in file_list:
        filepath = file[0]
        filename = file[1]
        gse_result,delay_result = read_source(filepath)
        gse_array.append((gse_result[0],(filename,gse_result)))
        delay_array.append((delay_result[0],(filename,delay_result)))
    gse_array.sort()
    delay_array.sort()
    gse_data = pd.DataFrame()
    gse_data['Data'] = ['Any Data','No Additional Data','Day of Week Data','Season Data','Weather Data']
    gse_data = gse_data.set_index('Data')
    for x in gse_array:
        gse_data[x[1][0]] = x[1][1]
    delay_data = pd.DataFrame()
    delay_data['Data'] = ['Any Data','No Additional Data','Day of Week Data','Season Data','Weather Data']
    delay_data = delay_data.set_index('Data')
    for x in delay_array:
        delay_data[x[1][0]] = x[1][1]
    gse_data = gse_data.transpose()
    delay_data = delay_data.transpose()
    return gse_data,delay_data

# saves data recieved to bar plot
def display_data(data,title,xlabel,ylabel,name):
    _data = data.reset_index()
    _data.columns = ['Data','Any Data','No Additional Data']
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

# compute average percentage decrease in RMSE for models with additional data
def compute_improvement(data,name):
    _data = data.transpose()
    _data.iloc[[0,1]] = _data.iloc[[1,0]]
    _data = _data.rename({'No Additional Data':'Any Data', 'Any Data':'No Additional Data'})
    print(f"\n{name}")
    for i in range(1,len(_data)):
        percentage_value = 100-_data.iloc[i].mean()/_data.iloc[0].mean()*100
        print(f"Percentage improvement off {_data.index[0]} for {_data.index[i]}: {percentage_value:.1f}%.")

# run on both gated station entry data and delay data
def main():
    gse_data,delay_data = find_data()
    print("GSE data:")
    print(gse_data.to_string())
    print("\nDelay data:")
    print(delay_data.to_string())

    gse_data_plot = gse_data.drop(['Day of Week Data','Season Data','Weather Data'],axis=1)
    delay_data_plot = delay_data.drop(['Day of Week Data','Season Data','Weather Data'],axis=1)
    display_data(gse_data_plot,"Model Performance on Gated Station Entry Data","Model Name","Model RMSE","delay_data")
    display_data(delay_data_plot,"Model Performance on MBTA Delay Data","Model Name","Model RMSE","GSE_data")

    compute_improvement(gse_data,"Improvement on gated station entry data:")
    compute_improvement(delay_data,"Improvement on delay data:")

main()
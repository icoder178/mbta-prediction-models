import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

model_list = [
    "RandomForest",
    "Linear",
    "Ridge",
    "Lasso",
    "GradientBoost",
    "SupportVector",
    "MultilayerPerceptron"
    ,"kNearestNeighbor",
    "MovingAverage",
    "Poisson"
]

# saves data recieved to bar plot
def display_data(data,title,xlabel,ylabel,name):
    # compute order
    mean_values = {x:[0.0,0.0] for x in model_list}
    for i in range(len(data)):
        if data.iloc[i,2] == 'Any Data':
            mean_values[data.iloc[i,1]][0] += data.iloc[i,3]
            mean_values[data.iloc[i,1]][1] += 1
    mean_array = []
    for x in mean_values:
        mean_array.append([mean_values[x][1]/mean_values[x][0],x])
    mean_array.sort(reverse=True)
    order = []
    for x in mean_array:
        order.append(x[1])
    plt.figure(figsize=(14,8))
    sns.barplot(data,x='Model',y='RMSE',hue='Data',estimator=np.mean,errorbar=("ci",95),order=order,hue_order=['Any Data','No Additional Data'])
    sns.despine()
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(f"../../output/results/{name}_bootstrapped.png")

# run on both gated station entry data and delay data
def main():
    gse_data = pd.read_csv("../../data/intermediate_data/Bootstrap_gse.csv")
    delay_data = pd.read_csv("../../data/intermediate_data/Bootstrap_delay.csv")
    display_data(gse_data,"Model Performance on Gated Station Entry Data, Bootstrapped","Model Name","Model RMSE","gse_data")
    display_data(delay_data,"Model Performance on MBTA Delay Data, Bootstrapped","Model Name","Model RMSE","delay_data")

if __name__ == "__main__":
    main()
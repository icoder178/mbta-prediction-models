import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t

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
def display_ranking_data(data,title,xlabel,ylabel,name):
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

def display_improvement_data(data,name,title):
    plt.figure(figsize=(14,8))
    sns.barplot(data,x='Data',y='RMSE',estimator=np.mean,errorbar=("ci",95))
    sns.despine()
    plt.title(title, fontsize=24)
    plt.xlabel(f"Additional Data", fontsize=18)
    plt.ylabel(f"Percentage Decrease in RMSE", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"../../output/results/{name}_improvement.png")

    # compute CIs
    grouped_data = data.groupby('Data')['RMSE']
    summary = grouped_data.agg(['mean', 'count', 'std']).reset_index()
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])  # Standard error
    confidence = 0.95
    summary['t_crit'] = summary['count'].apply(lambda n: t.ppf((1 + confidence) / 2, df=n-1))
    summary['ci95'] = summary['t_crit'] * summary['sem']
    summary['ci_lower'] = summary['mean'] - summary['ci95']
    summary['ci_upper'] = summary['mean'] + summary['ci95']
    
    # output CIs
    for i in range(len(summary)):
        print(f"{summary['Data'].iloc[i]}: mean {summary['mean'].iloc[i]}, 95% CI [{summary['ci_lower'].iloc[i]}, {summary['ci_upper'].iloc[i]}]")

# run on both gated station entry data and delay data
def main():
    gse_data = pd.read_csv("../../data/intermediate_data/Bootstrap_gse.csv")
    delay_data = pd.read_csv("../../data/intermediate_data/Bootstrap_delay.csv")

    gse_data_ranking = gse_data[((gse_data['Data'] == 'Any Data') | (gse_data['Data'] == 'No Additional Data'))]
    delay_data_ranking = delay_data[((delay_data['Data'] == 'Any Data') | (delay_data['Data'] == 'No Additional Data'))]
    display_ranking_data(gse_data_ranking,"Model Performance on Gated Station Entry Data","Model Name","Model RMSE","gse_data")
    display_ranking_data(delay_data_ranking,"Model Performance on MBTA Delay Data","Model Name","Model RMSE","delay_data")

    gse_data_improvement = gse_data[((gse_data['Data'] != 'Any Data') & (gse_data['Data'] != 'No Additional Data'))]
    delay_data_improvement = delay_data[((delay_data['Data'] != 'Any Data') & (delay_data['Data'] != 'No Additional Data'))]
    print("GSE data:")
    display_improvement_data(gse_data_improvement,"gse_data","GSE Model Improvement Given Additional Data")
    print("Delay data:")
    display_improvement_data(delay_data_improvement,"delay_data","Delay Model Improvement Given Additional Data")

if __name__ == "__main__":
    main()
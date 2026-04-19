# Combines per-model hyperparameter tuning artifacts.
import pandas as pd

model_list = [
    "RandomForest",
    "Linear",
    "Ridge",
    "Lasso",
    "GradientBoost",
    "SupportVector",
    "MultilayerPerceptron",
    "kNearestNeighbor",
    "MovingAverage",
    "Poisson"
]

# combines files with a shared prefix
def combine_files(_prefix,_output):
    data = []
    for model in model_list:
        data.append(pd.read_csv(f"../../data/intermediate_data/{_prefix}_{model}.csv"))
    pd.concat(data,ignore_index=True).to_csv(f"../../data/intermediate_data/{_output}.csv",index=False)

def main():
    combine_files("hyperparameter_tuning_results","hyperparameter_tuning_results")
    combine_files("selected_hyperparameters","selected_hyperparameters")
    print("hyperparameter tuning results combined!")

if __name__ == "__main__":
    main()

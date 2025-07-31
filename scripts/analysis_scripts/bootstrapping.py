import subprocess
import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

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

def run_model(index, model, seed):
    result = subprocess.run([f"python -W ignore models.py {model} 5 0.8 {seed}"],shell=True,capture_output=True,text=True)
    values = result.stdout.strip().split()
    return index, values

def compute_raw_results():
    run_count = 1000
    worker_count = 16
    check_count = 5
    results = np.zeros((run_count, 12))
    start_time = time.perf_counter()
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(run_model,i,model_list[i%len(model_list)],i/len(model_list)):i for i in range(run_count)}
        cnt = 0
        for future in as_completed(futures):
            index, values = future.result()
            results[index] = values
            cnt += 1
            if cnt%check_count == 0:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"{cnt}/{run_count} processes done! Time elapsed so far: {round(elapsed_time)}s.")
    return results

def write_to_file(data):
    names = [
        "No Additional Data",
        "Any Data",
        "Any Data Improvement",
        "Day of Week Data Improvement",
        "Season Data Improvement",
        "Weather Data Improvement"
    ]
    gse_data = []
    delay_data = []
    for i in range(len(data)):
        for j in range(len(names)):
            gse_data.append([model_list[i%10],names[j],data[i][j]])
            delay_data.append([model_list[i%10],names[j],data[i][j+len(names)]])
    gse_data = pd.DataFrame(gse_data,columns=["Model","Data","RMSE"])
    delay_data = pd.DataFrame(delay_data,columns=["Model","Data","RMSE"])
    gse_data.to_csv("../../data/intermediate_data/Bootstrap_gse.csv")
    delay_data.to_csv("../../data/intermediate_data/Bootstrap_delay.csv")

if __name__ == "__main__":
    data = compute_raw_results()
    write_to_file(data)
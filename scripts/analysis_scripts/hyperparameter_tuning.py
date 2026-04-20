# Tunes model hyperparameters on inner-train/validation data.
import pandas as pd
import numpy as np
import copy
import sys
import os
import hashlib
import models

candidate_count = 32

task_sources = {
    "gse": (
        "../../data/analysis_data/GSE_inner_train_inputs.csv",
        "../../data/analysis_data/GSE_validation_inputs.csv"
    ),
    "delay": (
        "../../data/analysis_data/delay_inner_train_inputs.csv",
        "../../data/analysis_data/delay_validation_inputs.csv"
    )
}

tuned_models = {
    "RandomForest",
    "Ridge",
    "Lasso",
    "GradientBoost",
    "SupportVector",
    "MultilayerPerceptron",
    "kNearestNeighbor",
    "Poisson"
}

processed_data = {}

# returns stable deterministic seed
def stable_seed(*parts):
    seed_text = "|".join([str(part) for part in parts])
    return int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16],16)%(2**32)

# samples a log-uniform floating point value
def loguniform(rng,low,high):
    return float(np.exp(rng.uniform(np.log(low),np.log(high))))

# samples a rounded log-uniform integer
def rounded_loguniform(rng,low,high):
    return max(1,int(round(loguniform(rng,low,high))))

# samples shared tree split parameters
def sample_tree_split_params(rng):
    return {
        "min_samples_split": 2 if rng.random() < 0.5 else int(rng.integers(3,51)),
        "min_samples_leaf": 1 if rng.random() < 0.5 else int(rng.integers(2,31))
    }

# samples shared max_features parameter
def sample_max_features(rng):
    sample = rng.random()
    if sample < 0.25:
        return "sqrt"
    if sample < 0.50:
        return "log2"
    if sample < 0.75:
        return 1.0
    return float(rng.uniform(0.3,0.8))

# samples model-specific hyperparameters
def sample_hyperparameters(_model_name,rng):
    if _model_name == "RandomForest":
        params = sample_tree_split_params(rng)
        params.update({
            "n_estimators": rounded_loguniform(rng,10,1000),
            "max_depth": None if rng.random() < 0.5 else rounded_loguniform(rng,5,80),
            "max_features": sample_max_features(rng),
            "bootstrap": bool(rng.random() < 0.75),
            "max_samples": None
        })
        return params
    if _model_name == "Ridge":
        return {"alpha": loguniform(rng,1e-4,1e4)}
    if _model_name == "Lasso":
        return {
            "alpha": loguniform(rng,1e-4,1e4),
            "selection": rng.choice(["cyclic","random"]).item()
        }
    if _model_name == "GradientBoost":
        params = sample_tree_split_params(rng)
        params.update({
            "n_estimators": rounded_loguniform(rng,10,1000),
            "learning_rate": loguniform(rng,0.01,1),
            "max_depth": int(rng.choice([1,2,3,4,5]).item()),
            "subsample": 1.0 if rng.random() < 0.5 else float(rng.uniform(0.5,1.0)),
            "max_features": sample_max_features(rng),
            "loss": rng.choice(["squared_error","huber"]).item()
        })
        return params
    if _model_name == "SupportVector":
        kernel_sample = rng.random()
        if kernel_sample < 0.70:
            kernel = "rbf"
        elif kernel_sample < 0.90:
            kernel = "linear"
        else:
            kernel = "poly"
        return {
            "kernel": kernel,
            "C": loguniform(rng,1e-2,1e2),
            "epsilon": loguniform(rng,1e-3,1e1),
            "gamma": rng.choice(["scale","auto"]).item(),
            "degree": int(rng.choice([2,3,4]).item())
        }
    if _model_name == "MultilayerPerceptron":
        if rng.random() < 2/3:
            hidden_layer_sizes = (rounded_loguniform(rng,25,400),)
        else:
            hidden_layer_sizes = (rounded_loguniform(rng,25,400),rounded_loguniform(rng,25,400))
        batch_size = "auto" if rng.random() < 0.25 else rounded_loguniform(rng,16,128)
        return {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": rng.choice(["relu","tanh"]).item(),
            "solver": "adam",
            "alpha": loguniform(rng,1e-6,1e-2),
            "learning_rate_init": loguniform(rng,1e-5,1e-1),
            "learning_rate": rng.choice(["constant","adaptive"]).item(),
            "batch_size": batch_size,
            "max_iter": 100,
            "early_stopping": False
        }
    if _model_name == "kNearestNeighbor":
        return {
            "n_neighbors": rounded_loguniform(rng,1,100),
            "weights": rng.choice(["uniform","distance"]).item(),
            "p": int(rng.choice([1,2]).item())
        }
    if _model_name == "Poisson":
        return {"alpha": loguniform(rng,1e-4,1e4)}
    return {}

# reads and processes data, with cache
def processed_single_data(_source,_input_rows,_input_cols):
    cache_key = (_source,_input_rows,tuple(_input_cols))
    if cache_key not in processed_data:
        processed_data[cache_key] = models.process_single_data(
            _source,
            _input_rows,
            len(_input_cols)*_input_rows,
            1,
            _input_cols,
            [1]
        )
    return processed_data[cache_key]

# evaluates one parameter candidate
def evaluate_candidate(_model_name,_base_model,_params,_train_input,_train_output,_validation_input,_validation_output):
    current_model = copy.deepcopy(_base_model)
    if len(_params) > 0:
        current_model.set_params(**_params)
    current_model = models.train_model(_train_input,_train_output,current_model)
    return float(models.test_model(_validation_input,_validation_output,current_model))

# tunes one feature set
def tune_feature_set(_model_name,_base_model,_task_name,_train_source,_validation_source,_input_rows,_inputs):
    input_cols = []
    for input_name in _inputs:
        input_cols += models.corresponding_cols[input_name]
    train_input,train_output = processed_single_data(_train_source,_input_rows,input_cols)
    validation_input,validation_output = processed_single_data(_validation_source,_input_rows,input_cols)
    feature_name = models.feature_set_name(_inputs)
    count = candidate_count if _model_name in tuned_models else 1
    rows = []
    best_rmse = float("inf")
    best_row = None
    for i in range(count):
        current_seed = stable_seed(_model_name,_task_name,feature_name,i)
        rng = np.random.default_rng(current_seed)
        params = sample_hyperparameters(_model_name,rng)
        if count > 1:
            print(f"Tuning {_model_name}, {_task_name}, {feature_name}, candidate {i+1}/{count}",flush=True)
        error = ""
        try:
            rmse = evaluate_candidate(_model_name,_base_model,params,train_input,train_output,validation_input,validation_output)
        except Exception as e:
            rmse = float("inf")
            error = repr(e)
        row = {
            "task": _task_name,
            "model": _model_name,
            "feature_set": feature_name,
            "candidate_index": i,
            "validation_rmse": rmse,
            "params": repr(params),
            "random_seed": current_seed,
            "error": error
        }
        rows.append(row)
        if rmse < best_rmse:
            best_rmse = rmse
            best_row = row
    if best_row is None or not np.isfinite(best_row["validation_rmse"]):
        raise ValueError(f"No valid hyperparameter candidate for {_model_name}, {_task_name}, {feature_name}.")
    selected_row = copy.deepcopy(best_row)
    return rows,selected_row

# tunes all feature sets for one model
def main():
    target_name = sys.argv[1]
    input_rows = int(sys.argv[2])
    target_model = models.model_dict[target_name][1]
    result_rows = []
    selected_rows = []
    for task_name in task_sources:
        train_source,validation_source = task_sources[task_name]
        for i in range(len(models.input_sets)):
            current_inputs = models.input_sets[i]
            print(f"Tuning {target_name}, {task_name}, feature set {i+1}/{len(models.input_sets)}: {models.feature_set_name(current_inputs)}",flush=True)
            current_rows,current_selected = tune_feature_set(
                target_name,
                target_model,
                task_name,
                train_source,
                validation_source,
                input_rows,
                current_inputs
            )
            for row in current_rows:
                row["selected"] = row["candidate_index"] == current_selected["candidate_index"]
                result_rows.append(row)
            selected_rows.append(current_selected)
    os.makedirs("../../data/intermediate_data",exist_ok=True)
    pd.DataFrame(result_rows).to_csv(f"../../data/intermediate_data/hyperparameter_tuning_results_{target_name}.csv",index=False)
    pd.DataFrame(selected_rows).to_csv(f"../../data/intermediate_data/selected_hyperparameters_{target_name}.csv",index=False)
    print(f"{target_name} tuning done!",flush=True)

if __name__ == "__main__":
    main()

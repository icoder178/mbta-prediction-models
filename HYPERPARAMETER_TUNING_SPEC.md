# Hyperparameter Tuning Implementation Specification

This document defines the intended hyperparameter tuning behavior for the MBTA prediction model comparison project. It is intended to be sufficient as an implementation reference. No code changes are made by this document.

## Goals

- Add hyperparameter tuning while preserving the project goal of fair multi-model comparison.
- Keep the final held-out test data out of hyperparameter selection.
- Use equal random-search budgets where models have tunable hyperparameters.
- Preserve the existing model-comparison outputs as much as possible.
- Reduce bootstrap cost relative to the previous 100-bootstrap-per-model design.

## Non-Goals

- Do not tune the global lookback window.
- Do not tune the raw/processed data generation logic beyond the train/validation/test split behavior described here.
- Do not use the final test set to select hyperparameters.
- Do not make SHAP depend on whichever model happens to win overall.

## Data Splits

The current train/test split should be extended into three chronological partitions:

- `70%` inner train
- `10%` validation
- `20%` test

The split must be chronological by date. No random shuffling should be used for these primary partitions.

The existing final test portion remains the final held-out evaluation set. The new validation portion is used only for hyperparameter selection.

Recommended implementation behavior:

- For each task, first build the cleaned task-specific daily dataset as currently done.
- Split by time into `inner_train`, `validation`, and `test`.
- Fit scaling on `inner_train` only for hyperparameter validation if scaled features are regenerated at that stage.
- For final non-bootstrap training, train on `inner_train + validation` and evaluate once on `test`.

If the implementation keeps the current preprocessing behavior that fits scalers on the full 80% train partition, that is acceptable for final training, but validation experiments should not use statistics from validation to transform inner-train. The cleanest implementation is to create all three partitions before scaling.

## Feature-Set Tuning Units

Each model should be tuned independently for each feature set and each prediction task.

The hyperparameter key should be:

```text
task + model_name + feature_set_name
```

where `task` is one of:

- `gse`
- `delay`

and `feature_set_name` is one of the existing feature-set combinations evaluated by `models.py`, including the target-only baseline.

This means the target-only configuration also receives a tuned/default hyperparameter entry. Models without meaningful hyperparameters should still have an explicit default entry for consistency.

## Tuning Procedure

For each `task + model + feature_set` unit:

1. Build supervised lagged examples from the `inner_train` partition.
2. Build supervised lagged examples from the `validation` partition.
3. Sample `32` hyperparameter candidates from the model-specific search distribution.
4. Train each candidate on inner-train examples.
5. Evaluate each candidate on validation examples using RMSE.
6. Select the candidate with the lowest validation RMSE.
7. Store the selected hyperparameters for that `task + model + feature_set`.

Tie-breaking:

- If two candidates have identical validation RMSE, choose the earlier sampled candidate.
- Candidate generation must be deterministic given a fixed random seed.

Models that should not receive random hyperparameter tuning:

- `LinearRegression`: use sklearn defaults.
- `MovingAverage`: use existing behavior and do not tune.

These models should still produce results for every feature set under their default behavior.

## Non-Bootstrap Final Evaluation

After hyperparameter selection:

1. Combine the original `inner_train` and `validation` partitions into the final training partition.
2. For each `task + model + feature_set`, train the model on the combined final training partition using the selected hyperparameters for that unit.
3. Evaluate on the held-out test partition.
4. Generate the same style of summaries and plots as the current non-bootstrap path.

The final test set must be used only for final reporting, not for hyperparameter selection.

If the non-bootstrap run displays bootstrap summaries from existing precomputed bootstrap CSVs, that behavior may be preserved. It must not be described as newly computed bootstrap output unless bootstrap computation was explicitly requested.

## Bootstrap Evaluation

Bootstrap behavior should use the selected hyperparameters from the tuning phase. Bootstrap should not retune hyperparameters inside each bootstrap replicate.

Number of bootstrap replicates:

- Use `32` bootstrap replicates per model, reduced from the previous `100` per model.

For each bootstrap replicate:

1. Sample a bootstrap training set with replacement from the original `inner_train + validation` supervised examples.
2. Sample a bootstrap test set with replacement from the original held-out test supervised examples.
3. For each model and feature set, train using the selected hyperparameters for that `task + model + feature_set`.
4. Evaluate on the bootstrap test sample.
5. Aggregate results into the same categories as before:
   - target-only baseline
   - `Best Added-Feature Set`
   - day-of-week feature group
   - season feature group
   - weather feature group
   - percentage improvements relative to target-only baseline

Bootstrap samples for train and test must be drawn separately. Do not draw from a combined train/test pool.

## SHAP Behavior

SHAP values should be generated for the `RandomForestRegressor`, not for the overall best model.

Rationale:

- Tree-based SHAP support is reliable for random forests.
- The current best model can be an estimator that SHAP cannot explain directly, such as `MLPRegressor`.
- Using RandomForest makes the SHAP output consistently available and comparable across runs.

Required behavior:

- For each task, identify the relevant RandomForest model artifact.
- Generate SHAP values for RandomForest using the selected RandomForest hyperparameters.
- Use the same output naming style as the current SHAP/feature-importance output, unless the implementation needs to distinguish RandomForest SHAP from best-model diagnostics.

Recommended output naming if changing names:

- `delay_randomforest_importance.png`
- `gse_randomforest_importance.png`

If keeping current filenames:

- Document clearly that `delay_predictor_importance.png` and `gse_predictor_importance.png` are RandomForest SHAP plots, not necessarily plots for the overall best predictor.

## Search Distributions

All distributions below are sampled independently unless otherwise specified.

When a parameter is marked `default`, it should be left unset so sklearn uses the estimator default, unless the implementation needs to set it explicitly for reproducibility.

Notation:

- `choice([...])` means a uniform categorical choice.
- `randint(a, b)` means a uniform integer from `a` through `b`, inclusive.
- `uniform(a, b)` means a continuous uniform sample from `a` to `b`.
- Weighted categorical distributions are written as `{value: probability}`.

When a value means "all features" for `max_features`, use `1.0`, not integer `1`. In sklearn, integer `1` can mean one feature, while float `1.0` means all features.

### General

- Do not tune the lookback window.
- Use log-uniform distributions around defaults for many positive scale parameters.
- Use the same number of random candidates per tunable model-feature-set unit.

### RandomForestRegressor

Tune these parameters:

- `n_estimators`: sample `round(loguniform(10, 1000))`.
- `max_depth`: `{None: 0.5, round(loguniform(5, 80)): 0.5}`.
- `min_samples_split`: `{2: 0.5, randint(3, 50): 0.5}`.
- `min_samples_leaf`: `{1: 0.5, randint(2, 30): 0.5}`.
- `max_features`: `{"sqrt": 0.25, "log2": 0.25, 1.0: 0.25, uniform(0.3, 0.8): 0.25}`.
- `bootstrap`: `{True: 0.75, False: 0.25}`.

Fixed parameters:

- `max_samples`: `None`.

Notes:

- If `bootstrap=False` and `max_features=1.0`, trees may be nearly identical. This candidate is allowed but should not dominate the search because `bootstrap=False` is only sampled 25% of the time.

### LinearRegression

Do not run random hyperparameter tuning.

Use sklearn defaults:

- `fit_intercept=True`
- `positive=False`

### Ridge

Tune these parameters:

- `alpha`: sample `loguniform(1e-4, 1e4)`.

Fixed/default parameters:

- `fit_intercept`: default.
- `solver`: default.
- `positive`: `False`.

### Lasso

Tune these parameters:

- `alpha`: sample `loguniform(1e-4, 1e4)`.
- `selection`: `choice(["cyclic", "random"])`.

Fixed/default parameters:

- `max_iter`: default.
- `tol`: default.
- `fit_intercept`: default.
- `positive`: default.

Notes:

- `alpha` uses the same range as Ridge for search-budget fairness, even though L1 and L2 penalties are not numerically comparable.
- If convergence warnings are frequent, a later implementation may raise `max_iter`, but that should be treated as a solver-stability change rather than a model-selection hyperparameter.

### GradientBoostingRegressor

Tune these parameters:

- `n_estimators`: same sampling as RandomForest, `round(loguniform(10, 1000))`.
- `learning_rate`: sample `loguniform(0.01, 1)`.
- `max_depth`: `choice([1, 2, 3, 4, 5])`.
- `min_samples_split`: same sampling as RandomForest.
- `min_samples_leaf`: same sampling as RandomForest.
- `subsample`: `{1.0: 0.5, uniform(0.5, 1.0): 0.5}`.
- `max_features`: same sampling as RandomForest.
- `loss`: `choice(["squared_error", "huber"])`.

Important sklearn spelling:

- Use `"squared_error"`, not `"squared-error"`.

### SVR

The model is called `SupportVector` in this repo, but the sklearn estimator is `SVR`.

Tune these parameters:

- `kernel`: `{"rbf": 0.70, "linear": 0.20, "poly": 0.10}`.
- `C`: sample `loguniform(1e-2, 1e2)`.
- `epsilon`: sample `loguniform(1e-3, 1e1)`.
- `gamma`: `choice(["scale", "auto"])`.
- `degree`: `choice([2, 3, 4])`.

Fixed/default parameters:

- `tol`: default.
- `cache_size`: default.

Notes:

- `degree` matters only when `kernel="poly"`.
- `gamma` matters only for non-linear kernels.
- Do not sample numeric gamma values.

### MLPRegressor

Tune these parameters:

- `hidden_layer_sizes`: `{(H,): 2/3, (H1, H2): 1/3}`, where `H`, `H1`, and `H2` are independent `round(loguniform(25, 400))` samples.
- `activation`: `choice(["relu", "tanh"])`.
- `alpha`: sample `loguniform(1e-6, 1e-2)`.
- `learning_rate_init`: sample `loguniform(1e-5, 1e-1)`.
- `learning_rate`: `choice(["constant", "adaptive"])`.
- `batch_size`: `{"auto": 0.25, round(loguniform(16, 128)): 0.75}`.

Fixed parameters:

- `solver`: `"adam"`.
- `max_iter`: `100`.
- `early_stopping`: `False`.
- `n_iter_no_change`: default.
- `tol`: default.

Notes:

- `learning_rate` only affects the `"sgd"` solver in sklearn, so with `solver="adam"` it may have no practical effect. It can be left in the sampled configuration for consistency, but it should not be interpreted as a meaningful Adam hyperparameter.
- `max_iter=100` is intentionally below the sklearn default of `200` to control runtime. Convergence warnings are possible and should be recorded.

### KNeighborsRegressor

Tune these parameters:

- `n_neighbors`: set to `round(loguniform(1, 100))`, clipped to at least `1`.
- `weights`: `choice(["uniform", "distance"])`.
- `p`: `choice([1, 2])`.

Fixed/default parameters:

- `metric`: default.
- `algorithm`: default.
- `leaf_size`: default.

### MovingAverage

Do not run random hyperparameter tuning.

Use current behavior.

Notes:

- The only meaningful tuning knob is the global lookback window, and this specification explicitly does not tune it.

### PoissonRegressor

Tune these parameters:

- `alpha`: sample `loguniform(1e-4, 1e4)`.

Fixed/default parameters:

- `fit_intercept`: default.
- `max_iter`: default.
- `tol`: default.
- `solver`: default.

## Output Requirements

The implementation must preserve the existing output contract. Existing output files should still be produced at the same paths, with updated contents reflecting validation-selected hyperparameters and the revised bootstrap behavior. New tuning artifacts may be added, but they should not replace the current files.

Existing analysis-data outputs that must remain:

- `data/analysis_data/GSE_train_inputs.csv`
- `data/analysis_data/GSE_test_inputs.csv`
- `data/analysis_data/delay_train_inputs.csv`
- `data/analysis_data/delay_test_inputs.csv`

The train files should represent the final `inner_train + validation` training partition used for final model fitting. The test files should represent the held-out `20%` test partition. Validation-only data may be stored in additional files or generated internally, but preserving these four files keeps the current downstream script contract intact.

Existing intermediate model outputs that must remain:

- `data/intermediate_data/{Model}_readable.txt`
- `data/intermediate_data/{Model}_gse_model_data.txt`
- `data/intermediate_data/{Model}_delay_model_data.txt`
- `data/intermediate_data/{Model}_gse_model.txt`
- `data/intermediate_data/{Model}_delay_model.txt`

`{Model}` refers to each current model prefix: `RandomForest`, `Linear`, `Ridge`, `Lasso`, `GradientBoost`, `SupportVector`, `MultilayerPerceptron`, `kNearestNeighbor`, `MovingAverage`, and `Poisson`.

Existing selected-model outputs that must remain:

- `output/data_appendix_output/delay_model.txt`
- `output/data_appendix_output/delay_model_data.txt`
- `output/data_appendix_output/gse_model.txt`
- `output/data_appendix_output/gse_model_data.txt`

Existing results outputs that must remain:

- `output/results/performance_summary.txt`
- `output/results/predictor_summary.txt`
- `output/results/delay_data_original.png`
- `output/results/gse_data_original.png`
- `output/results/delay_predictor_residuals_histplot.png`
- `output/results/delay_predictor_residuals_ecdf.png`
- `output/results/gse_predictor_residuals_histplot.png`
- `output/results/gse_predictor_residuals_ecdf.png`
- `output/results/delay_predictor_importance.png`
- `output/results/gse_predictor_importance.png`

Existing bootstrap outputs that must remain when bootstrap display data is available:

- `data/intermediate_data/Bootstrap_delay.csv`
- `data/intermediate_data/Bootstrap_gse.csv`
- `output/results/bootstrap_summary.txt`
- `output/results/delay_data_bootstrapped.png`
- `output/results/gse_data_bootstrapped.png`
- `output/results/delay_data_improvement.png`
- `output/results/gse_data_improvement.png`

Required normal-run outputs:

- model performance summaries
- model ranking plots
- selected model artifacts
- residual diagnostics for selected best models
- RandomForest SHAP feature-importance plots

Required bootstrap-run outputs:

- bootstrap summary text
- bootstrap ranking plots
- bootstrap improvement plots
- intermediate bootstrap CSVs

Required tuning outputs:

- selected hyperparameters for each `task + model + feature_set`
- validation RMSE for each selected hyperparameter set
- enough metadata to reproduce the selected candidates from random seeds

Suggested tuning artifact names:

- `data/intermediate_data/hyperparameter_tuning_results.csv`
- `data/intermediate_data/selected_hyperparameters.csv`

## Guardrails and Feasibility Checks

This design is feasible but has several implementation risks that should be handled explicitly:

- Test leakage risk: hyperparameters must be chosen using validation data only.
- Runtime risk: `32` candidates per task-model-feature-set can still be expensive because the existing code has many feature sets. The implementation should log progress clearly.
- SVR runtime risk: SVR can be slow on larger feature matrices. Progress logging should make this visible.
- MLP convergence risk: `max_iter=100` may produce convergence warnings. This is acceptable if recorded, but repeated failures should be reviewed.
- sklearn compatibility risk: `loss` must use `"squared_error"` for `GradientBoostingRegressor`.
- Parameter semantics risk: use float `1.0` for `max_features` when the intended meaning is all features.
- Bootstrap interpretation risk: bootstrap should use selected hyperparameters and should not retune inside each replicate.
- SHAP interpretation risk: RandomForest SHAP plots should be labeled or documented as RandomForest explanations, not necessarily explanations of the overall best model.

## Implementation Order

Recommended implementation sequence:

1. Refactor data processing to produce or expose `inner_train`, `validation`, and `test` partitions.
2. Add a hyperparameter sampler for each tunable model.
3. Add validation-based tuning for one task and one model.
4. Generalize tuning to all tasks, models, and feature sets.
5. Store selected hyperparameters and validation results.
6. Update final model training to use selected hyperparameters.
7. Update bootstrap to use selected hyperparameters with 32 replicates per model.
8. Update SHAP generation to use RandomForest.
9. Regenerate outputs and verify file naming.

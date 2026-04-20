# Linear Support Vector Regression Plan

This document plans a change from kernel-based support vector regression to Linear Support Vector Regression. It is intentionally a planning document only; no code behavior is changed by this file.

## Motivation

The current `SupportVector` tuning path uses sklearn `SVR`. With the current tuning design, this requires up to:

```text
2 tasks * 27 feature sets * 32 candidates = 1728 SVR fits
```

Even the first full-data candidate can take many minutes. This is expected for kernel SVR on thousands of rows because sklearn `SVR` is libsvm-based and can scale poorly with row count, especially for `rbf` and `poly` kernels.

The intended replacement is sklearn `LinearSVR`, referred to throughout user-facing text and documentation as **Linear Support Vector Regression**. This avoids confusion with nonlinear kernel SVR.

## Intended Model Change

Replace the current sklearn estimator:

```python
SVR()
```

with:

```python
LinearSVR(random_state=0)
```

The existing repo model prefix may remain:

```text
SupportVector
```

This preserves current output file paths such as:

```text
data/intermediate_data/SupportVector_readable.txt
data/intermediate_data/SupportVector_gse_model.txt
data/intermediate_data/SupportVector_delay_model.txt
```

However, descriptors should explicitly call the model **Linear Support Vector Regression** wherever practical, especially in docs, comments, and tuning summaries.

## Conceptual Difference

Kernel `SVR` can learn nonlinear functions through kernels such as `rbf` and `poly`. It is flexible but computationally expensive on this dataset.

Linear Support Vector Regression learns a linear function with an epsilon-insensitive support-vector loss. It is much closer computationally to other linear models in this project, but with a different loss and regularization objective.

This means the model comparison changes from:

```text
nonlinear kernel support-vector regression
```

to:

```text
linear support-vector regression
```

That should be documented because the two are not equivalent model classes.

## Hyperparameter Tuning Changes

Remove these kernel-SVR hyperparameters from tuning:

- `kernel`
- `gamma`
- `degree`

Tune these Linear Support Vector Regression hyperparameters:

- `C`: `loguniform(1e-2, 1e2)`
- `epsilon`: `loguniform(1e-3, 1e1)`

These are the same `C` and `epsilon` ranges previously specified for kernel `SVR` tuning. The intended change is to remove the nonlinear kernel-specific parameters, not to change the regularization or epsilon-insensitive loss search ranges.

Recommended fixed/default parameters:

- `loss`: default
- `tol`: default
- `max_iter`: default initially
- `fit_intercept`: default
- `dual`: default
- `random_state`: `0`

If convergence warnings are frequent, increasing `max_iter` should be treated as a solver-stability change, not as a model-selection hyperparameter.

## Expected Runtime Effect

Linear Support Vector Regression should be much more feasible than kernel `SVR` for full-data random search. It should still be slower than ordinary linear regression, Ridge, or Lasso, but it should not spend many minutes on the first target-only candidate.

The total tuning budget can remain:

```text
32 candidates per task + feature set
```

because the estimator is now intended to be scalable enough for the existing tuning framework.

## Output Compatibility

Existing output paths should remain unchanged:

- `data/intermediate_data/SupportVector_readable.txt`
- `data/intermediate_data/SupportVector_gse_model_data.txt`
- `data/intermediate_data/SupportVector_delay_model_data.txt`
- `data/intermediate_data/SupportVector_gse_model.txt`
- `data/intermediate_data/SupportVector_delay_model.txt`
- all existing performance, bootstrap, and selected-model outputs

The model name in plots/tables may remain `SupportVector` for compatibility, but documentation and console output should clarify that this now means Linear Support Vector Regression.

## Code Updates Needed

Expected implementation points:

- In `scripts/analysis_scripts/models.py`, import `LinearSVR` instead of `SVR`.
- In `scripts/analysis_scripts/models.py`, replace the `SupportVector` estimator with `LinearSVR(random_state=0)`.
- In `scripts/analysis_scripts/hyperparameter_tuning.py`, replace the `SupportVector` sampler with only `C` and `epsilon`.
- Remove `kernel`, `gamma`, and `degree` from saved `SupportVector` hyperparameter candidates.
- Update comments and user-facing strings where they currently imply kernel `SVR`.

## Validation Plan

Before a full run:

1. Run syntax checks.
2. Run a small SupportVector-only tuning smoke test with `candidate_count=2`.
3. Run the dedicated SupportVector tuning script and confirm it progresses past the first feature set quickly.
4. Inspect `selected_hyperparameters_SupportVector.csv` and confirm it contains only applicable Linear Support Vector Regression parameters.
5. Run the normal analysis script after confirming SupportVector tuning is feasible.

## Open Decision

The output model prefix can stay as `SupportVector` for compatibility, but the human-readable display name could be changed to `LinearSupportVector` or `Linear Support Vector`. Keeping the prefix avoids file compatibility churn; changing the displayed name reduces ambiguity.

Recommended compromise:

- Keep file prefix: `SupportVector`
- Use display/documentation name: `Linear Support Vector Regression`

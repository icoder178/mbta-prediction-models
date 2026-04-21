# Hyperparameter Tuning Ranges

This document specifies the hyperparameter search ranges used for the model comparison. Hyperparameter tuning used randomized search over the distributions listed below. Tuning was performed independently for each prediction task, model, and feature set. For each tuned combination, 100 candidate hyperparameter configurations were sampled.

## Sampling Notation

- `loguniform(a, b)` denotes a log-uniform sample on the interval `[a, b]`.
- `uniform(a, b)` denotes a continuous uniform sample on the interval `[a, b]`.
- `randint(a, b)` denotes a discrete uniform integer sample from `a` through `b`, inclusive.
- `round(loguniform(a, b))` denotes a log-uniform sample rounded to the nearest integer.

## Random Forest Regressor

Tuned hyperparameters:

- `n_estimators`: `round(loguniform(10, 1000))`
- `max_depth`: 50% chance of `None`; 50% chance of `round(loguniform(5, 80))`
- `min_samples_split`: 50% chance of `2`; 50% chance of `randint(3, 50)`
- `min_samples_leaf`: 50% chance of `1`; 50% chance of `randint(2, 30)`
- `max_features`: 25% chance of `"sqrt"`; 25% chance of `"log2"`; 25% chance of `1.0`; 25% chance of `uniform(0.3, 0.8)`
- `bootstrap`: 75% chance of `True`; 25% chance of `False`

Fixed hyperparameters:

- `max_samples`: `None`

## Linear Regression

No hyperparameters were tuned. The sklearn defaults were used.

## Ridge Regression

Tuned hyperparameters:

- `alpha`: `loguniform(1e-4, 1e4)`

All other hyperparameters used sklearn defaults, except `positive=False`.

## Lasso Regression

Tuned hyperparameters:

- `alpha`: `loguniform(1e-4, 1e4)`
- `selection`: 50% chance of `"cyclic"`; 50% chance of `"random"`

All other hyperparameters used sklearn defaults.

## Gradient Boosting Regressor

Tuned hyperparameters:

- `n_estimators`: `round(loguniform(10, 1000))`
- `learning_rate`: `loguniform(0.01, 1)`
- `max_depth`: `randint(1, 5)`
- `min_samples_split`: 50% chance of `2`; 50% chance of `randint(3, 50)`
- `min_samples_leaf`: 50% chance of `1`; 50% chance of `randint(2, 30)`
- `subsample`: 50% chance of `1.0`; 50% chance of `uniform(0.5, 1.0)`
- `max_features`: 25% chance of `"sqrt"`; 25% chance of `"log2"`; 25% chance of `1.0`; 25% chance of `uniform(0.3, 0.8)`
- `loss`: 50% chance of `"squared_error"`; 50% chance of `"huber"`

## Linear Support Vector Regression

Tuned hyperparameters:

- `C`: `loguniform(1e-2, 1e2)`
- `epsilon`: `loguniform(1e-3, 1e1)`

The estimator used was sklearn `LinearSVR`.

## Multilayer Perceptron Regressor

Tuned hyperparameters:

- `hidden_layer_sizes`: 2/3 chance of a one-hidden-layer configuration `(H,)`; 1/3 chance of a two-hidden-layer configuration `(H1, H2)`
- For one-hidden-layer configurations, `H = round(loguniform(25, 100))`
- For two-hidden-layer configurations, `H1 = round(loguniform(10, 50))` and `H2 = round(loguniform(10, 50))`
- `activation`: 50% chance of `"relu"`; 50% chance of `"tanh"`
- `alpha`: `loguniform(1e-6, 1e-2)`
- `learning_rate_init`: `loguniform(1e-5, 1e-1)`
- `learning_rate`: 50% chance of `"constant"`; 50% chance of `"adaptive"`

Fixed hyperparameters:

- `solver`: `"adam"`
- `batch_size`: `"auto"`
- `max_iter`: `50`
- `early_stopping`: `False`

## k-Nearest Neighbors Regressor

Tuned hyperparameters:

- `n_neighbors`: `round(loguniform(1, 100))`
- `weights`: 50% chance of `"uniform"`; 50% chance of `"distance"`
- `p`: 50% chance of `1`; 50% chance of `2`

All other hyperparameters used sklearn defaults.

## Moving Average

No hyperparameters were tuned.

## Poisson Regressor

Tuned hyperparameters:

- `alpha`: `loguniform(1e-4, 1e4)`

All other hyperparameters used sklearn defaults.

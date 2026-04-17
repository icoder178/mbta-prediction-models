### Instructions
* Implementation details should be described in the document following this.
* Generate a new MD document explicitly spelling out in a standard manner the information informally discussed here. That generated document alone should be sufficient to describe desired behavior.
* Also, check that this won't lead to significant errors and/or unjustifiable behaviors and/or unreasonably complicated implementations.
* Interrupt process and ask me questions if something is unclear. Do not assume.
### Workflow
1. Do a 70% inner train, 10% validation, 20% test split of the data, by time.
2. For each group of added features, tune an independent optimal hyperparameter via 32 tests.
3. We now have a set of hyperparameters for each set of added features + model.

Then, for no bootstrapping:
1. Simply train each model on the inner train + validation test sets, and test on the test set as usual.
2. Output bootstrapped results as we did before.

Then, for bootstrapping:
1. Generate the 32 bootstrapped datasets (note we lower this from 100 to save compute), bootstrapping train from a combination of the original inner train + validation, and test from a combination of the original test.
2. Then train on train w/ hyperparameters that we just found, test on test.
3. Output bootstrapped results as we did before.

Finally, SHAP values:
1. Generate the SHAP values on the RandomForest, rather than the best model. This guarantees we can find them. Then output as before.
### Hyperparam Ranges
### General
* Do not tune the lookback window.
* Observe that oftentimes we're sampling in a loguniform range around the default value.
### RandomForestRegressor
* ```n_estimators```: To save processing power, let X be uniformly distributed from 1 to 3. The tested value is $10^X$ rounded to the nearest integer.
* ```max_depth```: Half chance None, half chance uniformly spread across your other suggestions.
* ```min_samples_split```: Half chance 2, half chance uniformly spread across your other suggestions.
* ```min_samples_leaf```: Half chance 1, half chance uniformly spread across your other suggestions.
* ```max_features```: 25% chance "sqrt", "log2", 1; remaining 25% chance spread across your other suggestions.
* ```bootstrap```: 75% True, 25% False.
* ```max_samples```: None.
### LinearRegression
Just leave at defaults. Also don't do any complicated hyperparam tuning on this as a result.
### Ridge
* ```alpha```: Use your suggested values.
* ```fit_intercept```: Leave at default.
* ```solver```: Leave at default.
* ```positive```: Keep at False.
### Lasso
* ```alpha```: ```loguniform(1e-4,1e4)``` (same as Ridge)
* ```max_iter```: Leave as default.
* ```tol```: Leave as default.
* ```selection```: Half-half cyclic and random.
* ```fit_intercept```: Leave as default.
* ```positive```: Leave as default.
### GradientBoostingRegressor
* ```n_estimators```: Same setup as for RandomForest.
* ```learning_rate```: Let's do ```loguniform(0.01,1)```.
* ```max_depth```: Yes, uniform across 1,2,3,4,5 as you described works. Let's do that.
* ```min_samples_split```: Same setup as for RandomForest.
* ```min_samples_leaf```: Same setup as for RandomForest.
* ```subsample```: Half chance 1.0, half chance uniformly distributed from 0.5 to 1.0.
* ```max_features```: Same setup as for RandomForest.
* ```loss```: Half-half ```squared-error, huber```.
### SupportVectorRegressor
* ```kernel``` 70% rbf, 20% linear, 10% poly.
* ```C```: ```loguniform(1e-2,1e2)```
* ```epsilon```: ```loguniform(1e-3,1e1)```
* ```gamma```: Half-half scale, auto. ```loguniform(1e-5, 1e0)``` for numbers.
* ```degree```: Uniform across 2,3,4.
* ```tol```: Default.
* ```cache_size```: Default.
### MLPRegressor:
* ```hidden_layer_sizes```: (note: round all samples values to nearest integer) 2/3 chance single-layer, ```loguniform(25,400)```, 1/3 chance double-layer, both of ```loguniform(25,400)```.
* ```activation```: Half-half relu, tanh.
* ```solver```: Just use Adam.
* ```alpha```: ```loguniform(1e-6,1e-2)```
* ```learning_rate_init```: ```loguniform (1e-5,1e-1)```
* ```learning_rate```: Half-half constant, adaptive.
* ```batch_size```: 25% "auto", otherwise round to closest integer of ```loguniform(16,128)```.
* ```max_iter```: MLP is already kind of slow. Let's do 100.
* ```early_stopping```: False
* ```n_iter_no_change```: Default.
* ```tol```: Default.
### KNeighborsRegressor:
* ```n_neighbors```: Nearest integer, rounding from a ```loguniform(1,100)```.
* ```weights```: Half-half uniform, distance.
* ```p```: Half-half 1,2.
* ```metric```: Default.
* ```algorithm```: Default.
* ```leaf_size```: Default.
### MovingAverage:
Don't do any hyperparam tuning.
### PoissonRegressor:
* ```alpha```: ```loguniform(1e-4,1e4)```
* ```fit_intercept```: Default.
* ```max_iter```: Default.
* ```tol```: Default.
* ```solver```: Default.

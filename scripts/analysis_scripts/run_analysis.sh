#!/bin/bash
set -e
# script for running just analysis scripts; run in the directory it is in.
pids=()

cleanup() {
  if [ ${#pids[@]} -gt 0 ]; then
    kill "${pids[@]}" 2> /dev/null || true
    wait "${pids[@]}" 2> /dev/null || true
  fi
  exit 130
}

wait_for_pids() {
  for pid in "${pids[@]}"
  do
    if ! wait "$pid"; then
      kill "${pids[@]}" 2> /dev/null || true
      wait "${pids[@]}" 2> /dev/null || true
      exit 1
    fi
  done
  pids=()
}

trap cleanup INT TERM

echo "Starting analysis; hyperparameter tuning and model training will occupy significant computational resources"
# edit if models change
models=("RandomForest" "Linear" "Ridge" "Lasso" "GradientBoost" "SupportVector" "MultilayerPerceptron" "kNearestNeighbor" "MovingAverage" "Poisson")
echo "Starting hyperparameter tuning; this will occupy significant computational resources"
for model in "${models[@]}"
do
  python -u -W ignore hyperparameter_tuning.py $model 5 &
  pids+=($!)
done
wait_for_pids
python combine_tuning_results.py
echo "hyperparameter tuning done, starting final model training"
for model in "${models[@]}"
do
  python -u -W ignore models.py $model 5 NO_BOOTSTRAP &
  pids+=($!)
done
wait_for_pids
echo "model training done, outputting final results to output/results/"
python performance_display.py > ../../output/results/performance_summary.txt
echo "output done, selecting best model and placing in output/data_appendix_output"
python select_best_model.py delay
python select_best_model.py gse
echo "selection done, testing best model, graphing residuals and feature importance, and placing in output/results"
python test_model.py 5 > ../../output/results/predictor_summary.txt
if [ -z "$1" ]; then
  echo "By default, skipping bootstrap computation and proceeding with pre-computed values."
  echo "Specify COMPUTE_BOOTSTRAP to compute bootstraps from scratch."
  echo "For example, ./master_script.sh NO_ENV COMPUTE_BOOTSTRAP"
  echo "Or ./analysis_script COMPUTE_BOOTSTRAP"
  echo "Be warned this takes very long (often hours) on a standard computer."
else
  echo $1
  python bootstrapping.py
fi
echo "bootstrap computation done, building graphs with confidence intervals"
python bootstrap_display.py > ../../output/results/bootstrap_summary.txt
echo "testing done, analysis script done!"
